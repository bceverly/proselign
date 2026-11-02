# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""proselign tune: training pairs, the adapter's conversion and registration,
and rewrites that switch to the tuned model."""

from __future__ import annotations

import http.server
import io
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from proselign import cli
from proselign.errors import UsageError
from proselign.evaluate import HELD_OUT, Plan, evaluate, untrained_passages
from proselign.inference.base import Backend
from proselign.progress import NoProgress
from proselign.rewrite.engine import Options, Rewriter
from proselign.rewrite.markers import extract_input
from proselign.rewrite.prompt import TUNED_RULES
from proselign.tune import command, export, pairs, trainer
from proselign.voice.profile import Voice

from .conftest import SAMPLE_PARAGRAPHS


class Flattens(Backend):
    """Writes a 'neutral' version that keeps every placeholder."""

    def __init__(self, drop: bool = False, pad: bool = False) -> None:
        self.model, self.endpoint = "qwen2.5:14b-instruct", "http://127.0.0.1:9/v1"
        self.drop, self.pad, self.calls = drop, pad, 0

    def rewrite(self, messages, temperature):
        self.calls += 1
        text = extract_input(messages[-1].content).replace("I'd", "I would")
        if self.drop:
            text = re.sub(r"\{\{\d+\}\}", "", text)
        return text + (" padding" * 400 if self.pad else "")


def test_passages_join_short_paragraphs_and_cut_long_ones():
    short = "One two three four five six seven eight nine ten."
    long = " ".join(f"Sentence {n} has exactly seven words here." for n in range(60))
    found = pairs.passages([f"{short}\n\n" * 5, long])
    assert found[0].count(short) == 5 and "\n\n" in found[0]
    assert all(pairs.MIN_WORDS <= len(p.split()) <= pairs.MAX_WORDS for p in found)
    assert sum(len(p.split()) for p in found[1:]) == len(long.split())
    assert not pairs.passages(["Too short."])


def test_a_pair_keeps_the_facts_masked_on_both_sides():
    passage = "In 2011 we sent 42 letters. " * 8
    made = pairs.neutral_pair(Flattens(), passage)
    assert made is not None
    assert made["target"].startswith("In {{1}} we sent {{2}} letters.")
    assert "{{2}}" in made["source"] and "2011" not in made["source"]
    assert pairs.neutral_pair(Flattens(drop=True), passage) is None
    assert pairs.neutral_pair(Flattens(pad=True), passage) is None


def test_pairs_are_kept_and_reused(tmp_path: Path):
    todo = ["Plain words, written plainly, for a plain purpose. " * 6]
    kept: dict = {}
    backend = Flattens()
    made, failed = pairs.build_pairs(backend, todo, kept, parallel=2)
    assert (len(made), failed, backend.calls) == (1, 0, 1)
    path = tmp_path / pairs.PAIRS_FILE
    pairs.save_pairs(path, kept)
    assert path.stat().st_mode & 0o777 == 0o600
    again = pairs.load_pairs(path)
    assert pairs.build_pairs(backend, todo, again)[0] == made and backend.calls == 1
    _, failed = pairs.build_pairs(Flattens(drop=True), ["{{x}} 1 2 3 " * 20], {})
    assert failed == 1
    path.write_text("[1, 2]", "utf-8")
    assert pairs.load_pairs(path) == {}
    path.write_text("not json", "utf-8")
    assert pairs.load_pairs(path) == {}
    assert pairs.load_pairs(tmp_path / "missing.json") == {}


def test_adapter_tensor_names_and_architectures():
    peft = "base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight"
    assert export.gguf_name(peft) == "blk.7.attn_q.weight.lora_a"
    assert export.gguf_name(peft.replace("self_attn.q", "mlp.down").replace("_A", "_B")) == (
        "blk.7.ffn_down.weight.lora_b"
    )
    assert export.gguf_name("base_model.model.lm_head.lora_A.weight") is None
    assert export.architecture({"model_type": "qwen2"}) == "qwen2"
    assert export.architecture({"model_type": "mistral"}) == "llama"
    with pytest.raises(UsageError, match="gpt2"):
        export.architecture({"model_type": "gpt2"})


def test_adapter_converts_to_gguf(tmp_path: Path):
    numpy = pytest.importorskip("numpy")
    safe = pytest.importorskip("safetensors.numpy")
    gguf = pytest.importorskip("gguf")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"lora_alpha": 32}), "utf-8")
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    safe.save_file(
        {
            f"{prefix}.lora_A.weight": numpy.ones((4, 8), dtype=numpy.float32),
            f"{prefix}.lora_B.weight": numpy.zeros((8, 4), dtype=numpy.float32),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    out = export.to_gguf(adapter, {"model_type": "qwen2"}, tmp_path / "adapter.gguf")
    reader = gguf.GGUFReader(str(out))
    names = sorted(t.name for t in reader.tensors)
    assert names == ["blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"]
    alpha = reader.fields["adapter.lora.alpha"]
    assert float(alpha.parts[alpha.data[0]][0]) == 32.0
    assert out.stat().st_mode & 0o777 == 0o600
    safe.save_file(
        {"base_model.model.lm_head.lora_A.weight": numpy.ones((2, 2), dtype=numpy.float32)},
        str(adapter / "adapter_model.safetensors"),
    )
    with pytest.raises(UsageError, match="does not know"):
        export.to_gguf(adapter, {"model_type": "qwen2"}, tmp_path / "bad.gguf")


class FakeOllama(http.server.BaseHTTPRequestHandler):
    """Just enough of Ollama's API: blobs, create, ps and unloading."""

    seen: list[tuple[str, bytes]] = []
    fail_create = False

    def do_GET(self):  # pylint: disable=invalid-name  # the http.server API
        self.seen.append(("GET " + self.path, b""))
        self._reply({"models": [{"name": "qwen2.5:14b-instruct"}]})

    def do_POST(self):  # pylint: disable=invalid-name
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.seen.append(("POST " + self.path, body))
        if self.path == "/api/create" and self.fail_create:
            self._reply({"error": "no such base"})
        else:
            self._reply({"status": "success"})

    def _reply(self, data: dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def ollama():
    FakeOllama.seen, FakeOllama.fail_create = [], False
    server = http.server.HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_the_adapter_is_registered_through_the_api(ollama, tmp_path: Path):
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"GGUF-bytes")
    export.register("proselign-bryan", "qwen2.5:14b-instruct", adapter, ollama)
    (blob, data), (create, spec) = FakeOllama.seen  # pylint: disable=unbalanced-tuple-unpacking
    assert blob.startswith("POST /api/blobs/sha256:") and data == b"GGUF-bytes"
    request = json.loads(spec)
    assert create == "POST /api/create"
    assert request["from"] == "qwen2.5:14b-instruct"
    assert request["adapters"] == {"adapter.gguf": blob.split("/")[-1]}
    assert request["parameters"]["num_ctx"] == export.CONTEXT_TOKENS
    FakeOllama.fail_create = True
    with pytest.raises(UsageError, match="no such base"):
        export.register("proselign-bryan", "qwen2.5:14b-instruct", adapter, ollama)
    with pytest.raises(UsageError, match="valid Ollama model name"):
        export.register("Bad Name!", "qwen2.5:14b-instruct", adapter, ollama)
    with pytest.raises(UsageError, match="cannot reach"):
        export.register("proselign-bryan", "qwen", adapter, "http://127.0.0.1:9/v1")


def test_loaded_models_are_unloaded_before_training(ollama):
    assert export.unload_models(ollama) == ["qwen2.5:14b-instruct"]
    unload = json.loads(FakeOllama.seen[-1][1])
    assert unload == {"model": "qwen2.5:14b-instruct", "keep_alive": 0}
    assert export.unload_models("http://127.0.0.1:9/v1") == []
    with pytest.raises(UsageError):
        export.ollama_root("file:///etc/passwd")


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


def test_examples_learn_only_the_authors_passage():
    pair = {"source": "Plain {{1}}.", "target": "Mine {{1}}."}
    messages, target = trainer.chat(pair)
    assert messages[0]["content"] == TUNED_RULES and target == "Mine {{1}}."
    ids, labels = trainer.encode(FakeTokenizer(), pair, 10_000)
    reply = "Mine {{1}}.<|im_end|>"
    assert labels[-len(reply) :] == [ord(c) for c in reply]
    assert set(labels[: -len(reply)]) == {-100}
    assert len(ids) == len(labels)
    assert len(trainer.encode(FakeTokenizer(), pair, 50)[0]) == 50
    for bad in ({"epochs": 0}, {"epochs": 11}, {"rank": 12}):
        with pytest.raises(UsageError):
            trainer.Settings(**bad)


def test_missing_training_extras_are_named(monkeypatch):
    monkeypatch.setattr(trainer, "find_spec", lambda name: None if name == "peft" else object())
    with pytest.raises(UsageError, match="needs peft"):
        trainer.require_extras()


def _tune_args(**overrides):
    defaults = {
        "epochs": 2,
        "rank": 16,
        "pairs_only": False,
        "name": None,
        "base": str(Path(__file__).parent / "no-such-model"),
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _long_voice(trained, tmp_path: Path) -> Voice:
    """The trained voice, with a source long enough for many passages."""
    voice = trained.get("Bryan")
    source = tmp_path / "long.md"
    notes = (
        f"Note {n} says the plan for week {n} is to write the design down first, then talk. " * 4
        for n in range(30)
    )
    source.write_text("\n\n".join(notes), "utf-8")
    voice.sources = [{"path": str(source)}]
    trained.save(voice)
    return voice


def test_tune_runs_every_step_and_records_the_model(trained, tmp_path: Path, monkeypatch):
    voice = _long_voice(trained, tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type": "qwen2"}', "utf-8")
    steps: list[str] = []
    monkeypatch.setattr(trainer, "require_extras", lambda: steps.append("extras"))

    def fake_train(made, base_dir, out_dir, settings, on_step):
        steps.append(f"train {len(made)}")
        on_step(1, 1, 2.5)
        on_step(1, 1, 1.25)
        return out_dir

    monkeypatch.setattr(trainer, "train", fake_train)
    monkeypatch.setattr(export, "unload_models", lambda endpoint: ["qwen2.5:14b-instruct"])
    monkeypatch.setattr(export, "to_gguf", lambda adapter, config, out: out)
    monkeypatch.setattr(
        export, "register", lambda name, base_model, adapter, endpoint: steps.append(name)
    )
    monkeypatch.setattr(command, "MIN_PAIRS", 1)
    said: list[str] = []
    args = _tune_args(base=str(base))
    command.run(args, trained, voice, Options(), Flattens(), said.append, NoProgress())
    assert steps[0] == "extras" and steps[1].startswith("train ") and steps[2] == "proselign-bryan"
    assert any("loss 2.500 at the start, 1.250" in line for line in said)
    assert any("unloaded qwen2.5:14b-instruct" in line for line in said)
    saved = trained.get("Bryan").tuned
    assert saved["model"] == "proselign-bryan" and saved["base_model"] == "qwen2.5:14b-instruct"
    assert (trained.path_for("Bryan") / pairs.PAIRS_FILE).is_file()


def test_tune_pairs_only_and_its_limits(trained, tmp_path: Path, monkeypatch):
    voice = _long_voice(trained, tmp_path)
    said: list[str] = []
    command.run(
        _tune_args(pairs_only=True),
        trained,
        voice,
        Options(),
        Flattens(),
        said.append,
        NoProgress(),
    )
    assert said[-1].endswith("left out (the model lost a fact)")
    monkeypatch.setattr(trainer, "require_extras", lambda: None)
    with pytest.raises(UsageError, match="training pairs"):
        command.run(
            _tune_args(), trained, voice, Options(), Flattens(drop=True), said.append, NoProgress()
        )
    with pytest.raises(UsageError, match="model server"):
        command.run(
            _tune_args(),
            trained,
            voice,
            Options(),
            Backend.__new__(type("B", (Flattens,), {"model": None})),
            said.append,
            NoProgress(),
        )
    (tmp_path / "empty.md").write_text("", "utf-8")
    voice.sources = [{"path": str(tmp_path / "empty.md")}]
    with pytest.raises(UsageError, match="can be read"):
        command.training_passages(voice)


def test_rewrites_switch_to_the_tuned_model(trained):
    voice = trained.get("Bryan")
    voice.tuned = {"model": "proselign-bryan"}
    trained.save(voice)
    parser = cli.build_parser()
    io_ = {"stderr": io.StringIO()}
    args = parser.parse_args(["rewrite", "--voice", "Bryan", "--endpoint", "http://127.0.0.1:9/v1"])
    _, _, options, backend = cli._setup(args, trained, io_)  # pylint: disable=protected-access
    assert (backend.model, options.tuned) == ("proselign-bryan", True)
    for extra in (["--no-tuned"], ["--model", "other"]):
        args = parser.parse_args(["rewrite", "--voice", "Bryan", *extra])
        _, _, options, backend = cli._setup(args, trained, io_)  # pylint: disable=protected-access
        assert not options.tuned and backend.model != "proselign-bryan"


def test_a_tuned_rewrite_sends_the_short_prompt():
    class Echo(Backend):
        def __init__(self):
            self.prompts = []

        def rewrite(self, messages, temperature):
            self.prompts.append(messages)
            return extract_input(messages[-1].content)

    voice = Voice(display_name="B", examples=list(SAMPLE_PARAGRAPHS))
    backend = Echo()
    Rewriter(voice, backend, Options(tuned=True)).rewrite("We sent 42 letters in 2011.")
    system, user = backend.prompts[0]
    assert system.content == TUNED_RULES
    assert "Passages the author wrote" not in user.content and "{{1}}" in user.content


def test_a_tuned_voice_is_evaluated_on_passages_it_never_saw():
    voice = Voice(
        display_name="B",
        examples=[f"{p} Variant {n}." for n in range(15) for p in SAMPLE_PARAGRAPHS],
    )
    held = untrained_passages(voice)
    assert len(held) == HELD_OUT
    with pytest.raises(UsageError, match="never|not trained"):
        evaluate(voice, Flattens(), Options(tuned=True), Plan(HELD_OUT + 1))
    result = evaluate(voice, Flattens(), Options(tuned=True), Plan(2, control=False))
    assert result["samples"] == 2
