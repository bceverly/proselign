# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The ``proselign tune`` command: pairs, training, export, registration."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

from proselign.errors import UsageError
from proselign.evaluate import untrained_passages
from proselign.inference.base import Backend
from proselign.ingest.corpus import load_documents
from proselign.paths import ensure_private_dir
from proselign.rewrite.engine import Options
from proselign.tune import export, trainer
from proselign.tune.pairs import PAIRS_FILE, build_pairs, load_pairs, passages, save_pairs
from proselign.voice.profile import Voice
from proselign.voice.registry import Registry

MIN_PAIRS = 50
DEFAULT_BASE = "Qwen/Qwen2.5-14B-Instruct"


def tuned_name(voice: Voice) -> str:
    """The Ollama model name for a voice's fine-tune."""
    return f"proselign-{voice.voice_id.replace('_', '-')}"


def training_passages(voice: Voice, on_skip=None) -> list[str]:
    """The voice's passages to train on: from its source documents, without
    the passages ``eval`` holds out."""
    sources = [Path(s["path"]) for s in voice.sources if "path" in s]
    documents = load_documents(sources, on_skip, int(voice.corpus_summary.get("min_words", 0)))
    if not documents:
        raise UsageError(
            f"none of voice {voice.display_name!r}'s source documents can be read any more; "
            "retrain the voice from them first"
        )
    held = untrained_passages(voice)
    return [
        passage
        for passage in passages(doc.text for doc in documents)
        if not any(example in passage for example in held)
    ]


def base_config(base: str) -> dict:
    """The base model's ``config.json`` (a directory, or a Hugging Face id)."""
    local = Path(base) / "config.json"
    if local.is_file():
        return json.loads(local.read_text("utf-8"))
    transformers = import_module("transformers")  # pragma: no cover - training extras
    return transformers.AutoConfig.from_pretrained(base).to_dict()  # pragma: no cover


def run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    args, registry: Registry, voice: Voice, options: Options, backend: Backend, say, progress
) -> None:
    """Every step of ``proselign tune`` for ``voice``; ``say`` reports a step."""
    settings = trainer.Settings(epochs=args.epochs, rank=args.rank)
    if not args.pairs_only:
        trainer.require_extras()  # before an hour of pairs, not after
    base_model = getattr(backend, "model", None)
    endpoint = getattr(backend, "endpoint", None)
    if not base_model or not endpoint:
        raise UsageError("tune needs a model server (--backend openai-compat) and --model")
    directory = registry.path_for(voice.display_name)
    made = _pairs(voice, backend, options.parallel, directory, say, progress)
    if args.pairs_only:
        return
    if len(made) < MIN_PAIRS:
        raise UsageError(
            f"only {len(made)} training pairs; at least {MIN_PAIRS} are needed "
            "(train the voice on more writing)"
        )
    for model in export.unload_models(endpoint):
        say(f"unloaded {model} to free the GPU for training")
    adapter = export.to_gguf(
        _train(made, args.base, directory, settings, say, progress),
        base_config(args.base),
        directory / export.ADAPTER_FILE,
    )
    name = args.name or tuned_name(voice)
    export.register(name, base_model, adapter, endpoint)
    voice.tuned = {
        "model": name,
        "base_model": base_model,
        "base": args.base,
        "pairs": len(made),
        "epochs": settings.epochs,
        "rank": settings.rank,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    registry.save(voice)
    say(
        f"model {name} is ready: rewrites in voice {voice.display_name!r} now use it "
        "(--no-tuned to compare without it)"
    )


def _pairs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    voice: Voice, backend: Backend, parallel: int, directory: Path, say, progress
) -> list[dict[str, str]]:
    """The training pairs, made (or reused) and kept with the voice."""
    todo = training_passages(voice, lambda path, why: say(f"warning: skipped {path}: {why}"))
    kept = load_pairs(directory / PAIRS_FILE)
    say(f"pairing {len(todo)} passages with neutral versions written by {backend.model}")
    progress.begin(len(todo), "passages")
    try:
        made, failed = build_pairs(backend, todo, kept, parallel, progress.advance)
    finally:
        progress.close()
        save_pairs(directory / PAIRS_FILE, kept)
    say(f"{len(made)} training pairs; {failed} passage(s) left out (the model lost a fact)")
    return made


def _train(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    made: list[dict[str, str]], base: str, directory: Path, settings, say, progress
) -> Path:
    """Train the adapter into the voice's directory, reporting progress."""
    adapter_dir = ensure_private_dir(directory / "adapter")
    steps = math.ceil(len(made) / settings.accumulate) * settings.epochs
    say(f"training a rank-{settings.rank} adapter on {base}: {steps} steps")
    progress.begin(steps, "steps")
    losses: list[float] = []

    def on_step(_done: int, _total: int, loss: float) -> None:
        losses.append(loss)
        progress.advance()

    try:
        trainer.train(made, base, adapter_dir, settings, on_step)
    finally:
        progress.close()
    if losses:
        say(f"training loss {losses[0]:.3f} at the start, {losses[-1]:.3f} at the end")
    return adapter_dir
