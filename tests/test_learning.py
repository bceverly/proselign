# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Learning more of a voice: vocabulary, example passages, scoring, eval."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from proselign.errors import EXIT_OK, EXIT_USAGE, UsageError
from proselign.evaluate import _sample, evaluate, neutralize, report_text
from proselign.inference.base import Backend
from proselign.inference.mock import MockBackend
from proselign.ingest import lexicon as lexicon_mod
from proselign.ingest.examples import select_examples
from proselign.ingest.lexicon import build_lexicon
from proselign.ingest.stylometry import measure
from proselign.rewrite.engine import Options, Rewriter
from proselign.rewrite.exemplars import Library
from proselign.rewrite.markers import extract_input
from proselign.rewrite.prompt import build_messages
from proselign.rewrite.scoring import distance, drift, function_word_rates
from proselign.rewrite.style import lexicon_directives
from proselign.voice.profile import Voice

from .conftest import SAMPLE_PARAGRAPHS, run
from .test_rewrite import ScriptedBackend

# --- vocabulary fingerprint ---------------------------------------------------


def test_lexicon_preferences_openers_and_conventions():
    text = (
        "We use it daily. We use the tool. Then we use it again. So, we used it all week "
        "and nobody minded. So, we keep going with the plan. So, the team agreed on it "
        "today. Red, white and blue. Cats, dogs and birds. Tea, milk and sugar. One, two "
        "and three. Salt, pepper and oil. We organize, we analyze, we prioritize. Bob, "
    )
    lexicon = build_lexicon(text)
    assert {"prefer": "use", "over": "utilize"} in lexicon["prefers"]
    assert lexicon["openers"] == ["So"]
    assert lexicon["conventions"] == {"serial_comma": False, "spelling": "American"}
    assert lexicon["function_words"]["we"] > 50
    assert not lexicon["avoids"]  # too little text to claim the author never writes a word


def test_lexicon_absence_and_british_serial(monkeypatch):
    monkeypatch.setattr(lexicon_mod, "MIN_WORDS_FOR_ABSENCE", 10)
    text = (
        "We organised the colour, the flavour, and the honour. Red, white, and blue. "
        "Cats, dogs, and birds. Tea, milk, and sugar. One, two, and three. Utilize this."
    )
    lexicon = build_lexicon(text)
    assert "utilize" not in lexicon["avoids"] and "delve" in lexicon["avoids"]
    assert lexicon["conventions"] == {"serial_comma": True, "spelling": "British"}
    assert not build_lexicon("")["prefers"]


def test_lexicon_directives():
    text = "\n".join(
        lexicon_directives(
            {
                "prefers": [{"prefer": "use", "over": "utilize"}],
                "avoids": ["delve", "leverage"],
                "openers": ["So"],
                "conventions": {"serial_comma": True, "spelling": "American"},
            }
        )
    )
    for phrase in (
        "'use' (not 'utilize')",
        "delve, leverage",
        "'So,'",
        "serial (Oxford)",
        "American",
    ):
        assert phrase in text
    assert "No serial comma" in "\n".join(
        lexicon_directives({"conventions": {"serial_comma": False}})
    )
    assert not lexicon_directives({})


# --- example passages ---------------------------------------------------------

PROSE = "This is a sentence of eight words in it. " * 6


def test_select_examples_round_robin_dedupe_and_filters(monkeypatch):
    doc_a = "\n\n".join([PROSE.strip(), PROSE.strip(), "Too short.", "No ending " * 10])
    doc_b = "\n\n".join(["Visit https://x.example " + PROSE.strip(), ("Two words. " * 30).strip()])
    numbers = "\n\n".join([("12345 6789 " * 10 + "end.").strip()])
    chosen = select_examples([doc_a, doc_b, numbers], 9.0)
    assert chosen == [PROSE.strip(), ("Two words. " * 30).strip()]
    monkeypatch.setattr("proselign.ingest.examples.MAX_EXAMPLES", 1)
    assert len(select_examples([doc_a, doc_b], 9.0)) == 1


def test_library_prefers_on_topic_passages_and_respects_budget():
    passages = [
        "Gardens need water and sun to grow tomatoes well every summer.",
        "The repository history shows every commit and its author in Git.",
        "Budgets and forecasts guide the quarterly planning of the company.",
    ]
    library = Library(passages)
    assert library.pick("Who wrote this commit in the repository history?", 1) == [passages[1]]
    assert len(library.pick("anything", 3)) == 3
    assert library.pick("anything", 3, max_words=12) == [passages[0]]
    assert not Library([]).pick("x", 2)


def test_examples_reach_the_prompt_as_data():
    user = build_messages(Voice(display_name="B"), "Text.", 0.5, examples=["Mine, truly."])[1]
    assert "Passages the author wrote" in user.content
    assert "PROSELIGN_EXAMPLE_" in user.content and "Mine, truly." in user.content
    assert extract_input(user.content) == "Text."
    assert "Passages" not in build_messages(Voice(display_name="B"), "Text.", 0.5)[1].content


# --- scoring and steering -----------------------------------------------------


def _voice(**metrics) -> Voice:
    base = {"sentence_length_words": {"mean": 10.0}, "mean_word_length": 4.5}
    base["contractions_per_1k"] = metrics.pop("contractions", 12.0)
    base.update(metrics)
    lexicon = build_lexicon(" ".join(SAMPLE_PARAGRAPHS))
    return Voice(display_name="B", metrics=base, lexicon=lexicon, examples=list(SAMPLE_PARAGRAPHS))


def test_distance_is_lower_for_the_authors_own_style():
    voice = _voice()
    mine = " ".join(SAMPLE_PARAGRAPHS)
    stiff = (
        "The organization shall utilize comprehensive methodologies in order to facilitate "
        "the optimization of procedural frameworks across multiple heterogeneous domains. "
    ) * 3
    assert distance(mine, voice) < distance(stiff, voice)
    assert distance("[[1]] x <s1>y</s1>", Voice(display_name="Empty")) >= 0
    assert function_word_rates("the the cat")["the"] == pytest.approx(666.67, rel=1e-3)


def test_drift_feedback():
    voice = _voice()
    long_sentences = ("This sentence keeps going and going with many words in it " * 3 + ". ") * 4
    assert "split the long sentences" in drift(long_sentences, voice, 0.5)[0]
    choppy = "Go now. " * 60
    assert "combine the choppy ones" in drift(choppy, voice, 0.5)[0]
    formal = "We do not know what it is that we will do with this, so we will wait. " * 15
    assert any("use them as the author does" in p for p in drift(formal, voice, 0.5))
    casual_voice = _voice(contractions=1.0)
    casual = "We don't know what it's for, and we can't tell, so we'll wait here. " * 10
    assert any("write contractions out" in p for p in drift(casual, casual_voice, 0.5))
    assert not drift(long_sentences, voice, 0.3)
    assert not drift("Short.", voice, 0.9)
    assert not drift(long_sentences, Voice(display_name="Empty"), 0.9)


def test_engine_shows_examples_and_keeps_the_closest_candidate():
    class Seeded(Backend):
        def __init__(self):
            self.seed = 7
            self.seeds: list[int] = []
            self.prompts = []

        def rewrite(self, messages, temperature):
            self.seeds.append(self.seed)
            self.prompts.append(messages)
            if self.seed == 7:
                return "The organization shall utilize comprehensive methodologies."
            return "I don't like long meetings, so we write things down first."

    backend = Seeded()
    rewriter = Rewriter(_voice(), backend, Options(candidates=2, examples=2, strength=0.3))
    assert rewriter.rewrite("Meetings are long.") == (
        "I don't like long meetings, so we write things down first."
    )
    assert backend.seeds == [7, 8] and backend.seed == 7
    assert "Passages the author wrote" in backend.prompts[0][1].content


def test_engine_drift_nudges_once():
    voice = _voice()
    stiff = ("This sentence keeps going and going with many words in it " * 3 + ".") * 4
    backend = ScriptedBackend([stiff, stiff])
    rewriter = Rewriter(voice, backend, Options(examples=0))
    assert rewriter.rewrite("Plain.") == stiff
    assert "match the author's sentence length" in backend.prompts[1][1].content
    assert not rewriter.report.warnings


def test_candidates_without_a_seed_and_option_bounds():
    backend = ScriptedBackend(["One.", "Two."])
    assert Rewriter(_voice(), backend, Options(candidates=2, examples=0)).rewrite("Uno.") in (
        "One.",
        "Two.",
    )
    for bad in ({"candidates": 0}, {"candidates": 6}, {"examples": -1}, {"examples": 9}):
        with pytest.raises(UsageError):
            Options(**bad)


# --- training, storage, export ------------------------------------------------


def test_training_stores_lexicon_and_examples_privately(trained, tmp_path: Path):
    voice = trained.get("Bryan")
    assert voice.lexicon["function_words"] and voice.examples
    code, out, _ = run(["inspect", "Bryan"], registry=trained)
    assert code == EXIT_OK
    shown = json.loads(out)["examples"]
    assert shown == f"{len(voice.examples)} passages (stored privately)"
    archive = tmp_path / "b.proselign"
    assert run(["export", "Bryan", "-o", str(archive)], registry=trained)[0] == EXIT_OK
    with tarfile.open(archive) as tar:
        exported = json.loads(tar.extractfile("voice.json").read())
    assert exported["examples"] == [] and exported["lexicon"]
    voice.license_and_consent["raw_examples_exportable"] = True
    trained.save(voice)
    assert run(["export", "Bryan", "-o", str(archive)], registry=trained)[0] == EXIT_OK
    with tarfile.open(archive) as tar:
        assert json.loads(tar.extractfile("voice.json").read())["examples"]


# --- eval ---------------------------------------------------------------------


class Flattener(Backend):
    """Neutralizes to stiff prose, then 'restores' to the author's words."""

    def rewrite(self, messages, temperature):
        if "neutral, generic" in messages[0].content:
            return "The individual does not favour lengthy meetings within the organization."
        return extract_input(messages[-1].content) and SAMPLE_PARAGRAPHS[0]


def _measured_voice() -> Voice:
    text = "\n\n".join(SAMPLE_PARAGRAPHS)
    return Voice(
        display_name="B",
        metrics=measure(text),
        lexicon=build_lexicon(text),
        examples=list(SAMPLE_PARAGRAPHS),
    )


def test_evaluate_round_trip_and_report():
    result = evaluate(_measured_voice(), Flattener(), Options(examples=1), samples=3, seed=1)
    assert result["samples"] == 3
    assert result["distance"]["restored"] < result["distance"]["neutral"]
    assert result["gap_closed"] is not None
    text = report_text(result, "test-model")
    assert "model test-model, 3 sample passage(s)" in text and "recovered" in text
    assert "contractions/1k" in result["features_closer"]
    assert "Closer to the author's passages than the neutral version on" in text
    same = evaluate(_voice(), MockBackend(), Options(), samples=2)
    assert same["gap_closed"] is None
    assert "already matched" in report_text(same, "mock")


def test_evaluate_refusals_and_sampling():
    with pytest.raises(UsageError, match="retrain"):
        evaluate(Voice(display_name="Old"), MockBackend(), Options())
    with pytest.raises(UsageError, match="samples"):
        evaluate(_voice(), MockBackend(), Options(), samples=0)
    passages = [f"p{i}" for i in range(10)]
    assert _sample(passages, 3, 1) == _sample(passages, 3, 1) != _sample(passages, 3, 2)
    assert neutralize(MockBackend(), "Echo me.") == "Echo me."


def test_cli_eval(trained):
    code, out, err = run(
        ["eval", "--voice", "Bryan", "--backend", "mock", "--samples", "2"], registry=trained
    )
    assert code == EXIT_OK, err
    assert "Voice Bryan, model mock" in out
    code, out, _ = run(
        ["eval", "--voice", "Bryan", "--backend", "mock", "--json"], registry=trained
    )
    assert json.loads(out)["voice"] == "Bryan"
    assert run(["eval", "--backend", "mock", "--no-config"], registry=trained)[0] == EXIT_USAGE


def test_cli_examples_and_candidates_flags(trained, tmp_path: Path, monkeypatch):
    argv = ["--voice", "Bryan", "--backend", "mock", "--examples", "0", "--candidates", "2"]
    assert run(argv, "Hello there.\n", registry=trained)[1] == "Hello there.\n"
    assert run(argv[:-1] + ["9"], "x", registry=trained)[0] == EXIT_USAGE
    rc = tmp_path / "rc"
    rc.write_text("[rewrite]\nexamples = 2\ncandidates = 3\n", "utf-8")
    rc.chmod(0o600)
    monkeypatch.setenv("PROSELIGN_CONFIG", str(rc))
    from proselign import config as config_mod  # pylint: disable=import-outside-toplevel

    settings = config_mod.load()
    assert (settings.examples, settings.candidates) == (2, 3)
    rc.write_text("[rewrite]\nexamples = many\n", "utf-8")
    with pytest.raises(UsageError, match="examples"):
        config_mod.load()
