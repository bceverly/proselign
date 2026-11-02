# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Learning more of a voice: vocabulary, example passages, scoring, eval."""

from __future__ import annotations

import json
import stat
import tarfile
from pathlib import Path

import pytest

from proselign.errors import EXIT_OK, EXIT_USAGE, UsageError
from proselign.evaluate import (
    MAX_REPEATS,
    NEUTRAL_FILE,
    Plan,
    _control_lines,
    _part_verdict,
    _sample,
    evaluate,
    load_neutral,
    neutralize,
    present_parts,
    report_text,
    save_neutral,
)
from proselign.inference.base import Backend
from proselign.inference.mock import MockBackend
from proselign.ingest import lexicon as lexicon_mod
from proselign.ingest.examples import select_examples
from proselign.ingest.lexicon import build_lexicon
from proselign.ingest.stylometry import measure
from proselign.rewrite.engine import PROMPT_PARTS, Options, Rewriter
from proselign.rewrite.exemplars import Library
from proselign.rewrite.markers import extract_input
from proselign.rewrite.prompt import build_messages, style_summary
from proselign.rewrite.scoring import distance, distance_parts, drift, function_word_rates
from proselign.rewrite.style import StyleRules, directives, lexicon_directives
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
    assert "utilize" not in lexicon["avoids"]
    assert "delve" in lexicon["avoids"]
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
    assert "PROSELIGN_EXAMPLE_" in user.content
    assert "Mine, truly." in user.content
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
    assert "split the long sentences" in drift(long_sentences, voice, 0.8)[0]
    choppy = "Go now. " * 60
    assert "combine the choppy ones" in drift(choppy, voice, 0.8)[0]
    formal = "We do not know what it is that we will do with this, so we will wait. " * 15
    assert any("use them as the author does" in p for p in drift(formal, voice, 0.8))
    casual_voice = _voice(contractions=1.0)
    casual = "We don't know what it's for, and we can't tell, so we'll wait here. " * 10
    assert any("write contractions out" in p for p in drift(casual, casual_voice, 0.8))
    assert not drift(long_sentences, voice, 0.5)
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
                return "Our organizational meetings are, I find, considerably lengthy."
            return "I'd say our meetings run long."

    backend = Seeded()
    voice = _voice()
    rewriter = Rewriter(voice, backend, Options(candidates=2, examples=2, strength=0.3))
    both = (
        "Our organizational meetings are, I find, considerably lengthy.",
        "I'd say our meetings run long.",
    )
    closest = min(both, key=lambda text: distance(text, voice))
    assert distance(both[0], voice) != distance(both[1], voice)
    assert rewriter.rewrite("I find that our meetings are long.") == closest
    assert backend.seeds == [7, 8]
    assert backend.seed == 7
    assert "Passages the author wrote" in backend.prompts[0][1].content


def test_engine_drift_nudges_once():
    voice = _voice()
    stiff = ("This sentence keeps going and going with many words in it " * 3 + ".") * 4
    backend = ScriptedBackend([stiff, stiff])
    rewriter = Rewriter(voice, backend, Options(examples=0, strength=0.8))
    source = ("This sentence keeps going with many words. " * 15).strip()
    assert rewriter.rewrite(source) == stiff
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
    assert voice.lexicon["function_words"]
    assert voice.examples
    code, out, _ = run(["inspect", "Bryan"], registry=trained)
    assert code == EXIT_OK
    shown = json.loads(out)["examples"]
    assert shown == f"{len(voice.examples)} passages (stored privately)"
    archive = tmp_path / "b.proselign"
    assert run(["export", "Bryan", "-o", str(archive)], registry=trained)[0] == EXIT_OK
    with tarfile.open(archive) as tar:
        exported = json.loads(tar.extractfile("voice.json").read())
    assert exported["examples"] == []
    assert exported["lexicon"]
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
            return (
                "The individual does not favour lengthy meetings within the organization "
                "and prefers written documentation prior to discussion."
            )
        return extract_input(messages[-1].content) and SAMPLE_PARAGRAPHS[0]


class PlainControl(Flattener):
    """Like Flattener, but the control (a rewrite with no voice) changes nothing."""

    def rewrite(self, messages, temperature):
        if "Author: Original" in messages[-1].content:
            return extract_input(messages[-1].content)
        return super().rewrite(messages, temperature)


def _measured_voice() -> Voice:
    text = "\n\n".join(SAMPLE_PARAGRAPHS)
    return Voice(
        display_name="B",
        metrics=measure(text),
        lexicon=build_lexicon(text),
        examples=list(SAMPLE_PARAGRAPHS),
    )


def test_evaluate_round_trip_and_report():
    result = evaluate(_measured_voice(), Flattener(), Options(examples=1), Plan(3, seed=1))
    assert result["samples"] == 3
    assert result["distance"]["restored"] < result["distance"]["neutral"]
    assert result["gap_closed"] is not None
    text = report_text(result, "test-model")
    assert "model test-model, 3 sample passage(s)" in text
    assert "recovered" in text
    assert "contractions/1k" in result["features_closer"]
    assert "Closer to the author's passages than the neutral version on" in text
    same = evaluate(_voice(), MockBackend(), Options(), Plan(2))
    assert same["gap_closed"] is None
    assert "already matched" in report_text(same, "mock")
    assert same["rules_set_aside"] == []
    assert "Set aside" not in report_text(same, "mock")


def test_evaluate_compares_the_voice_with_a_control():
    result = evaluate(_measured_voice(), PlainControl(), Options(examples=1), Plan(3, seed=1))
    assert result["distance"]["control"] == result["distance"]["neutral"]
    assert result["voice_gain"] > 0
    assert "contractions/1k" in result["features_closer_than_control"]
    text = report_text(result, "m")
    assert "rewritten, no voice (control)" in text
    assert "closer to the author than the control" in text
    # When every rewrite says the same thing, the voice adds nothing.
    same = evaluate(_measured_voice(), Flattener(), Options(examples=1), Plan(3, seed=1))
    assert same["voice_gain"] == 0
    assert "the voice instructions are not helping" in report_text(same, "m")
    skipped = evaluate(_measured_voice(), Flattener(), Options(), Plan(3, control=False))
    assert "control" not in skipped["distance"]
    assert "voice_gain" not in skipped
    assert "control" not in report_text(skipped, "m")


class Wavering(PlainControl):
    """The voiced rewrite depends on the sampling seed: the author's own words
    on one run, stiffer prose than the control's on the next."""

    seed = 0

    def rewrite(self, messages, temperature):
        if "neutral, generic" in messages[0].content or "Author: Original" in messages[-1].content:
            return super().rewrite(messages, temperature)
        if self.seed % 2000:
            return (
                "Notwithstanding considerable organizational heterogeneity, comprehensive "
                "documentation requirements necessitate extraordinarily meticulous "
                "administrative coordination throughout institutional hierarchies."
            )
        return SAMPLE_PARAGRAPHS[0]


def test_distance_parts_add_up_to_the_distance():
    voice = _measured_voice()
    text = "We shipped it. It's done, and the team can rest now."
    parts = distance_parts(text, voice)
    assert set(parts) == {"function words", "sentence length", "word length", "contractions"}
    assert round(sum(parts.values()), 4) == distance(text, voice)
    assert set(distance_parts(text, Voice(display_name="Empty"))) == set()


def test_evaluate_repeats_report_the_average_and_spread():
    backend = Wavering()
    result = evaluate(_measured_voice(), backend, Options(examples=1), Plan(2, repeats=2))
    assert backend.seed == 0  # restored after the repeats
    assert result["runs"] == 2
    low, high = result["voice_gain_range"]
    assert low < 0 < high
    assert result["distance_range"]["neutral"][0] == result["distance_range"]["neutral"][1]
    for key, parts in result["parts"].items():
        assert abs(sum(parts.values()) - result["distance"][key]) < 0.001
    text = report_text(result, "m")
    assert "2 runs (averaged)" in text
    assert "Where the distance comes from" in text
    assert "across the runs" in text
    assert "the difference is within the noise" in text
    with pytest.raises(UsageError, match="repeats"):
        evaluate(_measured_voice(), backend, Options(), Plan(1, repeats=MAX_REPEATS + 1))


class SentenceSensitive(PlainControl):
    """The voiced rewrite comes out stiff whenever the prompt mentions
    sentence length, and as the author wrote it otherwise."""

    def rewrite(self, messages, temperature):
        prompt = messages[-1].content
        if "neutral, generic" in messages[0].content or "Author: Original" in prompt:
            return super().rewrite(messages, temperature)
        if "words per sentence" in prompt:
            return (
                "Notwithstanding considerable organizational heterogeneity, comprehensive "
                "documentation requirements necessitate extraordinarily meticulous "
                "administrative coordination throughout institutional hierarchies."
            )
        return SAMPLE_PARAGRAPHS[0]


def test_evaluate_ablate_finds_the_part_that_hurts():
    voice, options = _measured_voice(), Options(examples=1, strength=0.8)
    result = evaluate(voice, SentenceSensitive(), options, Plan(2, ablate=True))
    assert list(result["ablation"]) == present_parts(voice, options)
    assert {"sentences", "tone", "examples", "steering"} <= set(result["ablation"])
    assert result["ablation"]["sentences"]["effect"] < 0
    assert result["ablation"]["tone"]["effect"] == 0
    text = report_text(result, "m")
    assert "What each part of the voice does" in text
    lines = text.splitlines()
    first = lines.index(next(line for line in lines if line.startswith("part of the voice")))
    assert lines[first + 1].startswith("sentences") and lines[first + 1].endswith("hurts")
    assert "--repeats 2 or more" in text
    twice = evaluate(voice, SentenceSensitive(), options, Plan(2, repeats=2, ablate=True))
    assert "+0% to +0%  no effect" in report_text(twice, "m")
    assert "ablation" not in evaluate(voice, SentenceSensitive(), options, Plan(2))


def test_present_parts_and_verdicts():
    assert not present_parts(Voice(display_name="Empty"), Options())
    quiet = present_parts(_measured_voice(), Options(examples=0, strength=0.2))
    assert "examples" not in quiet and "steering" not in quiet
    assert set(present_parts(_measured_voice(), Options())) <= set(PROMPT_PARTS)
    assert _part_verdict(None, None, 1) == "unknown"
    assert _part_verdict(0.1, [-0.1, 0.2], 2) == "within the noise"
    assert _part_verdict(0.1, [0.05, 0.2], 2) == "helps"
    assert _part_verdict(0.0, None, 1) == "no effect"


def test_omitted_parts_leave_the_prompt():
    voice = _measured_voice()
    full = "\n".join(directives(voice, StyleRules()))
    assert "words per sentence" in full
    assert "words per sentence" not in "\n".join(
        directives(voice, StyleRules(), frozenset({"sentences"}))
    )
    assert "Overall tone" in style_summary(voice)
    assert "Overall tone" not in style_summary(voice, omit=frozenset({"tone"}))
    with pytest.raises(UsageError, match="unknown part"):
        Options(omit=frozenset({"rhythm"}))
    seen = []

    class Recording(MockBackend):
        def rewrite(self, messages, temperature):
            seen.append(messages[-1].content)
            return super().rewrite(messages, temperature)

    Rewriter(voice, Recording(), Options(examples=2, omit=frozenset({"examples"}))).rewrite(
        "Short text here."
    )
    assert "Passages the author wrote" not in seen[0]


def test_control_lines_with_consistent_runs():
    result = {
        "voice_gain": 0.1,
        "voice_gain_range": [0.05, 0.15],
        "features_closer_than_control": ["word length"],
    }
    assert "closer to the author than the control" in _control_lines(result, 3)[1]
    result.update(voice_gain=-0.1, voice_gain_range=[-0.15, -0.05])
    assert "from -15% to -5% across the runs" in _control_lines(result, 3)[1]


def test_control_lines_when_the_control_already_matches():
    lines = _control_lines({"voice_gain": None, "features_closer_than_control": []})
    assert lines == [
        "Closer to the author's passages than the control on 0 of 5 features.",
        "The control already matched the author's passages.",
    ]


def test_evaluate_sets_aside_rules_that_depart_from_the_voice():
    rules = StyleRules(passive_voice=False, first_person="plural", em_dashes=False)
    seen = []

    class Recording(MockBackend):
        def rewrite(self, messages, temperature):
            seen.append(messages[0].content)
            return super().rewrite(messages, temperature)

    result = evaluate(_voice(), Recording(), Options(rules=rules), Plan(1))
    assert result["rules_set_aside"] == ["passive_voice = off", "first_person = plural"]
    assert "Set aside for this measurement" in report_text(result, "mock")
    rewrite_prompts = [p for p in seen if "first person plural" in p or "active voice only" in p]
    assert not rewrite_prompts


def test_evaluate_refusals_and_sampling():
    old, voice, backend, options = Voice(display_name="Old"), _voice(), MockBackend(), Options()
    with pytest.raises(UsageError, match="retrain"):
        evaluate(old, backend, options)
    with pytest.raises(UsageError, match="samples"):
        evaluate(voice, backend, options, Plan(0))
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


def test_cli_eval_keeps_and_reuses_neutral_versions(trained):
    argv = ["eval", "--voice", "Bryan", "--backend", "mock", "--samples", "2", "--json"]
    kept = trained.path_for("Bryan") / NEUTRAL_FILE
    first = json.loads(run(argv, registry=trained)[1])
    assert first["neutral_reused"] == 0
    assert first["neutral_models"] == ["mock"]
    assert stat.S_IMODE(kept.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    saved = load_neutral(kept)
    assert len(saved) == 2
    # A later run, with any model, starts from the same neutral texts.
    for entry in saved.values():
        entry["model"] = "earlier-model"
    save_neutral(kept, saved)
    code, out, _ = run(argv[:-1], registry=trained)
    assert code == EXIT_OK
    assert "Neutral versions written by earlier-model (2 kept from an earlier run)." in out
    twice = json.loads(run(argv + ["--repeats", "2"], registry=trained)[1])
    assert twice["runs"] == 2
    quick = json.loads(run(argv + ["--no-control"], registry=trained)[1])
    assert "voice_gain" not in quick
    ablated = json.loads(run(argv + ["--ablate"], registry=trained)[1])
    assert "sentences" in ablated["ablation"]
    fresh = json.loads(run(argv + ["--fresh"], registry=trained)[1])
    assert fresh["neutral_reused"] == 0
    assert {e["model"] for e in load_neutral(kept).values()} == {"mock"}


def test_load_neutral_tolerates_damage(tmp_path: Path):
    path = tmp_path / NEUTRAL_FILE
    assert not load_neutral(path)
    path.write_text("not json", "utf-8")
    assert not load_neutral(path)
    path.write_text("[1, 2]", "utf-8")
    assert not load_neutral(path)
    good = {"text": "Plain.", "model": "m"}
    path.write_text(json.dumps({"a": good, "b": {"text": 3}, "c": "x"}), "utf-8")
    assert load_neutral(path) == {"a": good}
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert not load_neutral(link)


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
