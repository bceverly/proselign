# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Sentence by sentence review: alignment, repairs, checks and the checker."""

from __future__ import annotations

import pytest

from proselign.errors import ValidationError
from proselign.inference.base import Backend
from proselign.rewrite import review
from proselign.rewrite.engine import Options, Rewriter
from proselign.rewrite.markers import extract_input
from proselign.rewrite.prompt import build_messages, build_tuned_messages
from proselign.rewrite.style import StyleRules
from proselign.voice.profile import Voice

from .conftest import run

VOICE = Voice(
    display_name="Bryan",
    metrics={"sentence_length_words": {"mean": 10, "stdev": 3}, "contractions_per_1k": 20},
)
ACTIVE = StyleRules(passive_voice=False)


class Edits(Backend):
    """Applies ``changes`` (old -> new) to the text it is sent."""

    def __init__(self, changes: dict[str, str]) -> None:
        self.changes = changes
        self.inputs: list[str] = []
        self.prompts: list[str] = []

    def rewrite(self, messages, temperature):
        self.prompts.append(messages[-1].content)
        text = extract_input(messages[-1].content)
        self.inputs.append(text)
        for old, new in self.changes.items():
            text = text.replace(old, new)
        return text


class Judge(Backend):
    """Says DIFFERENT for any change whose edited side contains ``bad``."""

    def __init__(self, bad: str = "\0") -> None:
        self.bad = bad
        self.asked: list[str] = []

    def rewrite(self, messages, temperature):
        self.asked.append(messages[-1].content)
        edited = messages[-1].content.split("EDITED:")[1]
        return "DIFFERENT: who did it" if self.bad in edited else "SAME"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "One.",
        "One. Two!  Three?\n\n- four\n- five",
        'He said "no." Then left.\n',
        "  lead",
        "A. \n\nB. \n\n",
    ],
)
def test_units_join_back_into_the_text(text):
    assert "".join(review.units(text)) == text


def test_units_split_sentences_and_lines():
    assert review.units("One. Two.\n- three\n\nFour") == ["One. ", "Two.\n", "- three\n\n", "Four"]


def test_align_pairs_reworded_dropped_and_added_sentences():
    groups = review.align(
        "Alpha one here. Beta two here. Gamma.", "Alpha one there. Gamma. Delta new."
    )
    assert [(g.before, g.after) for g in groups] == [
        ("Alpha one here.", "Alpha one there."),
        ("Beta two here.", ""),
        ("Gamma.", "Gamma."),
        ("", "Delta new."),
    ]


def test_align_groups_sentences_that_were_merged():
    groups = review.align("We met. We talked.", "We met and talked.")
    assert len(groups) == 1 and groups[0].changed


def test_align_keeps_a_dropped_sentence_among_kept_ones_in_their_group():
    groups = review.align(
        "Ann met Bob. Rain fell hard. Bob left Ann.", "Ann met Bob and Bob left Ann."
    )
    assert [(g.before, g.after) for g in groups] == [
        ("Ann met Bob. Rain fell hard. Bob left Ann.", "Ann met Bob and Bob left Ann.")
    ]


def test_align_with_nothing_in_common():
    groups = review.align("Ann met Bob. Cats purr.", "Zebras run. Owls hoot. Fish swim.")
    assert [(g.before, g.after) for g in groups] == [
        ("Ann met Bob.", ""),
        ("Cats purr.", ""),
        ("", "Zebras run."),
        ("", "Owls hoot."),
        ("", "Fish swim."),
    ]


def test_align_pairs_up_several_reworded_sentences():
    source = "The cat sat on the mat. Dogs bark at night loudly. Birds sing."
    edited = "The cat sat on a mat. Dogs bark loudly at night. Fish swim."
    assert [(g.before, g.after) for g in review.align(source, edited)] == [
        ("The cat sat on the mat.", "The cat sat on a mat."),
        ("Dogs bark at night loudly.", "Dogs bark loudly at night."),
        ("Birds sing.", ""),
        ("", "Fish swim."),
    ]


def test_align_keeps_a_paragraph_with_formatting_tags_whole():
    groups = review.align("A <s1>b</s1>. C.", "A <s1>b</s1>. D.")
    assert len(groups) == 1


@pytest.mark.parametrize(
    "source, edited, expected",
    [
        ("- An item.", "• An item, edited.", "- An item, edited."),
        ("- An item.", "Item, edited.", "- Item, edited."),
        ("Thank you for it.", "thank you so much for it.", "Thank you so much for it."),
        ("Plain.", "Plain words.", "Plain words."),
    ],
)
def test_restore_form(source, edited, expected):
    assert review.restore_form(source, edited) == expected


def test_needs_work():
    assert not review.needs_work("We like it.", VOICE, ACTIVE)
    assert "active voice" in review.needs_work("It was written.", VOICE, ACTIVE)[0]
    long = " ".join(["word"] * 30) + "."
    assert "split this 30-word sentence" in review.needs_work(long, VOICE, ACTIVE)[0]
    assert "contractions" in review.needs_work("We do not agree.", VOICE, ACTIVE)[0]
    assert not review.needs_work("We do not agree.", Voice(display_name="x"), ACTIVE)


def test_spelled_out_words_are_left_when_the_text_already_contracts():
    rules = StyleRules()
    text = "We don't know yet. It's early. We can't say. We do not agree."
    assert review.contracted_enough(text, VOICE)
    assert not review.flagged(text, VOICE, rules)
    assert review.flagged("We do not agree. It is late.", VOICE, rules)
    assert not review.needs_work("We do not agree.", VOICE, rules, contracted=True)


def test_flagged_and_focus():
    found = review.flagged("We like it. It was written.", VOICE, ACTIVE)
    assert [sentence for sentence, _ in found] == ["It was written."]
    lines = review.focus(found)
    assert lines[0].startswith("Change only the sentences listed here")
    assert lines[1].startswith('"It was written.": ')


@pytest.mark.parametrize(
    "before, after, words",
    [
        ("The interview was edited.", "The interview was edited by Korotkoff.", "Korotkoff"),
        ('He wrote "no." Then he left.', 'He wrote "no." Then he wrote "no."', "repeated"),
        ("It is {{1}}.", "It is {{1}}.", None),
        ("We have 3.", "We have {{1}}.", "placeholder"),
        ("We will go.", "We may go.", "'may'"),
        ("They met.", "We met.", "I/we"),
    ],
)
def test_checks(before, after, words):
    problems = review.checks(before, after)
    if words is None:
        assert problems == []
    else:
        assert any(words in problem for problem in problems), problems


def test_checks_can_ignore_person():
    assert review.checks("They met.", "We met.", keep_person=False) == []


@pytest.mark.parametrize(
    "reply, same",
    [
        ("SAME", True),
        ("**Same.**", True),
        ("SAME: but it names who did it", False),
        ("DIFFERENT: who did it", False),
        ("", False),
        ("I think so", False),
    ],
)
def test_verdict(reply, same):
    assert (review.verdict(reply) is None) is same


def test_verdict_names_the_change():
    assert review.verdict("DIFFERENT: tense") == "the checker found a change of meaning: tense"


def test_judge_messages_mark_the_passages_as_data():
    messages = review.judge_messages("A <s1>b</s1>.", "A b c.")
    assert messages[0].content == review.JUDGE_RULES
    assert "<s1>" not in messages[1].content
    assert messages[1].content.count("<<<PROSELIGN_INPUT_") == 2


def test_assemble_keeps_the_source_whitespace():
    groups = review.align("- One.\n\n- Two.\n", "- One, edited.\n- Two.\n\n\n")
    groups[0].accepted = True
    assert review.assemble(groups) == "- One, edited.\n\n- Two.\n"


def _rewriter(changes, judge=None, **options):
    options = Options(review=True, rules=ACTIVE, **options)
    return Rewriter(VOICE, Edits(changes), options, judge)


def test_review_keeps_the_changes_the_checker_accepts():
    judge = Judge(bad="Korotkoff")
    rewriter = _rewriter({"was edited": "was edited by Korotkoff", "big": "large"}, judge)
    out = rewriter.rewrite("The interview was edited. It is a big list.\n")
    assert out == "The interview was edited. It is a large list.\n"
    assert len(judge.asked) == 1  # the invented name failed the rule checks first
    assert (
        rewriter.report.summary()[0] == "kept the original wording of 1 sentence: name changed (1)"
    )


def test_review_reverts_what_the_checker_rejects():
    judge = Judge(bad="large")
    rewriter = _rewriter({"big": "large", "small": "tiny"}, judge)
    out = rewriter.rewrite("It is a big list. It is a small list.\n")
    assert out == "It is a big list. It is a tiny list.\n"
    assert rewriter.report.summary() == [
        "kept the original wording of 1 sentence: meaning changed (1)"
    ]
    assert "kept the original of 'It is a big list.'" in rewriter.report.warnings[0]


def test_review_restores_dropped_and_removes_added_sentences():
    rewriter = _rewriter({"Two here. ": "", "Three.": "Three. Four more."})
    out = rewriter.rewrite("One here. Two here. Three.\n")
    assert out == "One here. Two here. Three.\n"
    assert rewriter.report.summary() == [
        "kept the original wording of 2 sentences: sentence dropped (1), sentence added (1)"
    ]


def test_review_without_a_checker_keeps_changes_that_pass_the_rules():
    rewriter = _rewriter({"big": "large", "—": ","})
    assert rewriter.rewrite("It is a big list.\n") == "It is a large list.\n"


def test_review_is_strict_when_asked():
    rewriter = _rewriter({"{{1}}": "forty"}, strict=True)
    with pytest.raises(ValidationError):
        rewriter.rewrite("It is a big list. We have 42 of them.\n")


def test_a_failed_chunk_keeps_its_good_sentences():
    rewriter = _rewriter({"big": "large", "{{1}}": "forty"}, max_repairs=0)
    out = rewriter.rewrite("It is a big list. We have 42 of them.\n")
    assert out == "It is a large list. We have 42 of them.\n"
    assert rewriter.report.fallbacks == 0
    assert "number changed" in rewriter.report.summary()[0]


def test_targeted_leaves_fitting_text_alone():
    backend = Edits({"like": "love"})
    rewriter = Rewriter(VOICE, backend, Options(review=True, targeted=True, rules=ACTIVE))
    source = "We like it.\n\n```\ncode\n```\n"
    assert rewriter.rewrite(source) == source
    assert not backend.inputs
    assert rewriter.report.untouched == 1


def test_targeted_changes_only_the_flagged_sentences():
    backend = Edits({"like": "love", "It was written by Ann.": "Ann wrote it."})
    rewriter = Rewriter(VOICE, backend, Options(review=True, targeted=True, rules=ACTIVE))
    out = rewriter.rewrite("We like it. It was written by Ann.\n")
    assert out == "We like it. Ann wrote it.\n"
    assert "Change only the sentences listed here" in backend.prompts[0]
    assert 'It was written by Ann.": ' in backend.prompts[0]


def test_targeted_paragraphs_of_a_document():
    backend = Edits({"was signed by Ann": "Ann signed"})
    options = Options(review=True, targeted=True, rules=ACTIVE)
    rewriter = Rewriter(VOICE, backend, options)
    units = ["We like it.", "The deal was signed by Ann.", "We left."]
    assert rewriter.rewrite_units(units) == ["We like it.", "The deal Ann signed.", "We left."]
    assert Rewriter(VOICE, backend, options).rewrite_units(["We like it."]) == ["We like it."]


def test_prompts_carry_the_focus():
    focus = ["Change only these", '"A.": fix it']
    for messages in (
        build_messages(VOICE, "A.", 0.5, focus=focus),
        build_tuned_messages("A.", focus=focus),
    ):
        assert 'Change only these:\n- "A.": fix it' in messages[-1].content


def test_cli_reviews_by_default_and_can_be_told_not_to(trained, monkeypatch):
    from proselign.inference import mock  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(mock.MockBackend, "rewrite", lambda self, m, t: "We met. They left.")
    source = "We met. They left. We stayed.\n"
    code, out, _ = run(["--voice", "Bryan", "--backend", "mock"], source, registry=trained)
    assert code == 0 and out == source
    argv = ["--voice", "Bryan", "--backend", "mock", "--no-review"]
    code, out, _ = run(argv, source, registry=trained)
    assert code == 0 and out == "We met. They left.\n"  # too short for the chunk checks


def test_judge_model_choice():
    from proselign import cli  # pylint: disable=import-outside-toplevel
    from proselign.config import Config  # pylint: disable=import-outside-toplevel
    from proselign.inference.openai_compat import (  # pylint: disable=import-outside-toplevel
        OpenAICompatBackend,
    )

    pick = cli._judge  # pylint: disable=protected-access
    backend = OpenAICompatBackend("http://127.0.0.1:9/v1", "tuned-model")
    voice = Voice(display_name="x", tuned={"model": "tuned-model", "base_model": "base"})
    args = cli.build_parser().parse_args(["rewrite", "--voice", "x"])
    on = Options(review=True)
    assert pick(args, Config(), voice, Options(review=True, tuned=True), backend).model == "base"
    assert pick(args, Config(), voice, on, backend).model == "tuned-model"
    assert pick(args, Config(judge_model="j"), voice, on, backend).model == "j"
    args = cli.build_parser().parse_args(["rewrite", "--judge-model", "k"])
    assert pick(args, Config(), voice, on, backend).model == "k"
    assert pick(args, Config(), voice, Options(), backend) is None
    assert backend.model == "tuned-model"


def test_a_sentence_with_an_avoided_word_is_flagged_for_rewording():
    rules = StyleRules(avoid_words=("appears",))
    found = review.flagged("We like it. It appears to work.", VOICE, rules)
    assert [sentence for sentence, _ in found] == ["It appears to work."]
    assert "appears" in found[0][1][0]
