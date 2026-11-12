# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The word-fix pass: sentences that still use avoided words are reworded."""

from __future__ import annotations

from proselign.inference.base import Backend
from proselign.rewrite.engine import Options, Rewriter
from proselign.rewrite.markers import extract_input
from proselign.rewrite.style import StyleRules
from proselign.rewrite.wordfix import build_word_fix, fix_words
from proselign.voice.profile import Voice

RULES = StyleRules(avoid_words=("appears", "rather than"))


class Fixer(Backend):
    """Rewords by ``changes`` (old -> new), and as the checker says SAME
    unless the edited side contains ``bad``."""

    def __init__(self, changes: dict[str, str], bad: str = "\0") -> None:
        self.changes, self.bad = changes, bad
        self.asked: list[str] = []

    def rewrite(self, messages, temperature):
        content = messages[-1].content
        if "EDITED:" in content:
            return "DIFFERENT: tense" if self.bad in content.split("EDITED:")[1] else "SAME"
        self.asked.append(content)
        text = extract_input(content)
        for old, new in self.changes.items():
            text = text.replace(old, new)
        return text


def test_a_sentence_is_reworded_and_the_rest_left_alone():
    text = (
        'The log appears complete. F5 wrote "it appears so" on May 2.\n\n'
        "We read 1,149 notices rather than 12 copies."
    )
    fixer = Fixer({"appears": "looks", "rather than": "instead of"})
    outcome = fix_words(text, RULES, fixer, fixer)
    assert outcome.text == (
        'The log looks complete. F5 wrote "it appears so" on May 2.\n\n'
        "We read 1,149 notices instead of 12 copies."
    )
    assert outcome.fixed == 2 and not outcome.kept
    # Only sentences with an avoided word outside quotations are sent, masked.
    assert len(fixer.asked) == 2
    assert fixer.asked[0].startswith("Replace: appears\n")
    assert "{{" in fixer.asked[1] and "1,149" not in fixer.asked[1]


def test_a_failed_fix_keeps_the_sentence():
    cases = [
        (Fixer({}), "still uses appears"),  # word left in
        (Fixer({"appears": "looks", "complete": "done"}, bad="done"), "change of meaning"),
        (Fixer({"appears complete": "was completed"}), "passive"),
        (Fixer({"The log appears complete.": ""}), "empty"),
    ]
    for fixer, why in cases:
        outcome = fix_words("The log appears complete.", RULES, fixer, fixer)
        assert outcome.text == "The log appears complete." and outcome.fixed == 0
        assert outcome.kept and why in " ".join(outcome.kept[0][1])
    # A name or a number the fix changes is caught too.
    fixer = Fixer({"appears": "looks", "F5": "NGINX"})
    outcome = fix_words("The log from F5 appears complete.", RULES, fixer, fixer)
    assert outcome.fixed == 0 and "name" in " ".join(outcome.kept[0][1])
    fixer = Fixer({"{{1}}": ""})
    outcome = fix_words("It appears 12 are gone.", RULES, fixer, fixer)
    assert outcome.fixed == 0 and outcome.kept


def test_nothing_to_do_asks_nothing():
    fixer = Fixer({"appears": "looks"})
    assert fix_words("It looks fine.", RULES, fixer).text == "It looks fine."
    assert fix_words("It appears fine.", StyleRules(), fixer).text == "It appears fine."
    assert not fixer.asked
    # Without a checker, the fidelity checks alone decide.
    assert fix_words("It appears fine.", RULES, fixer).text == "It looks fine."


def test_the_prompt_names_the_words_and_marks_the_data():
    system, user = build_word_fix("It appears so.", ["appears"])
    assert "nothing else" in system.content
    assert user.content.startswith("Replace: appears\n")
    assert extract_input(user.content) == "It appears so."


class Echo(Backend):
    """A voice model that hands every text back unchanged."""

    def rewrite(self, messages, temperature):
        return extract_input(messages[-1].content)


def test_the_rewriter_runs_the_pass_with_the_checker_model():
    voice = Voice(display_name="B", metrics={"sentence_length_words": {"mean": 12, "stdev": 4}})
    judge = Fixer({"appears": "looks"})
    rewriter = Rewriter(voice, Echo(), Options(rules=RULES, review=True), judge)
    text = "# Notes\n\nThe log appears complete. It was read twice by the team.\n"
    assert (
        rewriter.rewrite(text)
        == "# Notes\n\nThe log looks complete. It was read twice by the team.\n"
    )
    assert rewriter.report.word_fixes == 1
    # A fix the checker rejects is reported with the summary.
    judge.bad = "looks"
    rewriter = Rewriter(voice, Echo(), Options(rules=RULES, review=True, parallel=1), judge)
    assert "appears" in rewriter.rewrite(text)
    assert any("kept avoided words in 1 sentence" in line for line in rewriter.report.summary())
    assert any("kept the avoided words in" in w for w in rewriter.report.warnings)
    # .docx paragraphs get the pass too.
    rewriter = Rewriter(voice, Echo(), Options(rules=RULES), Fixer({"appears": "looks"}))
    assert rewriter.rewrite_units(["The log appears complete."]) == ["The log looks complete."]


def test_a_fix_may_change_only_the_avoided_words_and_a_little_more():
    rules = StyleRules(avoid_words=("absence", "appear"))
    text = "So their absence from the file is out of the ordinary."
    polished = Fixer({"their absence": "their lack", "out of the ordinary": "unusual"})
    outcome = fix_words(text, rules, polished, polished)
    assert outcome.fixed == 0 and "changes other words too" in outcome.kept[0][1][0]
    small = Fixer({"appear nowhere": "don't show up"})
    assert (
        fix_words("The changes appear nowhere.", rules, small).text == "The changes don't show up."
    )
    padded = Fixer({"appear": "can be seen, by anyone who looks for them closely, to be"})
    outcome = fix_words("The changes appear missing.", rules, padded)
    assert outcome.fixed == 0 and any("adds" in p for p in outcome.kept[0][1])


def test_a_fix_may_not_add_a_comma():
    rules = StyleRules(avoid_words=("rather than",))
    serial = Fixer({"rather than": "instead of", "tests and": "tests, and"})
    outcome = fix_words("It keeps logs, tests and notes rather than files.", rules, serial)
    assert outcome.fixed == 0 and "adds a comma" in outcome.kept[0][1]


def test_the_authors_conventions_reach_the_fixer():
    system, _ = build_word_fix(
        "It appears so.", ["appears"], ["No serial comma: 'red, white and blue'."]
    )
    assert system.content.endswith(
        "The author's conventions: No serial comma: 'red, white and blue'."
    )
    assert "conventions" not in build_word_fix("It appears so.", ["appears"])[0].content
