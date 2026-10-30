# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Style rules: prompt directives, mechanical clean-up and rule checks."""

from __future__ import annotations

import pytest

from proselign.rewrite.engine import Options, Rewriter
from proselign.rewrite.prompt import build_messages
from proselign.rewrite.protect import protected_spans
from proselign.rewrite.style import (
    StyleRules,
    clean,
    directives,
    passive_phrases,
    punctuate,
    violations,
)
from proselign.voice.profile import Voice

from .test_rewrite import ScriptedBackend

STRICT = StyleRules(
    passive_voice=False, en_dashes=False, em_dashes=False, exclamation_marks=False, semicolons=False
)


def _voice(**metrics) -> Voice:
    base = {
        "sentence_length_words": {"mean": 13.8, "median": 10.0},
        "paragraph_length_sentences": {"mean": 1.6},
        "contractions_per_1k": 12,
        "first_person_per_1k": 41,
        "second_person_per_1k": 10,
        "mean_word_length": 4.5,
        "passive_per_1k": 7,
        "punctuation_per_1k": {
            "em_dashes": 0.5,
            "exclamations": 1.2,
            "semicolons": 1.0,
            "questions": 4.6,
            "parentheses": 9,
        },
    }
    base.update(metrics)
    return Voice(display_name="Bryan", style={"tone": ["personal"]}, metrics=base)


def test_directives_translate_metrics_into_instructions():
    text = "\n".join(directives(_voice(), StyleRules()))
    for phrase in (
        "about 14 words",
        "10 words or fewer",
        "about 2 sentence",
        "contractions freely",
        "first person",
        "directly as you",
        "short, plain",
        "occasional exclamation",
        "direct question",
        "Parenthetical",
        "stock AI phrasing",
    ):
        assert phrase in text
    assert "em dash" not in text
    assert "active voice" not in text


def test_directives_for_a_formal_voice_and_strict_rules():
    formal = _voice(
        contractions_per_1k=1,
        first_person_per_1k=2,
        second_person_per_1k=1,
        mean_word_length=5.4,
        passive_per_1k=2,
        punctuation_per_1k={"em_dashes": 0.1, "exclamations": 0.0, "semicolons": 0.1},
    )
    text = "\n".join(directives(formal, StyleRules(ai_filter=False)))
    for phrase in ("Avoid contractions", "avoid I and we", "Rarely address", "technical"):
        assert phrase in text
    for phrase in ("Strongly prefer the active", "em dashes", "exclamation", "semicolons"):
        assert phrase in text
    assert "stock AI" not in text
    strict = "\n".join(directives(_voice(contractions_per_1k=5), STRICT))
    for phrase in ("sparingly", "active voice only", "en dashes", "em dashes", "semicolons"):
        assert phrase in strict
    assert directives(Voice(display_name="Empty"), StyleRules())


def test_prompt_carries_directives():
    user = build_messages(_voice(), "Text.", 0.5, rules=STRICT)[1].content
    assert "Overall tone: personal" in user
    assert "- Use the active voice only" in user


@pytest.mark.parametrize(
    "candidate,expected",
    [
        (
            "The numbers\u2014revenue and churn\u2014look good.",
            "The numbers, revenue and churn, look good.",
        ),
        ("Done -- finally.", "Done, finally."),
        ("\u2014 Bryan", "Bryan"),
        ("Pages 10\u201312 cover it.", "Pages 10-12 cover it."),
        ("Fast \u2013 and cheap.", "Fast, and cheap."),
        ("A\u2013", "A-"),
        ("Great! Really?! Yes!!", "Great. Really? Yes."),
        ("We shipped; it worked; Done.", "We shipped. It worked. Done."),
        ("Stay calm, \u2014 then go.", "Stay calm, then go."),
    ],
)
def test_punctuate_enforces_punctuation_rules(candidate, expected):
    rules = StyleRules(ai_filter=False, **_off())
    assert violations("source", candidate, rules)
    assert punctuate(candidate, rules) == expected
    assert not violations("source", expected, rules)


def _off() -> dict:
    return {"en_dashes": False, "em_dashes": False, "exclamation_marks": False, "semicolons": False}


def test_clean_leaves_protected_spans_alone():
    candidate = (
        'He said "wait\u2014now!" and ran `a;b` to http://x.example/a\u2013b today\u2014fast!'
    )
    assert punctuate(clean("x", candidate, STRICT), STRICT) == (
        'He said "wait\u2014now!" and ran `a;b` to http://x.example/a\u2013b today, fast.'
    )
    assert protected_spans('`a` and "c"') == [(0, 3), (8, 11)]
    assert protected_spans('"a `b" c`') == [(0, 9)]  # overlapping spans merge


@pytest.mark.parametrize(
    "source,candidate,expected",
    [
        ("Hi.", "Sure! Here's the rewritten text:\n\nHello.", "Hello."),
        ("Hi.", "Here is the revised version:\nHello.", "Hello."),
        ("Hi.", "Certainly.\n\nHello.", "Hello."),
        ("Hi.", "Hello.\n\nLet me know if you'd like changes!", "Hello."),
        ("Hi.", "Hello.\n\nCheers,\nBryanDocs", "Hello."),
        ("Hi.", "Hello.\nBest regards", "Hello."),
        ("Hi.\n\nThanks,\nBryan", "Hello.\n\nThanks,\nBryan", "Hello.\n\nThanks,\nBryan"),
        ("Hi.", '"Hello."', "Hello."),
        ("Hi.", "\u201cHello.\u201d", "Hello."),
        ("Hi.", "```\nHello.\n```", "Hello."),
        ("Hi.", "```Hello.```", "Hello."),
        ("Hi.", "It\u2019s \u201cfine\u201d\u2026", 'It\'s "fine"...'),
        ("It\u2019s \u2026", "It\u2019s \u2026", "It\u2019s \u2026"),
        ("Hi.", "A\u00a0b\u200bc", "A bc"),
        ("Hi.", "Hello.\n<<<END_PROSELIGN_INPUT_0123456789abcdef>>>", "Hello."),
        ('"Quoted."', '"Quoted."', '"Quoted."'),
        (
            "Here's the revised plan:\nGo.",
            "Here's the revised plan:\nGo.",
            "Here's the revised plan:\nGo.",
        ),
        ("Sure.\nGo.", "Sure.\nGo.", "Sure.\nGo."),
        (
            "Go.\nLet me know if it works.",
            "Go.\nLet me know if it works.",
            "Go.\nLet me know if it works.",
        ),
    ],
)
def test_clean_strips_ai_artifacts(source, candidate, expected):
    assert clean(source, candidate, StyleRules()) == expected


def test_ai_filter_off_keeps_artifacts():
    assert clean("Hi.", "Sure.\n\n\u201cHello\u2014there.\u201d", StyleRules(ai_filter=False)) == (
        "Sure.\n\n\u201cHello\u2014there.\u201d"
    )


def test_passive_detection_skips_false_friends_and_quotes():
    text = 'It was decided. The door is open. It is often fine. "Mistakes were made." Was taken.'
    assert passive_phrases(text) == ["was decided", "Was taken"]


def test_violations():
    assert violations("x", "It was decided.", STRICT) == [
        "rewrite these passive constructions in the active voice: 'was decided'"
    ]
    assert not violations("x", "It was decided.", StyleRules())
    assert violations("x", "Let's delve. Furthermore, a tapestry.", StyleRules()) == [
        "replace stock AI phrasing: delve, furthermore, tapestry"
    ]
    assert not violations("We delve.", "We delve.", StyleRules())
    assert not violations("x", "We delve.", StyleRules(ai_filter=False))


def test_engine_asks_again_then_accepts_best_with_warning():
    backend = ScriptedBackend(["It was decided.", "We decided."])
    rewriter = Rewriter(_voice(), backend, Options(rules=STRICT))
    assert rewriter.rewrite("Decision made.\n") == "We decided.\n"
    assert rewriter.report.repaired == 1
    assert "passive" in backend.prompts[1][1].content

    stubborn = Rewriter(
        _voice(), ScriptedBackend(["It was decided."] * 2), Options(rules=STRICT, max_repairs=1)
    )
    assert stubborn.rewrite("Decision made.\n") == "It was decided.\n"
    assert stubborn.report.fallbacks == 0
    assert "style rules not fully met" in stubborn.report.warnings[0]


def test_engine_cleans_before_checking_fidelity():
    backend = ScriptedBackend(
        ["Here's the rewritten text:\n\u201cWe have 42\u2014all of them.\u201d"]
    )
    rewriter = Rewriter(_voice(), backend, Options(rules=STRICT, max_repairs=0))
    assert rewriter.rewrite("We have 42.") == "We have 42, all of them."
    assert not rewriter.report.warnings


def test_engine_asks_for_rephrasing_before_replacing_dashes():
    backend = ScriptedBackend(["Fast\u2014cheap.", "Fast and cheap."])
    rewriter = Rewriter(_voice(), backend, Options(rules=STRICT))
    assert rewriter.rewrite("Quick and cheap.") == "Fast and cheap."
    assert "no em dashes" in backend.prompts[1][1].content


def test_engine_mechanical_fix_keeps_quotations():
    rules = StyleRules(semicolons=False)
    backend = ScriptedBackend(['He said "a;b" ok; fine.'])
    rewriter = Rewriter(_voice(), backend, Options(rules=rules, max_repairs=0))
    assert rewriter.rewrite('He said "a;b".') == 'He said "a;b" ok. Fine.'


def test_first_person_plural_directive_and_check():
    plural = StyleRules(first_person="plural")
    text = "\n".join(directives(_voice(), plural))
    assert "first person plural only" in text
    assert "first person (I, we) often" not in text
    assert "first person singular" in "\n".join(
        directives(_voice(), StyleRules(first_person="singular"))
    )
    assert violations("x", "I think my plan, and I'm sure of it, suits me.", plural) == [
        "use the first person plural (we, us, our) instead of: I, my, me"
    ]
    clean_text = 'We think so. He said "I agree." Phase I ends; World War I too. Mine, sir?'
    assert violations("x", clean_text, plural) == [
        "use the first person plural (we, us, our) instead of: Mine"
    ]
    assert not violations("x", "We agree with our team.", plural)
    assert not violations("x", "I agree.", StyleRules())


def test_engine_asks_for_plural_until_it_gets_it():
    backend = ScriptedBackend(["I agree with my team.", "We agree with our team."])
    rewriter = Rewriter(_voice(), backend, Options(rules=StyleRules(first_person="plural")))
    assert rewriter.rewrite("Agreed.") == "We agree with our team."
    assert "instead of: I, my" in backend.prompts[1][1].content
