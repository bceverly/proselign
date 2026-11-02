# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Turn a voice and the user's style rules into instructions, checks and fixes.

A model follows "average 14 words per sentence, use contractions freely"
far better than a table of per-1,000-word rates, so the profile is
translated into plain directives.  After each attempt, punctuation rules
and common AI artifacts are fixed deterministically; what cannot be fixed
mechanically (passive voice, stock AI phrasing) is reported back to the
model for another try.  Protected spans (quotations, code, URLs, e-mail
addresses) are never altered.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from proselign.rewrite.protect import protected_spans
from proselign.voice.profile import Voice


@dataclass(frozen=True)
class StyleRules:
    """User-level switches, usually read from ``~/.proselignrc``."""

    passive_voice: bool = True
    en_dashes: bool = True
    em_dashes: bool = True
    exclamation_marks: bool = True
    semicolons: bool = True
    ai_filter: bool = True
    # "voice" follows the profile; "plural" forces we/us/our; "singular" prefers I/me/my.
    first_person: str = "voice"


# Past participles that the passive heuristic would otherwise misread.
_NOT_PARTICIPLES = frozenset(
    {"often", "even", "open", "seven", "eleven", "ten", "then", "when", "garden", "token"}
)
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(\w+(?:ed|en))\b", re.IGNORECASE
)
AI_PHRASES = (
    "delve",
    "delves",
    "delving",
    "tapestry",
    "testament to",
    "in today's fast-paced",
    "ever-evolving",
    "it's worth noting",
    "it is worth noting",
    "it's important to note",
    "it is important to note",
    "navigate the complexities",
    "in the realm of",
    "a myriad of",
    "plethora",
    "game-changer",
    "game changer",
    "seamlessly",
    "embark on",
    "unlock the potential",
    "rest assured",
    "i hope this email finds you well",
    "i hope this message finds you well",
    "don't hesitate to reach out",
    "feel free to reach out",
    "let's dive in",
    "dive into",
    "in conclusion",
    "furthermore",
    "moreover",
    "pivotal",
    "paramount",
    "underscores",
    "vibrant",
    "bustling",
    "intricate",
)
_SINGULAR_RE = re.compile(r"\b(?:I|[Mm]e|[Mm]y|[Mm]ine|[Mm]yself)\b")
# "Phase I", "Section I": a Roman numeral, not the pronoun.
_NUMERAL_BEFORE = re.compile(
    r"(?:phase|section|part|chapter|title|article|class|type|tier|stage|level|appendix"
    r"|exhibit|schedule|volume|book|world war)\s+\Z",
    re.IGNORECASE,
)
_AI_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in AI_PHRASES) + r")\b", re.IGNORECASE
)
_PREAMBLE_RE = re.compile(
    r"\A(?:(?:sure|certainly|of course|absolutely)[!.,]?\s+)?"
    r"(?:here(?:'s| is| are) (?:the|a|an|your) [^\n]{0,60}"
    r"(?:rewrit|revis|updat|edit|reword|version)[^\n]{0,60}:[ \t]*\n+)",
    re.IGNORECASE,
)
_CHATTY_LEAD_RE = re.compile(r"\A(?:sure|certainly|of course|absolutely)[!.,][ \t]*\n+", re.I)
_OUTRO_RE = re.compile(
    r"\n+[^\n]{0,40}\b(?:let me know if|i hope this helps|hope this helps|feel free to ask"
    r"|happy to (?:help|adjust)|hope (?:everything|all|things|you)\b[^\n]*\b(?:well|good|great))"
    r"[^\n]*\Z",
    re.IGNORECASE,
)
_SIGN_OFF_RE = re.compile(
    r"\n+(?:cheers|best|best regards|kind regards|regards|warm regards|thanks|thank you"
    r"|many thanks|sincerely|all the best|take care)[,.!]?[ \t]*(?:\n[^\n]{1,40})?[ \t]*\Z",
    re.IGNORECASE,
)
_LEAKED_MARKER_RE = re.compile(r"<<<(?:END_)?PROSELIGN_INPUT_[0-9a-f]{16}>>>[ \t]*\n?")
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_SPACES = {ord("\u00a0"): " ", ord("\u202f"): " "}
_CURLY = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"'}


def _outside(text: str, fix: Callable[[str], str]) -> str:
    """Apply ``fix`` to everything except protected spans."""
    out, pos = [], 0
    for start, end in protected_spans(text):
        out.append(fix(text[pos:start]))
        out.append(text[start:end])
        pos = end
    out.append(fix(text[pos:]))
    return "".join(out)


def _strip_protected(text: str) -> str:
    out, pos = [], 0
    for start, end in protected_spans(text):
        out.append(text[pos:start])
        out.append(" ")
        pos = end
    out.append(text[pos:])
    return "".join(out)


# --- instructions ---------------------------------------------------------


# Naming the actor the original names; "we" here would make the author the
# subject of whatever someone else did.
ACTIVE_VOICE = (
    "Use the active voice only, naming whoever the original says acted: 'the board "
    "decided', not 'it was decided by the board'. Never make the author ('we') the "
    "subject of something someone else did; if the original does not say who acted, "
    "leave that sentence as it is."
)


# The groups the voice's instructions fall into; ``proselign eval --ablate``
# leaves out one at a time to see what each contributes.
DIRECTIVE_GROUPS = (
    "sentences",
    "contractions",
    "person",
    "passive",
    "word length",
    "punctuation",
    "vocabulary",
)


def directives(voice: Voice, rules: StyleRules, omit: frozenset[str] = frozenset()) -> list[str]:
    """Concrete writing instructions for ``voice`` under ``rules``, leaving
    out the groups (from ``DIRECTIVE_GROUPS``) named in ``omit``."""
    if not voice.metrics:
        return _keep_style_directives(rules)
    out = [text for group, text in grouped_directives(voice, rules) if group not in omit]
    if rules.ai_filter:
        out.append(
            "Avoid stock AI phrasing (for example: delve, tapestry, testament to, "
            "it's worth noting, seamlessly, furthermore, moreover, in conclusion)."
        )
    return out


def grouped_directives(voice: Voice, rules: StyleRules) -> list[tuple[str, str]]:
    """The voice's instructions, each with the group it belongs to."""
    m = voice.metrics or {}
    out: list[tuple[str, str]] = []
    sentence = m.get("sentence_length_words", {})
    if sentence.get("mean"):
        out.append(
            (
                "sentences",
                f"Average about {round(sentence['mean'])} words per sentence; half of the "
                f"sentences have {round(sentence.get('median', sentence['mean']))} words or "
                "fewer. Mix short and long sentences rather than making them uniform, and "
                "do not shorten the author's sentences on principle.",
            )
        )
    paragraph = m.get("paragraph_length_sentences", {}).get("mean")
    if paragraph:
        out.append(
            ("sentences", f"Keep paragraphs to about {max(1, round(paragraph))} sentence(s).")
        )
    out.extend(_rate_directives(m, rules))
    out.extend(
        ("punctuation", text)
        for text in _punctuation_directives(m.get("punctuation_per_1k", {}), rules)
    )
    out.extend(("vocabulary", text) for text in lexicon_directives(voice.lexicon or {}))
    return out


def _keep_style_directives(rules: StyleRules) -> list[str]:
    """For a voice with no measurements (``proselign eval``'s control): keep
    the text's own style, apart from what the user's settings forbid."""
    out = [
        "Keep the original text's own style: its sentence lengths, vocabulary, "
        "contractions, use of I and we, and tone. Rephrase it without moving it "
        "toward any particular writer."
    ]
    forbidden = [
        ("em_dashes", "Do not use em dashes (\u2014); use commas, colons or separate sentences."),
        (
            "en_dashes",
            "Do not use en dashes (\u2013); write number ranges with a hyphen (2019-2021).",
        ),
        ("exclamation_marks", "Do not use exclamation marks."),
        ("semicolons", "Do not use semicolons."),
    ]
    out.extend(text for name, text in forbidden if not getattr(rules, name))
    if not rules.passive_voice:
        out.append(ACTIVE_VOICE)
    out.extend(_person_directives(30.0, rules) if rules.first_person != "voice" else [])
    if rules.ai_filter:
        out.append(
            "Avoid stock AI phrasing (for example: delve, tapestry, testament to, "
            "it's worth noting, seamlessly, furthermore, moreover, in conclusion)."
        )
    return out


def lexicon_directives(lexicon: dict) -> list[str]:
    """Word-choice instructions learned from the author's vocabulary."""
    out = []
    prefers = lexicon.get("prefers", [])[:10]
    if prefers:
        pairs = ", ".join(f"'{p['prefer']}' (not '{p['over']}')" for p in prefers)
        out.append(f"Choose the author's plain words: {pairs}.")
    avoids = lexicon.get("avoids", [])[:20]
    if avoids:
        out.append(f"The author never writes these words; do not use them: {', '.join(avoids)}.")
    openers = lexicon.get("openers", [])[:6]
    if openers:
        quoted = ", ".join(f"'{o},'" for o in openers)
        out.append(f"Sentence openers the author often uses, where they fit: {quoted}.")
    conventions = lexicon.get("conventions", {})
    if conventions.get("serial_comma") is True:
        out.append("Use the serial (Oxford) comma: 'red, white, and blue'.")
    elif conventions.get("serial_comma") is False:
        out.append("No serial comma: 'red, white and blue'.")
    if conventions.get("spelling"):
        out.append(f"Use {conventions['spelling']} spelling.")
    return out


def _rate_directives(m: dict, rules: StyleRules) -> list[tuple[str, str]]:
    out = [("contractions", _contraction_directive(m.get("contractions_per_1k", 0)))]
    out.extend(
        ("person", text) for text in _person_directives(m.get("first_person_per_1k", 0), rules)
    )
    second = m.get("second_person_per_1k", 0)
    if second >= 8:
        out.append(("person", "Address the reader directly as you."))
    elif second < 2:
        out.append(("person", "Rarely address the reader as you."))
    word_length = m.get("mean_word_length", 0)
    if word_length and word_length < 4.7:
        out.append(("word length", "Prefer short, plain, everyday words."))
    elif word_length > 5.2:
        out.append(("word length", "Precise technical and business vocabulary is natural here."))
    if not rules.passive_voice:
        out.append(("passive", ACTIVE_VOICE))
    elif m.get("passive_per_1k", 0) < 5:
        out.append(("passive", "Strongly prefer the active voice."))
    return out


def _contraction_directive(contractions: float) -> str:
    # A concrete rate: "occasionally" alone makes models contract far too often.
    every = round(1000 / contractions, -1) if contractions else 0
    if contractions >= 8:
        return f"Use contractions freely (don't, it's, we're), about one every {every:.0f} words."
    if contractions >= 3:
        return (
            f"Use contractions sparingly: about one every {every:.0f} words; write most "
            "words out in full."
        )
    return "Avoid contractions; write words out in full."


def _person_directives(first: float, rules: StyleRules) -> list[str]:
    if rules.first_person == "plural":
        return [
            "When the author speaks of themselves, use the first person plural only: we, "
            "us, our, ours, ourselves. Never use I, me, my, mine or myself; 'I think' "
            "becomes 'we think'. Do not turn anyone else into 'we': what other people "
            "did, said or own stays theirs, and 'you' and 'your' stay as they are."
        ]
    if rules.first_person == "singular":
        return ["Speak for the author in the first person singular: I, me, my."]
    if first >= 30:
        return ["Write in the first person (I, we) often."]
    if first < 5:
        return ["Keep the author out of it; avoid I and we."]
    return []


def _punctuation_directives(punct: dict, rules: StyleRules) -> list[str]:
    out = []
    if not rules.em_dashes or punct.get("em_dashes", 0) < 0.5:
        out.append("Do not use em dashes (\u2014); use commas, colons or separate sentences.")
    if not rules.en_dashes:
        out.append("Do not use en dashes (\u2013); write number ranges with a hyphen (2019-2021).")
    if not rules.exclamation_marks or punct.get("exclamations", 0) < 0.3:
        out.append("Do not use exclamation marks.")
    elif punct.get("exclamations", 0) >= 1:
        out.append("An occasional exclamation mark is fine.")
    if not rules.semicolons or punct.get("semicolons", 0) < 0.3:
        out.append("Do not use semicolons.")
    if punct.get("questions", 0) >= 2:
        out.append("Where the original asks something, a direct question suits this voice.")
    if punct.get("parentheses", 0) >= 8:
        out.append("Parenthetical asides (like this) are characteristic.")
    return out


# --- deterministic clean-up -------------------------------------------------


def clean(source: str, candidate: str, rules: StyleRules) -> str:
    """Strip AI artifacts from a model's ``candidate`` rewrite of ``source``."""
    text = _strip_artifacts(source, candidate) if rules.ai_filter else candidate
    return text.strip()


def punctuate(text: str, rules: StyleRules) -> str:
    """Mechanically remove punctuation the rules forbid (the last resort:
    the model is asked to rephrase first, which reads better)."""
    return _outside(text, lambda part: _punctuate(part, rules)).strip()


def _strip_artifacts(source: str, text: str) -> str:
    text = _drop_markers(text).translate(_INVISIBLE).translate(_SPACES)
    if not _PREAMBLE_RE.match(source):
        text = _PREAMBLE_RE.sub("", text, count=1)
    if not _CHATTY_LEAD_RE.match(source):
        text = _CHATTY_LEAD_RE.sub("", text, count=1)
    if not _OUTRO_RE.search(source):
        text = _OUTRO_RE.sub("", text)
    if not _SIGN_OFF_RE.search(source.rstrip()):
        text = _SIGN_OFF_RE.sub("", text.rstrip())
    text = _unwrap(source, text.strip())
    if not any(c in source for c in "\u2018\u2019\u201c\u201d"):
        text = text.translate(_CURLY)
    if "\u2026" not in source:
        text = text.replace("\u2026", "...")
    if "**" not in source:  # bold the original did not have
        text = text.replace("**", "")
    return text


def _drop_markers(text: str) -> str:
    """``text`` without leaked input markers or the spaces before them."""
    pieces = _LEAKED_MARKER_RE.split(text)
    return "".join(piece.rstrip(" \t") for piece in pieces[:-1]) + pieces[-1]


def _unwrap(source: str, text: str) -> str:
    """``text`` without quotes or a code fence wrapped around all of it."""
    for opener, closer in (('"', '"'), ("\u201c", "\u201d"), ("```", "```")):
        wrapped = text.startswith(opener) and text.endswith(closer) and len(text) > 2 * len(opener)
        if wrapped and not source.strip().startswith(opener):
            text = text[len(opener) : -len(closer)].strip()
            if opener == "```":
                text = text.split("\n", 1)[1] if "\n" in text else text
    return text


def _replace_marks(text: str, mark: str, pick) -> str:
    """Replace every ``mark`` in ``text``: ``pick(before, after)`` gets the
    text on either side and returns the new before, replacement and after.
    (Plain string work: a regex with optional spaces on both sides of the
    mark backtracks badly on long runs of spaces.)  A run of marks counts as one."""
    pieces = text.split(mark)
    if len(pieces) > 2:
        pieces = [pieces[0], *(p for p in pieces[1:-1] if p.strip(" \t")), pieces[-1]]
    out = [pieces[0]]
    for piece in pieces[1:]:
        before, middle, after = pick(out[-1], piece)
        out[-1:] = [before, middle, after]
    return "".join(out)


def _em_dash(before: str, after: str) -> tuple[str, str, str]:
    # "word \u2014 word" and "word\u2014word" become "word, word"; a dash at
    # the edge of a line just goes.
    left, right = before.rstrip(" \t"), after.lstrip(" \t")
    between = left[-1:].strip() and right[:1].strip()
    return left, ", " if between else " ", right


def _en_dash(before: str, after: str) -> tuple[str, str, str]:
    # "word \u2013 word" becomes "word, word"; ranges (2019\u20132021) and
    # anything else get a hyphen.
    left, right = before.rstrip(" \t"), after.lstrip(" \t")
    if left != before and right != after and left[-1:].strip() and right[:1].strip():
        return left, ", ", right
    return before, "-", after


def _punctuate(text: str, rules: StyleRules) -> str:
    if not rules.em_dashes:
        text = _replace_marks(text.replace(" -- ", " \u2014 "), "\u2014", _em_dash)
    if not rules.en_dashes:
        text = _replace_marks(text, "\u2013", _en_dash)
    if not rules.exclamation_marks:
        text = re.sub(r"!+(?=\s|$)", ".", text)
        text = text.replace("?.", "?")
    if not rules.semicolons:
        text = re.sub(r";[ \t]+([a-z])", lambda m: ". " + m.group(1).upper(), text)
        text = re.sub(r";[ \t]+", ". ", text)
    return re.sub(r",[ \t]*,", ",", text)


# --- checks the model must fix itself ---------------------------------------


def passive_phrases(text: str) -> list[str]:
    """Probable passive constructions in ``text`` (a heuristic)."""
    return [
        m.group(0)
        for m in _PASSIVE_RE.finditer(_strip_protected(text))
        if m.group(1).lower() not in _NOT_PARTICIPLES
    ]


def singular_pronouns(text: str) -> list[str]:
    """First person singular pronouns outside protected spans, in order, once each."""
    plain = _strip_protected(text)
    found: list[str] = []
    for match in _SINGULAR_RE.finditer(plain):
        word = match.group(0)
        if word == "I" and _NUMERAL_BEFORE.search(plain[: match.start()]):
            continue
        if word not in found:
            found.append(word)
    return found


def violations(source: str, candidate: str, rules: StyleRules) -> list[str]:
    """Rule breaches in ``candidate`` that only another attempt can fix."""
    problems = []
    plain = _strip_protected(candidate)
    banned = [
        (rules.em_dashes, "\u2014", "em dashes"),
        (rules.en_dashes, "\u2013", "en dashes"),
        (rules.exclamation_marks, "!", "exclamation marks"),
        (rules.semicolons, ";", "semicolons"),
    ]
    for allowed, char, name in banned:
        if not allowed and (char in plain or (char == "\u2014" and " -- " in plain)):
            problems.append(f"rephrase to use no {name}")
    if rules.first_person == "plural":
        found = singular_pronouns(candidate)
        if found:
            problems.append(
                "use the first person plural (we, us, our) instead of: " + ", ".join(found)
            )
    if not rules.passive_voice:
        found = passive_phrases(candidate)
        if found:
            quoted = ", ".join(repr(p) for p in found[:5])
            problems.append(f"rewrite these passive constructions in the active voice: {quoted}")
    if rules.ai_filter:
        allowed = {p.lower() for p in _AI_PHRASE_RE.findall(source)}
        found = sorted({p.lower() for p in _AI_PHRASE_RE.findall(plain)} - allowed)
        if found:
            problems.append(f"replace stock AI phrasing: {', '.join(found)}")
    return problems
