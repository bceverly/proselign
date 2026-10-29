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
    r"|happy to (?:help|adjust))[^\n]*\Z",
    re.IGNORECASE,
)
_SIGN_OFF_RE = re.compile(
    r"\n+(?:cheers|best|best regards|kind regards|regards|warm regards|thanks|thank you"
    r"|many thanks|sincerely|all the best|take care)[,.!]?[ \t]*(?:\n[^\n]{1,40})?[ \t]*\Z",
    re.IGNORECASE,
)
_LEAKED_MARKER_RE = re.compile(r"[ \t]*<<<(?:END_)?PROSELIGN_INPUT_[0-9a-f]{16}>>>[ \t]*\n?")
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


def directives(voice: Voice, rules: StyleRules) -> list[str]:
    """Concrete writing instructions for ``voice`` under ``rules``."""
    m = voice.metrics or {}
    out: list[str] = []
    sentence = m.get("sentence_length_words", {})
    if sentence.get("mean"):
        out.append(
            f"Average about {round(sentence['mean'])} words per sentence; half of the "
            f"sentences have {round(sentence['median'])} words or fewer. Mix short and "
            "long sentences rather than making them uniform."
        )
    paragraph = m.get("paragraph_length_sentences", {}).get("mean")
    if paragraph:
        out.append(f"Keep paragraphs to about {max(1, round(paragraph))} sentence(s).")
    out.extend(_rate_directives(m, rules))
    out.extend(_punctuation_directives(m.get("punctuation_per_1k", {}), rules))
    if rules.ai_filter:
        out.append(
            "Avoid stock AI phrasing (for example: delve, tapestry, testament to, "
            "it's worth noting, seamlessly, furthermore, moreover, in conclusion)."
        )
    return out


def _rate_directives(m: dict, rules: StyleRules) -> list[str]:
    out = []
    contractions = m.get("contractions_per_1k", 0)
    if contractions >= 8:
        out.append("Use contractions freely (don't, it's, we're, I'd).")
    elif contractions >= 3:
        out.append("Use contractions occasionally, not in every sentence.")
    else:
        out.append("Avoid contractions; write words out in full.")
    first = m.get("first_person_per_1k", 0)
    if first >= 30:
        out.append("Write in the first person (I, we) often.")
    elif first < 5:
        out.append("Keep the author out of it; avoid I and we.")
    second = m.get("second_person_per_1k", 0)
    if second >= 8:
        out.append("Address the reader directly as you.")
    elif second < 2:
        out.append("Rarely address the reader as you.")
    word_length = m.get("mean_word_length", 0)
    if word_length and word_length < 4.7:
        out.append("Prefer short, plain, everyday words.")
    elif word_length > 5.2:
        out.append("Precise technical and business vocabulary is natural here.")
    if not rules.passive_voice:
        out.append("Use the active voice only: never 'was decided', always 'we decided'.")
    elif m.get("passive_per_1k", 0) < 5:
        out.append("Strongly prefer the active voice.")
    return out


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
    text = _LEAKED_MARKER_RE.sub("", text).translate(_INVISIBLE).translate(_SPACES)
    if not _PREAMBLE_RE.match(source):
        text = _PREAMBLE_RE.sub("", text, count=1)
    if not _CHATTY_LEAD_RE.match(source):
        text = _CHATTY_LEAD_RE.sub("", text, count=1)
    if not _OUTRO_RE.search(source):
        text = _OUTRO_RE.sub("", text)
    if not _SIGN_OFF_RE.search(source.rstrip()):
        text = _SIGN_OFF_RE.sub("", text.rstrip())
    text = text.strip()
    for opener, closer in (('"', '"'), ("\u201c", "\u201d"), ("```", "```")):
        wrapped = text.startswith(opener) and text.endswith(closer) and len(text) > 2 * len(opener)
        if wrapped and not source.strip().startswith(opener):
            text = text[len(opener) : -len(closer)].strip()
            if opener == "```":
                text = text.split("\n", 1)[1] if "\n" in text else text
    if not any(c in source for c in "\u2018\u2019\u201c\u201d"):
        text = text.translate(_CURLY)
    if "\u2026" not in source:
        text = text.replace("\u2026", "...")
    return text


def _punctuate(text: str, rules: StyleRules) -> str:
    if not rules.em_dashes:
        text = re.sub(r"(?<=\S)[ \t]*(?:\u2014| -- )[ \t]*(?=\S)", ", ", text)
        text = re.sub(r"[ \t]*\u2014[ \t]*", " ", text)
    if not rules.en_dashes:
        text = re.sub(r"(?<=\d)\u2013(?=\d)", "-", text)
        text = re.sub(r"(?<=\S)[ \t]+\u2013[ \t]+(?=\S)", ", ", text)
        text = text.replace("\u2013", "-")
    if not rules.exclamation_marks:
        text = re.sub(r"!+(?=\s|$)", ".", text)
        text = re.sub(r"\?\.", "?", text)
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
