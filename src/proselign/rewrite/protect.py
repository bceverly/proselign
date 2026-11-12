# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Protected spans: the tokens a rewrite must carry through untouched.

Preflight extracts them from the source; postflight checks the rewrite still
contains every one, as many times as the source did. A rewrite that drops a
figure, URL, citation or quotation is a changed meaning, not a style change.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from proselign.ingest.lexicon import FUNCTION_WORDS

_FUNCTION_WORDS = frozenset(FUNCTION_WORDS)

_PATTERNS = {
    "inline_code": re.compile(r"`[^`\n]+`"),
    "url": re.compile(r"\b(?:https?|ftp)://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"]"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "quotation": re.compile(r"\"[^\"\n]{1,400}\"|“[^”\n]{1,400}”"),
    # A bold lead-in at the start of a line or list item ("4. **Records of
    # what was deleted.** These are..."): models drop its markup.
    "bold_label": re.compile(
        r"(?m)(?:(?<=^\d\. )|(?<=^\d\d\. )|(?<=^[-*+] )|^)\*\*[^*\n]{1,160}\*\*(?= )"
    ),
    "citation": re.compile(r"(?<!\[)\[\d+(?:[,–-]\s*\d+)*\](?!\])|\[@[\w:-]+\]"),
    # Thousands separators only between digits: "42, all" is the number 42.
    # Paragraph markers and formatting tags used when rewriting .docx files.
    "markup": re.compile(r"\[\[\d+\]\]|</?s\d+>"),
    # Document IDs such as Bates numbers (INDIV_DEF_00004939, F5_00213467),
    # whose digits the number pattern does not see after a letter or "_".
    "identifier": re.compile(r"\b[A-Za-z][\w-]*?[_-]?\d{3,}\b"),
    "number": re.compile(r"(?<![\w.])[-+]?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?!\w|\.\w)"),
    # A short lead-in that labels a list item or line ("1. Period. Your list
    # ..."): models fold it into the sentence or drop it.
    "label": re.compile(
        r"(?m)(?:(?<=^\d\. )|(?<=^\d\d\. )|(?<=^[-*+] )|^)"
        r"[A-Z][\w'\u2019-]*(?: [\w'\u2019-]+){0,2}\.(?= +[A-Z])"
    ),
}


def extract(text: str) -> Counter:
    """Count every protected token in ``text``, keyed by (kind, token)."""
    found: Counter = Counter()
    for kind, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            found[(kind, match.group(0))] += 1
    return found


def missing(source: str, rewrite: str) -> list[str]:
    """Human-readable list of protected tokens the rewrite lost."""
    want = extract(source)
    have = extract(rewrite)
    problems = []
    for (kind, token), count in sorted(want.items()):
        got = have.get((kind, token), 0)
        if got < count:
            problems.append(f"{kind} {token!r} appears {count}x in the source but {got}x after")
    for (kind, token), got in sorted(have.items()):
        # Markers, formatting tags and figures must not be invented either
        # ("five more months" written as "5 more months" is a new figure).
        if kind in ("markup", "number", "identifier") and got > want.get((kind, token), 0):
            problems.append(f"{kind} {token!r} is not in the source; remove it")
    return problems


# A rewrite this much longer than its source has invented something (models
# fill a near-empty request with text from the examples); this much shorter,
# it has dropped something.
MAX_GROWTH, GROWTH_SLACK_WORDS = 1.35, 8
MIN_SHRINK, SHRINK_MIN_WORDS = 0.6, 20


def length_change(source: str, rewrite: str) -> list[str]:
    """A problem when ``rewrite`` is far longer or shorter than ``source``."""
    before, after = len(source.split()), len(rewrite.split())
    if after > before * MAX_GROWTH + GROWTH_SLACK_WORDS:
        return [
            f"the rewrite has {after} words for the original's {before}; add nothing "
            "the original does not say"
        ]
    if before >= SHRINK_MIN_WORDS and after < before * MIN_SHRINK:
        return [
            f"the rewrite has {after} words for the original's {before}; keep every "
            "point the original makes"
        ]
    return []


# A capitalized word after a lower-case word or a comma is a name, not a
# sentence opening; a rewrite must name the same people and things as often.
_NAME_RE = re.compile(r"(?<=[a-z,;:(] )[A-Z][a-z]+(?:[A-Z][a-z]+)*(?: [A-Z][A-Za-z]+)*\b")
# Acronyms (MSD, BVI, IP) name things too, wherever they stand.
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&]{1,7}\b")
# "US" (the country) is not "us"; "I" is always upper case.
_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|[Mm]e|[Mm]y|[Mm]ine|[Mm]yself|[Ww]e|[Uu]s|[Oo]urs?|[Oo]urselves)\b"
)
_SECOND_PERSON_RE = re.compile(r"\b(?:[Yy]ou|[Yy]ours?|[Yy]ourself|[Yy]ourselves)\b")


def person_changes(source: str, rewrite: str) -> list[str]:
    """A rewrite that speaks as the author more often than the original
    ("while we negotiated" for "while the Series A was negotiated") has
    given someone else's actions to the author; one that adds or drops
    "you" has changed whom it addresses.  (I becoming we is fine.)"""
    problems = []
    first = len(_FIRST_PERSON_RE.findall(source)), len(_FIRST_PERSON_RE.findall(rewrite))
    if first[1] > first[0]:
        problems.append(
            f"the rewrite says I/we/our {first[1]}x where the original says it {first[0]}x; "
            "never make the author the subject of what someone else did"
        )
    second = len(_SECOND_PERSON_RE.findall(source)), len(_SECOND_PERSON_RE.findall(rewrite))
    if second[1] != second[0]:
        problems.append(
            f"the rewrite says you/your {second[1]}x where the original says it "
            f"{second[0]}x; address the reader exactly as the original does"
        )
    return problems


def name_changes(source: str, rewrite: str) -> list[str]:
    """Names (of people, companies, places) the rewrite dropped or added:
    a model that writes "Korotkoff" where the original said "he" has
    changed who did what."""
    problems = []
    names = {m.group(0) for m in _NAME_RE.finditer(source)}
    names |= {m.group(0) for m in _ACRONYM_RE.finditer(source)}
    for name in sorted(names):
        word = re.compile(rf"\b{re.escape(name)}\b")
        before, after = len(word.findall(source)), len(word.findall(rewrite))
        if before != after:
            problems.append(
                f"name {name!r} appears {before}x in the source but {after}x after; "
                "keep who did what exactly as the original says"
            )
    return problems


# Words that decide what a sentence claims: dropping a "not" or turning
# "will" into "could" changes the meaning, however natural it reads.
_NEGATIONS = frozenset("not never no nor none nothing neither nobody nowhere without".split())
_MODALS = frozenset("may might must will would shall should can could".split())
_CONTRACTED = (
    (re.compile(r"\bcan't\b|\bcannot\b", re.I), "can not"),
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"\bshan't\b", re.I), "shall not"),
    (re.compile(r"n't\b", re.I), " not"),
    (re.compile(r"'ll\b", re.I), " will"),
)
_PLAIN_WORD_RE = re.compile(r"[a-z]+")


def _claim_words(text: str) -> Counter:
    text = text.replace("\u2019", "'")
    for pattern, expanded in _CONTRACTED:
        text = pattern.sub(expanded, text)
    found: Counter = Counter()
    for word in _PLAIN_WORD_RE.findall(text.lower()):
        if word in _MODALS:
            found[word] += 1
        elif word in _NEGATIONS:
            found["negation"] += 1
    return found


def claim_changes(source: str, rewrite: str) -> list[str]:
    """Negations and modal verbs the rewrite added, dropped or swapped."""
    before, after = _claim_words(source), _claim_words(rewrite)
    problems = []
    for word in sorted(set(before) | set(after)):
        if before[word] != after[word]:
            what = "negations (not, no, never)" if word == "negation" else f"{word!r}"
            problems.append(
                f"the rewrite changes {what}: {before[word]}x in the original, "
                f"{after[word]}x after; keep what is and is not claimed, and how surely"
            )
    return problems


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"\u201d')]*\s+|\n+")
_CONTENT_RE = re.compile(r"[a-z][a-z'-]{3,}")
# A sentence whose content words (stems) mostly have no counterpart on the
# other side was dropped or invented; a reworded one keeps most of them.
MIN_SHARED = 1 / 3


def _stems(text: str) -> set[str]:
    return {w[:4] for w in _CONTENT_RE.findall(text.lower()) if w not in _FUNCTION_WORDS}


def sentence_changes(source: str, rewrite: str) -> list[str]:
    """Sentences of the original with no counterpart in the rewrite, and
    sentences of the rewrite the original has nothing like."""
    problems = []
    for here, there, verb, advice in (
        (source, rewrite, "drops", "keep every point the original makes"),
        (rewrite, source, "adds", "add nothing the original does not say"),
    ):
        other = _stems(there)
        for sentence in _SENTENCE_END_RE.split(here):
            stems = _stems(sentence)
            if len(stems) >= 2 and len(stems & other) < len(stems) * MIN_SHARED:
                problems.append(f"the rewrite {verb} the sentence {sentence.strip()!r}; {advice}")
    return problems


# Spans that style fixes must leave byte for byte as they are.
_VERBATIM_KINDS = ("inline_code", "url", "email", "quotation", "bold_label", "markup")


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ``(start, end)`` ranges of verbatim tokens."""
    spans = sorted(m.span() for kind in _VERBATIM_KINDS for m in _PATTERNS[kind].finditer(text))
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


# What the model never sees: each such span is swapped for a short
# placeholder before the request and put back after, so a model cannot drop
# a digit of a Bates number or reword a quotation; it only has to keep the
# placeholders. Earlier kinds win where spans overlap (a number in a quote).
_MASKED_KINDS = (
    "inline_code",
    "url",
    "email",
    "quotation",
    "bold_label",  # before numbers, which may sit inside one
    "citation",
    "identifier",
    "number",
    "label",
)
_PLACEHOLDER_RE = re.compile(r"\{\{(\d+)\}\}")
_QUOTED_PLACEHOLDER_RE = re.compile(r"([\"\u201c]?)\{\{(\d+)\}\}([\"\u201d]?)([,;:.]?)")


def mask(text: str) -> tuple[str, list[str]]:
    """``text`` with its protected spans replaced by ``{{1}}``, ``{{2}}``...
    in order, and the spans they stand for.  Text that already contains
    ``{{`` is left as it is (its own braces could not be told apart)."""
    if "{{" in text:
        return text, []
    # Paragraph markers and formatting tags travel as they are.
    kept = [m.span() for m in _PATTERNS["markup"].finditer(text)]
    taken: list[tuple[int, int]] = []
    for kind in _MASKED_KINDS:
        for match in _PATTERNS[kind].finditer(text):
            start, end = match.span()
            if not any(start < e and s < end for s, e in taken + kept):
                taken.append((start, end))
    taken.sort()
    out, spans, last = [], [], 0
    for n, (start, end) in enumerate(taken, 1):
        out.append(text[last:start] + f"{{{{{n}}}}}")
        spans.append(text[start:end])
        last = end
    return "".join(out) + text[last:], spans


def unmask(text: str, spans: list[str]) -> tuple[str, list[str]]:
    """Put ``spans`` back in place of their placeholders; the problems list
    names every placeholder that was repeated or invented (a lost one leaves
    its words missing, which :func:`missing` reports)."""
    counts: dict[int, int] = {}
    for match in _PLACEHOLDER_RE.finditer(text):
        n = int(match.group(1))
        counts[n] = counts.get(n, 0) + 1
    problems = [
        f"placeholder {{{{{n}}}}} appears {counts.get(n, 0)}x; keep each placeholder "
        "exactly once, where its words belong"
        for n in range(1, len(spans) + 1)
        if counts.get(n, 0) > 1
    ]
    problems += [
        f"placeholder {{{{{n}}}}} is not in the original; remove it"
        for n in sorted(counts)
        if not 1 <= n <= len(spans)
    ]

    def restore(match: re.Match) -> str:
        n = int(match.group(2))
        if not 1 <= n <= len(spans):
            return ""
        span, after = spans[n - 1], match.group(4)
        if span[0] in '"\u201c':  # a quotation: drop quote marks the model added
            # and a comma after its own comma, or a full stop after its own
            # full stop, question or exclamation mark
            if after and span[-2:-1] in (".?!" if after == "." else ",."):
                after = ""
        else:
            span = match.group(1) + span + match.group(3)
        return span + after

    restored = _QUOTED_PLACEHOLDER_RE.sub(restore, text)
    return restored, problems


_TRAILING = ".,;:!?"


def repair_quotations(source: str, rewrite: str) -> str:
    """Undo the one change models make to quotations on their own: moving a
    trailing comma or full stop across the closing quote mark (American
    style), in either direction.  The quoted words must otherwise be
    identical; anything else is left for :func:`missing` to report."""
    for token in dict.fromkeys(m.group(0) for m in _PATTERNS["quotation"].finditer(source)):
        if token in rewrite:
            continue
        opener, inner, closer = token[0], token[1:-1], token[-1]
        for mark in _TRAILING:
            moved_in = f"{opener}{inner}{mark}{closer}"
            if moved_in in rewrite:  # "Policy" -> "Policy."
                rewrite = rewrite.replace(moved_in, f"{token}{mark}")
                break
            if inner.endswith(mark):  # "Plus," -> "Plus", (or the comma dropped)
                moved_out = f"{opener}{inner[:-1]}{closer}"
                rewrite = rewrite.replace(moved_out + mark, token)
                rewrite = re.sub(re.escape(moved_out) + r"(?![.,;:!?])", token, rewrite)
    return rewrite


def _script(char: str) -> str | None:
    """The writing system of a letter outside basic and extended Latin."""
    code = ord(char)
    if 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
        return "CJK"  # ideographic punctuation and full-width forms
    if code <= 0x24F or not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    script = name.split(" ", 1)[0] if name else "UNKNOWN"
    return "CJK" if script in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL") else script


def foreign_script(source: str, rewrite: str) -> list[str]:
    """Text in a writing system the source never uses (a model drifting into
    Chinese mid-sentence, say), reported with a sample."""
    allowed = {_script(c) for c in source}
    found: dict[str, list[str]] = {}
    for char in rewrite:
        script = _script(char)
        if script is not None and script not in allowed:
            found.setdefault(script, []).append(char)
    return [
        f"the rewrite switched to {script.title()} text ({''.join(chars[:12])}); "
        "write only in the language of the original"
        for script, chars in found.items()
    ]
