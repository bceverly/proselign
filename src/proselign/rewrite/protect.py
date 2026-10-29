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

_PATTERNS = {
    "inline_code": re.compile(r"`[^`\n]+`"),
    "url": re.compile(r"\b(?:https?|ftp)://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"]"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "quotation": re.compile(r"\"[^\"\n]{1,400}\"|“[^”\n]{1,400}”"),
    "citation": re.compile(r"(?<!\[)\[\d+(?:[,–-]\s*\d+)*\](?!\])|\[@[\w:-]+\]"),
    # Thousands separators only between digits: "42, all" is the number 42.
    # Paragraph markers and formatting tags used when rewriting .docx files.
    "markup": re.compile(r"\[\[\d+\]\]|</?s\d+>"),
    "number": re.compile(r"(?<![\w.])[-+]?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?(?![\w])"),
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
        # Markers and formatting tags must not be invented either.
        if kind == "markup" and got > want.get((kind, token), 0):
            problems.append(f"{kind} {token!r} is not in the source; remove it")
    return problems


# Spans that style fixes must leave byte for byte as they are.
_VERBATIM_KINDS = ("inline_code", "url", "email", "quotation", "markup")


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
