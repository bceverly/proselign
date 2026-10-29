# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Protected spans: the tokens a rewrite must carry through untouched.

Preflight extracts them from the source; postflight checks the rewrite still
contains every one, as many times as the source did. A rewrite that drops a
figure, URL, citation or quotation is a changed meaning, not a style change.
"""

from __future__ import annotations

import re
from collections import Counter

_PATTERNS = {
    "inline_code": re.compile(r"`[^`\n]+`"),
    "url": re.compile(r"\b(?:https?|ftp)://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"]"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "quotation": re.compile(r"\"[^\"\n]{1,400}\"|“[^”\n]{1,400}”"),
    "citation": re.compile(r"\[\d+(?:[,–-]\s*\d+)*\]|\[@[\w:-]+\]"),
    "number": re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?(?![\w])"),
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
    return problems
