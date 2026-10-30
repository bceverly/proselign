# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Choose which of the author's passages to show the model for one chunk.

Okapi BM25 over content words picks the passages closest in subject to the
text being rewritten, so a technical paragraph is shown technical writing.
Passages that share nothing fall back to the most typical ones (the order
training stored them in).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from proselign.ingest.lexicon import FUNCTION_WORDS

_WORD = re.compile(r"[a-z][a-z'-]+")
_STOP = frozenset(FUNCTION_WORDS)
K1, B = 1.5, 0.75


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


class Library:
    """A searchable set of example passages."""

    def __init__(self, passages: list[str]) -> None:
        self.passages = passages
        self.terms = [Counter(_terms(p)) for p in passages]
        self.average = sum(sum(t.values()) for t in self.terms) / max(len(passages), 1)
        frequency: Counter = Counter()
        for terms in self.terms:
            frequency.update(terms.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def pick(self, text: str, count: int, max_words: int = 350) -> list[str]:
        """Up to ``count`` passages for ``text``, best first, within ``max_words``."""
        query = set(_terms(text))
        scores = []
        for index, terms in enumerate(self.terms):
            length = sum(terms.values()) or 1
            score = sum(
                self.idf[t]
                * terms[t]
                * (K1 + 1)
                / (terms[t] + K1 * (1 - B + B * length / max(self.average, 1)))
                for t in query & terms.keys()
            )
            scores.append((-score, index))
        chosen, used = [], 0
        for _, index in sorted(scores):
            passage = self.passages[index]
            size = len(passage.split())
            if len(chosen) >= count:
                break
            if used + size <= max_words:
                chosen.append(passage)
                used += size
        return chosen
