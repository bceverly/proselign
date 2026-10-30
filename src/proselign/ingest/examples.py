# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Representative passages of the author's own writing, kept with the voice.

Models imitate examples far better than descriptions, so training keeps a
few hundred typical paragraphs.  A passage qualifies when it is a whole
prose paragraph of moderate length; the most typical ones (sentence length
closest to the author's usual) are taken from every document in turn, so
no single long document dominates.  They stay on this machine: export
leaves them out unless the voice is marked shareable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from proselign.ingest.stylometry import split_paragraphs, split_sentences, words

MAX_EXAMPLES = 200
MIN_WORDS, MAX_WORDS = 25, 160
_ENDS_LIKE_PROSE = re.compile(r"[.!?][\"'”’)]*\Z")
_NOISE = re.compile(r"https?://|@|\t|\|")


def select_examples(texts: Iterable[str], mean_sentence: float) -> list[str]:
    """Up to ``MAX_EXAMPLES`` typical paragraphs, most typical first."""
    per_document = [_ranked(text, mean_sentence) for text in texts]
    chosen: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(chosen) < MAX_EXAMPLES and any(depth < len(d) for d in per_document):
        for ranked in per_document:
            if depth < len(ranked) and len(chosen) < MAX_EXAMPLES:
                key = " ".join(ranked[depth].lower().split())
                if key not in seen:
                    seen.add(key)
                    chosen.append(ranked[depth])
        depth += 1
    return chosen


def _ranked(text: str, mean_sentence: float) -> list[str]:
    candidates = []
    for paragraph in split_paragraphs(text):
        count = len(words(paragraph))
        if not MIN_WORDS <= count <= MAX_WORDS or not _ENDS_LIKE_PROSE.search(paragraph):
            continue
        if _NOISE.search(paragraph) or sum(c.isdigit() for c in paragraph) > len(paragraph) / 10:
            continue
        sentences = split_sentences(paragraph)
        own_mean = count / max(len(sentences), 1)
        candidates.append((abs(own_mean - mean_sentence) / max(mean_sentence, 1.0), paragraph))
    return [paragraph for _, paragraph in sorted(candidates, key=lambda pair: pair[0])]
