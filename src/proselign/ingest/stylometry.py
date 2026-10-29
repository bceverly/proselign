# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Objective, reproducible measurements of a writing sample.

Everything here is plain counting: no model, no network, deterministic
output for the same input. These numbers anchor the style profile.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

# Applied to whitespace-folded paragraphs, so a single literal space suffices;
# a ``\s+`` before the lookahead would backtrack on long whitespace runs.
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]{0,3} (?=[A-Z0-9\"'(\[])")
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_CONTRACTION_RE = re.compile(r"\b[a-z]+['’](?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.IGNORECASE
)

FIRST_PERSON = frozenset({"i", "me", "my", "mine", "we", "us", "our", "ours"})
SECOND_PERSON = frozenset({"you", "your", "yours"})
PUNCTUATION = {
    "semicolons": ";",
    "colons": ":",
    "em_dashes": "—",
    "exclamations": "!",
    "questions": "?",
    "parentheses": "(",
}


def split_sentences(text: str) -> list[str]:
    """Rough sentence segmentation, good enough for distributions."""
    sentences = []
    for paragraph in split_paragraphs(text):
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip())
    return sentences


def split_paragraphs(text: str) -> list[str]:
    """Blank-line separated blocks, with internal newlines folded."""
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]


def words(text: str) -> list[str]:
    """Word tokens, apostrophes kept inside words."""
    return _WORD_RE.findall(text)


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": float(statistics.median(values)),
        "stdev": round(statistics.pstdev(values), 2),
    }


def measure(text: str) -> dict:
    """Compute the stylometric feature set for ``text``."""
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    tokens = words(text)
    lowered = [w.lower() for w in tokens]
    n_words = len(tokens) or 1
    per_1k = 1000.0 / n_words

    counts = Counter(lowered)
    return {
        "words": len(tokens),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "sentence_length_words": _distribution([len(words(s)) for s in sentences]),
        "paragraph_length_sentences": _distribution([len(split_sentences(p)) for p in paragraphs]),
        "mean_word_length": round(sum(len(w) for w in tokens) / n_words, 2),
        "type_token_ratio": round(len(counts) / n_words, 3),
        "contractions_per_1k": round(len(_CONTRACTION_RE.findall(text)) * per_1k, 2),
        "first_person_per_1k": round(sum(counts[w] for w in FIRST_PERSON) * per_1k, 2),
        "second_person_per_1k": round(sum(counts[w] for w in SECOND_PERSON) * per_1k, 2),
        "passive_per_1k": round(len(_PASSIVE_RE.findall(text)) * per_1k, 2),
        "punctuation_per_1k": {
            name: round(text.count(char) * per_1k, 2) for name, char in PUNCTUATION.items()
        },
    }


def describe(metrics: dict) -> dict:
    """Turn measurements into the coarse, human-readable ``style`` hints.

    Deliberately conservative: thresholds only, no claims a count cannot back.
    """
    tone: list[str] = []
    if metrics["contractions_per_1k"] >= 8:
        tone.append("conversational")
    if metrics["first_person_per_1k"] >= 15:
        tone.append("personal")
    if metrics["second_person_per_1k"] >= 10:
        tone.append("direct-address")
    if metrics["mean_word_length"] >= 5.0:
        tone.append("technical")

    mean_sentence = metrics["sentence_length_words"]["mean"]
    if mean_sentence and mean_sentence < 14:
        sentences = "short"
    elif mean_sentence > 24:
        sentences = "long"
    else:
        sentences = "medium"

    paragraph_mean = metrics["paragraph_length_sentences"]["mean"]
    return {
        "tone": tone,
        "sentence_length": sentences,
        "paragraphing": "compact" if paragraph_mean and paragraph_mean <= 3 else "extended",
        "uses_contractions": metrics["contractions_per_1k"] >= 3,
    }
