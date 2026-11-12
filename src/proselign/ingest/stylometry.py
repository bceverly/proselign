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

_SPACED_DASH_RE = re.compile(r"\w[,.]? (?:\u2013|-{1,2}) \w")
_ELLIPSIS_RE = re.compile(r"\u2026|\.\.\.")
_CONJUNCTION_START_RE = re.compile(r"^[\"'(\u201c]?(?:And|But|So|Or|Yet)\b")
_LY_RE = re.compile(r"\b[a-z]{3,}ly\b")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
# Not adverbs, though they end in -ly.
_NOT_LY_ADVERBS = frozenset(
    {"only", "family", "early", "likely", "apply", "reply", "supply", "rely", "july", "italy"}
    | {"daily", "weekly", "monthly", "yearly", "friendly", "lovely", "ugly", "holy", "jelly"}
)
SHORT_SENTENCE_WORDS = 8
LONG_SENTENCE_WORDS = 30
HEDGES = frozenset({"maybe", "perhaps", "probably", "possibly", "arguably", "somewhat", "seems"})
INTENSIFIERS = frozenset(
    {"very", "really", "pretty", "quite", "extremely", "totally", "absolutely", "super"}
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
_PATTERN_PUNCTUATION = {"ellipses": _ELLIPSIS_RE, "spaced_dashes": _SPACED_DASH_RE}


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


def syllables(word: str) -> int:
    """Approximate syllable count: vowel groups, less a silent final e."""
    word = word.lower()
    count = len(_VOWEL_GROUPS.findall(word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def _readability(n_words: int, n_sentences: int, tokens: list[str]) -> dict[str, float]:
    """Flesch reading ease and Flesch-Kincaid grade (higher ease is easier)."""
    if not n_words or not n_sentences:
        return {"flesch_reading_ease": 0.0, "grade_level": 0.0}
    per_sentence = n_words / n_sentences
    per_word = sum(syllables(w) for w in tokens) / n_words
    return {
        "flesch_reading_ease": round(206.835 - 1.015 * per_sentence - 84.6 * per_word, 1),
        "grade_level": round(0.39 * per_sentence + 11.8 * per_word - 15.59, 1),
    }


def _rhythm(sentences: list[str], lengths: list[int]) -> dict[str, float]:
    """Shares of short, long and conjunction-led sentences, and comma density."""
    n = len(sentences) or 1
    return {
        "short_share": round(sum(1 for x in lengths if x <= SHORT_SENTENCE_WORDS) / n, 3),
        "long_share": round(sum(1 for x in lengths if x >= LONG_SENTENCE_WORDS) / n, 3),
        "conjunction_start_share": round(
            sum(1 for s in sentences if _CONJUNCTION_START_RE.match(s)) / n, 3
        ),
        "commas_per_sentence": round(sum(s.count(",") for s in sentences) / n, 2),
    }


def measure(text: str) -> dict:
    """Compute the stylometric feature set for ``text``."""
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    tokens = words(text)
    lowered = [w.lower() for w in tokens]
    n_words = len(tokens) or 1
    per_1k = 1000.0 / n_words
    lengths = [len(words(s)) for s in sentences]

    counts = Counter(lowered)
    adverbs = sum(n for w, n in counts.items() if _LY_RE.fullmatch(w) and w not in _NOT_LY_ADVERBS)
    return {
        "words": len(tokens),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "sentence_length_words": _distribution(lengths),
        "sentence_rhythm": _rhythm(sentences, lengths),
        "paragraph_length_sentences": _distribution([len(split_sentences(p)) for p in paragraphs]),
        "mean_word_length": round(sum(len(w) for w in tokens) / n_words, 2),
        "type_token_ratio": round(len(counts) / n_words, 3),
        "contractions_per_1k": round(len(_CONTRACTION_RE.findall(text)) * per_1k, 2),
        "first_person_per_1k": round(sum(counts[w] for w in FIRST_PERSON) * per_1k, 2),
        "second_person_per_1k": round(sum(counts[w] for w in SECOND_PERSON) * per_1k, 2),
        "passive_per_1k": round(len(_PASSIVE_RE.findall(text)) * per_1k, 2),
        "hedges_per_1k": round(sum(counts[w] for w in HEDGES) * per_1k, 2),
        "intensifiers_per_1k": round(sum(counts[w] for w in INTENSIFIERS) * per_1k, 2),
        "ly_adverbs_per_1k": round(adverbs * per_1k, 2),
        "readability": _readability(len(tokens), len(sentences), tokens),
        "punctuation_per_1k": {
            **{name: round(text.count(char) * per_1k, 2) for name, char in PUNCTUATION.items()},
            **{
                name: round(len(pattern.findall(text)) * per_1k, 2)
                for name, pattern in _PATTERN_PUNCTUATION.items()
            },
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
