# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""How close a piece of text is to a voice, and how to steer it closer.

``distance`` is a Burrows' Delta style measure: the mean relative gap
between the text's function-word rates and the author's, plus gaps in
sentence length, word length and contraction rate, each gap capped so no
single feature dominates a short text.  Lower is closer; it is
used to choose among several candidate rewrites and by ``proselign eval``.
``drift`` turns large, reliable gaps into feedback for another attempt.
"""

from __future__ import annotations

import re

from proselign.ingest.lexicon import FUNCTION_WORDS
from proselign.ingest.stylometry import measure, words
from proselign.voice.profile import Voice

_MARKUP = re.compile(r"\[\[\d+\]\]|</?s\d+>")
MIN_WORDS_TO_STEER = 80
STEER_STRENGTH = 0.5


def _plain(text: str) -> str:
    return _MARKUP.sub("", text)


def function_word_rates(text: str) -> dict[str, float]:
    """Per-1,000-word rates of the common function words in ``text``."""
    tokens = [w.lower() for w in words(text)]
    total = len(tokens) or 1
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return {w: counts.get(w, 0) * 1000 / total for w in FUNCTION_WORDS}


def distance(text: str, voice: Voice) -> float:
    """Stylistic distance from ``text`` to ``voice`` (0 is identical)."""
    text = _plain(text)
    target = voice.lexicon.get("function_words", {}) if voice.lexicon else {}
    rates = function_word_rates(text)
    common = sorted(target, key=target.get, reverse=True)[:50]
    lexical = (
        sum(min(abs(rates[w] - target[w]) / (target[w] + 5.0), 1.0) for w in common) / len(common)
        if common
        else 0.0
    )
    metrics, wanted = measure(text), voice.metrics or {}
    gaps = []
    for read in (
        lambda m: m["sentence_length_words"]["mean"],
        lambda m: m["mean_word_length"],
        lambda m: m["contractions_per_1k"],
    ):
        try:
            mine, theirs = read(metrics), read(wanted)
        except (KeyError, TypeError):
            continue
        # Capped, so one quirk of a short text (a single contraction) cannot dominate.
        gaps.append(min(abs(mine - theirs) / (abs(theirs) + 2.0), 1.0))
    return round(lexical + (sum(gaps) / len(gaps) if gaps else 0.0), 4)


def drift(text: str, voice: Voice, strength: float) -> list[str]:
    """Feedback for a rewrite that is clearly off the author's profile."""
    text = _plain(text)
    count = len(words(text))
    wanted = voice.metrics or {}
    if strength < STEER_STRENGTH or count < MIN_WORDS_TO_STEER or not wanted:
        return []
    metrics = measure(text)
    out = []
    mine = metrics["sentence_length_words"]["mean"]
    theirs = wanted.get("sentence_length_words", {}).get("mean", 0)
    if theirs and mine > theirs * 1.35:
        out.append(
            f"match the author's sentence length: yours average {mine:.0f} words, the "
            f"author's {theirs:.0f}; split the long sentences"
        )
    elif theirs and mine < theirs * 0.65:
        out.append(
            f"match the author's sentence length: yours average {mine:.0f} words, the "
            f"author's {theirs:.0f}; combine the choppy ones"
        )
    contractions = wanted.get("contractions_per_1k", 0)
    if contractions >= 8 and count >= 150 and metrics["contractions_per_1k"] == 0:
        out.append("match the author's contractions: use them as the author does (it's, don't)")
    elif contractions < 3 and metrics["contractions_per_1k"] >= 8:
        out.append("match the author's formality: write contractions out in full")
    return out
