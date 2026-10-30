# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The author's vocabulary fingerprint: word choices, openers, conventions.

Rates describe rhythm; this describes diction.  Everything is counting
against fixed lists, so it is deterministic and needs no reference corpus:

* ``function_words``: per-1,000 rates of common function words, the classic
  authorship signal (Burrows' Delta), used to score how close a rewrite is;
* ``prefers``: which of two near-synonyms the author picks ("use" over
  "utilize"), when the corpus shows a clear habit;
* ``avoids``: words language models love that the author never writes;
* ``openers``: comma-led sentence openers the author keeps returning to;
* ``conventions``: serial (Oxford) comma and American or British spelling.
"""

from __future__ import annotations

import re
from collections import Counter

from proselign.ingest.stylometry import split_sentences, words

FUNCTION_WORDS = (
    "a about after all also an and any are as at be because been but by can could "
    "did do does even for from had has have he her here his how i if in into is it "
    "its just like may me more most much must my no not now of on one only or other "
    "our out over own same she should so some such than that the their them then "
    "there these they this those through to too under up us very was we were what "
    "when where which while who why will with would yet you your"
).split()

# (the plain word the author might prefer, the inflated alternatives)
_SYNONYMS = (
    ("use", r"use[sd]?|using", ("utilize", r"utili[sz](?:e[sd]?|ing|ation)")),
    ("help", r"help(?:s|ed|ing)?", ("facilitate", r"facilitat(?:e[sd]?|ing)")),
    ("show", r"show(?:s|ed|n|ing)?", ("demonstrate", r"demonstrat(?:e[sd]?|ing)")),
    ("start", r"start(?:s|ed|ing)?", ("commence", r"commenc(?:e[sd]?|ing)")),
    ("about", r"about", ("approximately", r"approximately")),
    ("buy", r"buy(?:s|ing)?|bought", ("purchase", r"purchas(?:e[sd]?|ing)")),
    ("need", r"need(?:s|ed|ing)?", ("require", r"requir(?:e[sd]?|ing)")),
    ("get", r"get(?:s|ting)?|got", ("obtain", r"obtain(?:s|ed|ing)?")),
    ("enough", r"enough", ("sufficient", r"sufficient(?:ly)?")),
    ("try", r"tr(?:y|ies|ied|ying)", ("attempt", r"attempt(?:s|ed|ing)?")),
    ("make sure", r"make sure|making sure|made sure", ("ensure", r"ensur(?:e[sd]?|ing)")),
    ("improve", r"improv(?:e[sd]?|ing)", ("enhance", r"enhanc(?:e[sd]?|ing)")),
    ("also", r"also", ("additionally", r"additionally")),
    ("but", r"but", ("however", r"however")),
    ("so", r"so", ("therefore", r"therefore|thus|hence")),
)
MODEL_WORDS = (
    "utilize leverage facilitate delve robust seamless seamlessly comprehensive crucial "
    "pivotal foster streamline enhance additionally furthermore moreover notably showcase "
    "underscore meticulous holistic synergy paradigm empower bolster elevate intricate "
    "tapestry myriad plethora realm landscape navigate unlock embark vibrant"
).split()
# An opener must lead into a sentence; "Jeff," on its own is a greeting.
_OPENER = re.compile(r"^((?:[A-Z][\w']*)(?: [\w']+){0,2}), (?:[\w'\"(]+[ ,;:]+){3}")
_SERIAL = re.compile(r"\b\w+, \w+(?: \w+)?, (?:and|or) \w")
_NO_SERIAL = re.compile(r"\b\w+, \w+(?: \w+)? (?:and|or) \w")
_US = re.compile(
    r"\b(?:organiz|realiz|recogniz|prioritiz|analyz|summariz|minimiz|emphasiz)\w*"
    r"|\b(?:color|behavior|favor|honor|labor|center|theater|defense)s?\b",
    re.IGNORECASE,
)
_UK = re.compile(
    r"\b(?:organis|realis|recognis|prioritis|analys|summaris|minimis|emphasis)"
    r"(?:e|ed|es|ing|ation)\w*"
    r"|\b(?:colour|behaviour|favour|honour|labour|centre|theatre|defence)s?\b",
    re.IGNORECASE,
)
MIN_WORDS_FOR_ABSENCE = 5000


def build_lexicon(text: str) -> dict:
    """The vocabulary fingerprint of ``text``."""
    tokens = [w.lower() for w in words(text)]
    total = len(tokens) or 1
    counts = Counter(tokens)
    lowered = text.lower()
    return {
        "function_words": {
            w: round(counts[w] * 1000 / total, 2) for w in FUNCTION_WORDS if counts[w]
        },
        "prefers": _preferences(lowered),
        "avoids": (
            [w for w in MODEL_WORDS if not counts[w]] if total >= MIN_WORDS_FOR_ABSENCE else []
        ),
        "openers": _openers(text),
        "conventions": _conventions(text),
    }


def _count(pattern: str, text: str) -> int:
    return len(re.findall(rf"\b(?:{pattern})\b", text))


def _preferences(text: str) -> list[dict]:
    out = []
    for plain, plain_re, (fancy, fancy_re) in _SYNONYMS:
        mine, theirs = _count(plain_re, text), _count(fancy_re, text)
        if mine + theirs >= 3 and mine >= 4 * theirs:
            out.append({"prefer": plain, "over": fancy})
    return out


def _openers(text: str) -> list[str]:
    sentences = split_sentences(text)
    found = Counter(m.group(1) for s in sentences if (m := _OPENER.match(s)))
    floor = max(3, len(sentences) // 250)
    return [phrase for phrase, count in found.most_common(8) if count >= floor]


def _conventions(text: str) -> dict:
    out = {}
    serial, plain = len(_SERIAL.findall(text)), len(_NO_SERIAL.findall(text))
    if serial + plain >= 5 and max(serial, plain) >= 2 * min(serial, plain):
        out["serial_comma"] = serial > plain
    us, uk = len(_US.findall(text)), len(_UK.findall(text))
    if us + uk >= 3 and max(us, uk) >= 3 * min(us, uk):
        out["spelling"] = "American" if us > uk else "British"
    return out
