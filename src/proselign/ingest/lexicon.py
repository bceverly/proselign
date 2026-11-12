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
* ``conventions``: serial (Oxford) comma and American or British spelling;
* ``phrases``: three-word phrases the author returns to across many
  documents ("at this point", "first things first"), never names or topics;
* ``asides``: the author's recurring interjections, such as ``<grin>``.
"""

from __future__ import annotations

import re
from collections import Counter

from proselign.ingest.stylometry import split_paragraphs, split_sentences, words

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
_OPENER = re.compile(r"^([A-Z][\w']*(?: [\w']+){0,2}), (?:[\w'\"(]+[ ,;:]+){3}")
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
_PHRASE_WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
# Asides in angle brackets or parentheses, and emoticons.
_ASIDE = re.compile(r"<[a-z]{1,12}>|\((?:grin|g|sigh|smile|lol|wink)\)|[:;]-?[)D]")
MAX_PHRASES = 8
MIN_PHRASE_DOCUMENTS = 3
MIN_ASIDES = 3
# Below this many documents (one mailbox, a few long files), spread is
# counted over paragraphs instead.
MIN_DOCUMENTS_FOR_SPREAD = 10


def build_lexicon(text: str, documents: list[str] | None = None) -> dict:
    """The vocabulary fingerprint of ``text``; ``documents`` (the separate
    pieces ``text`` was joined from) let recurring phrases be told apart from
    one document's topic."""
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
        "phrases": _phrases(documents if documents is not None else [text]),
        "asides": [a for a, n in Counter(_ASIDE.findall(text)).most_common(4) if n >= MIN_ASIDES],
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


def _phrases(documents: list[str]) -> list[str]:
    """Three-word phrases found in many separate documents.

    A phrase with a capitalized word (other than I or a sentence's first
    word) is a name or a topic, and one made only of function words carries
    no voice; both are skipped.  A phrase overlapping a more widespread one
    ("this point I" after "at this point") is skipped too.
    """
    if len(documents) >= MIN_DOCUMENTS_FOR_SPREAD:
        floor = max(MIN_PHRASE_DOCUMENTS, len(documents) // 5)
    else:
        documents = [p for d in documents for p in split_paragraphs(d)]
        floor = max(MIN_PHRASE_DOCUMENTS, len(documents) // 100)
    function = frozenset(FUNCTION_WORDS)
    spread: Counter[tuple[str, ...]] = Counter()
    total: Counter[tuple[str, ...]] = Counter()
    for document in documents:
        seen: dict[tuple[str, ...], None] = {}  # ordered: ties go to the first seen
        for sentence in split_sentences(document):
            tokens = _PHRASE_WORD.findall(sentence)
            for i in range(len(tokens) - 2):
                gram = tokens[i : i + 3]
                if any(t[0].isupper() and t != "I" and i + j for j, t in enumerate(gram)):
                    continue
                lowered = tuple(t.lower() for t in gram)
                if not set(lowered) <= function and lowered[0] != "to":
                    seen[lowered] = None
                    total[lowered] += 1
        spread.update(seen.keys())
    kept: list[tuple[str, ...]] = []
    for gram, count in sorted(spread.items(), key=lambda g: (-g[1], -total[g[0]])):
        if count < floor or len(kept) == MAX_PHRASES:
            break
        if not any(gram[:2] == k[1:] or gram[1:] == k[:2] or gram[1:] == k[1:] for k in kept):
            kept.append(gram)
    return [" ".join("I" if t == "i" else t for t in gram) for gram in kept]
