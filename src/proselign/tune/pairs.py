# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Training pairs for a fine-tune: neutral prose in, the author's prose out.

The author's documents are cut into passages of a paragraph or a few;
each passage is masked (numbers, quotations, document IDs become ``{{n}}``,
as in a rewrite) and flattened into neutral prose by the model.  A
neutral version that lost or invented a placeholder, or changed length
wildly, is dropped.  Pairs are kept with the voice (private, never
exported) and reused, so a second ``tune`` run only flattens new passages.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from proselign.evaluate import NEUTRAL_RULES
from proselign.inference.base import Backend, Message
from proselign.ingest.stylometry import split_paragraphs, split_sentences
from proselign.paths import write_private_file
from proselign.rewrite import protect
from proselign.rewrite.markers import make_markers

PAIRS_FILE = "tune-pairs.json"
MAX_PAIRS_BYTES = 64 * 1024 * 1024
MIN_WORDS, MAX_WORDS = 40, 220
# A neutral version outside these bounds (relative to the passage) has
# dropped or invented content, and would teach the model to do the same.
MIN_RATIO, MAX_RATIO = 0.6, 1.6

PAIR_RULES = (
    NEUTRAL_RULES
    + "\nKeep every placeholder such as {{3}} exactly once, unchanged, in the sentence "
    "it belongs to, and keep every paragraph break and list marker."
)


def passages(texts: Iterable[str]) -> list[str]:
    """Passages of ``MIN_WORDS`` to ``MAX_WORDS`` words: whole paragraphs,
    joined while short, and long ones cut at sentence boundaries."""
    found: list[str] = []
    for text in texts:
        current: list[str] = []
        size = 0
        for paragraph in split_paragraphs(text):
            for piece in _cut(paragraph):
                count = len(piece.split())
                if current and size + count > MAX_WORDS:
                    if size >= MIN_WORDS:
                        found.append("\n\n".join(current))
                    current, size = [], 0
                current.append(piece)
                size += count
        if current and size >= MIN_WORDS:
            found.append("\n\n".join(current))
    return list(dict.fromkeys(found))


def _cut(paragraph: str) -> list[str]:
    """``paragraph``, or its sentences grouped under ``MAX_WORDS`` if longer."""
    if len(paragraph.split()) <= MAX_WORDS:
        return [paragraph]
    pieces, current = [], ""
    for sentence in split_sentences(paragraph):
        if current and len((current + " " + sentence).split()) > MAX_WORDS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def key_for(passage: str) -> str:
    """The stable identity of a passage in the kept pairs."""
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()


def load_pairs(path: Path) -> dict[str, dict[str, str]]:
    """Pairs kept by earlier runs: passage hash -> source and target.
    A missing, oversized or damaged file just means none are kept."""
    try:
        if path.is_symlink() or path.stat().st_size > MAX_PAIRS_BYTES:
            return {}
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: {"source": entry["source"], "target": entry["target"]}
        for key, entry in data.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("source"), str)
        and isinstance(entry.get("target"), str)
    }


def save_pairs(path: Path, pairs: dict[str, dict[str, str]]) -> None:
    """Keep the pairs for later runs, readable only by the user."""
    write_private_file(path, json.dumps(pairs, indent=1, sort_keys=True).encode("utf-8"))


def neutral_pair(backend: Backend, passage: str) -> dict[str, str] | None:
    """The (neutral, author) pair for ``passage``, both masked, or None when
    the model's neutral version lost or invented a fact."""
    masked, spans = protect.mask(passage)
    begin, end = make_markers()
    messages = [Message("system", PAIR_RULES), Message("user", f"{begin}\n{masked}\n{end}")]
    neutral = backend.rewrite(messages, 0.2).strip()
    restored, problems = protect.unmask(neutral, spans)
    if problems or protect.missing(passage, restored) or "{{" in restored:
        return None
    ratio = len(neutral.split()) / max(len(masked.split()), 1)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        return None
    return {"source": neutral, "target": masked}


def build_pairs(
    backend: Backend,
    todo: list[str],
    kept: dict[str, dict[str, str]],
    parallel: int = 1,
    advance: Callable[[], None] | None = None,
) -> tuple[list[dict[str, str]], int]:
    """A pair for every passage in ``todo`` (reusing ``kept`` ones and
    adding new ones to it), and how many passages could not be paired."""

    def pair(passage: str) -> dict[str, str] | None:
        key = key_for(passage)
        if key not in kept:
            made = neutral_pair(backend, passage)
            if made is not None:
                kept[key] = made
        if advance is not None:
            advance()
        return kept.get(key)

    with ThreadPoolExecutor(max(1, parallel)) as pool:
        results = list(pool.map(pair, todo))
    made = [result for result in results if result is not None]
    return made, len(results) - len(made)
