# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Shared limits and errors for the document text extractors.

Every extractor parses untrusted bytes: all offsets are bounds-checked,
every loop over file-supplied links is capped, and every decompression has
an output ceiling, so a hostile file fails with :class:`ExtractError`
instead of hanging, exhausting memory or crashing.
"""

from __future__ import annotations

import zlib

MAX_OUTPUT_BYTES = 64 * 1024 * 1024


class ExtractError(Exception):
    """A document whose text cannot be extracted (corrupt, encrypted, unsupported)."""


def inflate(data: bytes, limit: int = MAX_OUTPUT_BYTES) -> bytes:
    """Zlib-decompress ``data``, refusing to produce more than ``limit`` bytes."""
    decoder = zlib.decompressobj()
    try:
        out = decoder.decompress(data, limit)
    except zlib.error as exc:
        raise ExtractError(f"corrupt compressed data: {exc}") from exc
    if decoder.unconsumed_tail:
        raise ExtractError("compressed data expands beyond the size limit")
    return out


def tidy(paragraphs: list[str]) -> str:
    """Join paragraphs with blank lines, folding whitespace inside each."""
    cleaned = (" ".join(p.split()) for p in paragraphs)
    return "\n\n".join(p for p in cleaned if p)
