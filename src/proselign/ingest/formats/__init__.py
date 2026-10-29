# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Plain text from the document formats a corpus may contain.

Pure standard library, no external programs: ``.docx`` (Office Open XML),
``.doc`` (Word 97-2003 and Word 6/95), ``.pdf`` and ``.mbox`` mailboxes.
Plain text is decoded as UTF-8; Markdown also has its markup removed.
"""

from __future__ import annotations

from pathlib import Path

from proselign.ingest.formats.base import ExtractError
from proselign.ingest.formats.doc import doc_text
from proselign.ingest.formats.docx import docx_text
from proselign.ingest.formats.markdown import markdown_text
from proselign.ingest.formats.mbox import mbox_messages, mbox_text
from proselign.ingest.formats.pdf import pdf_text

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})
MAILBOX_SUFFIXES = frozenset({".mbox"})
_EXTRACTORS = {".docx": docx_text, ".doc": doc_text, ".pdf": pdf_text, ".mbox": mbox_text}
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | frozenset(_EXTRACTORS)

__all__ = [
    "ExtractError",
    "MAILBOX_SUFFIXES",
    "SUPPORTED_SUFFIXES",
    "TEXT_SUFFIXES",
    "extract_parts",
    "extract_text",
]


def extract_text(path: Path, data: bytes) -> str:
    """The text of ``data``, read according to ``path``'s suffix.

    Raises :class:`ExtractError` when a binary document cannot be read.
    """
    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        text = data.decode("utf-8", errors="replace")
        return markdown_text(text) if path.suffix.lower() in _MARKDOWN_SUFFIXES else text
    try:
        return extractor(data)
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, MemoryError) as exc:
        # A defect in a hostile or damaged file must not abort a whole corpus.
        raise ExtractError(f"unreadable {path.suffix.lower()} file ({type(exc).__name__})") from exc


def extract_parts(path: Path, data: bytes) -> list[str]:
    """The separate pieces of writing in ``data``: one per message of a
    mailbox, otherwise the whole file as a single piece.

    Raises :class:`ExtractError` as :func:`extract_text` does.
    """
    if path.suffix.lower() not in MAILBOX_SUFFIXES:
        return [extract_text(path, data)]
    try:
        return mbox_messages(data)
    except (ValueError, IndexError, KeyError, TypeError, OverflowError, MemoryError) as exc:
        raise ExtractError(f"unreadable mailbox ({type(exc).__name__})") from exc
