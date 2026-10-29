# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Collect the documents a voice is learned from.

Plain text and Markdown are read directly; Word (``.docx``, ``.doc``),
PDF and mbox mailbox files go through :mod:`proselign.ingest.formats`.  Documents are data,
never instructions: nothing here interprets their content beyond reading
it as text.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from proselign.errors import UsageError
from proselign.ingest.formats import (
    MAILBOX_SUFFIXES,
    SUPPORTED_SUFFIXES,
    ExtractError,
    extract_parts,
)

MAX_FILE_BYTES = 64 * 1024 * 1024
# Mailboxes are mostly attachments, which are never decoded, so they may be larger.
MAX_MAILBOX_BYTES = 512 * 1024 * 1024

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
_QUOTE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Document:
    """One ingested file; ``parts`` counts the messages in a mailbox."""

    path: Path
    sha256: str
    text: str
    parts: int = 1


def _walk(root: Path) -> list[Path]:
    """Supported files under ``root``: hidden entries and symlinks are skipped.

    Symlinks are never followed inside a directory walk, so a link in a
    corpus cannot pull in files from elsewhere (``~/.ssh``, ``/etc``).
    """
    found = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            path = Path(current) / name
            if (
                not name.startswith(".")
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not path.is_symlink()
                and path.is_file()
            ):
                found.append(path)
    return found


def find_files(paths: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of supported files.

    Files named explicitly are taken as given (the user chose them).
    """
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            found.update(_walk(path))
        elif path.is_file():
            found.add(path)
        else:
            raise UsageError(f"no such file or directory: {path}")
    return sorted(found)


def normalize(text: str) -> str:
    """Strip material that is not the author's own prose.

    Removes front matter, fenced code and block quotes (usually someone
    else's words), and collapses runs of blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FRONT_MATTER_RE.sub("", text)
    text = _FENCE_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(
    paths: list[Path], on_skip: Callable[[Path, str], None] | None = None
) -> list[Document]:
    """Read, normalize and de-duplicate (by content hash) every document.

    Files that are too large or cannot be read are left out; ``on_skip`` is
    told about each one and why.
    """
    documents: list[Document] = []
    seen: set[str] = set()
    for path in find_files(paths):
        limit = MAX_MAILBOX_BYTES if path.suffix.lower() in MAILBOX_SUFFIXES else MAX_FILE_BYTES
        if path.stat().st_size > limit:
            _skip(on_skip, path, f"larger than {limit // (1024 * 1024)} MB")
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            parts = [t for t in (normalize(p) for p in extract_parts(path, raw)) if t]
        except ExtractError as exc:
            _skip(on_skip, path, str(exc))
            continue
        if parts:
            documents.append(
                Document(path=path, sha256=digest, text="\n\n".join(parts), parts=len(parts))
            )
        else:
            _skip(on_skip, path, "no text")
    return documents


def _skip(on_skip: Callable[[Path, str], None] | None, path: Path, reason: str) -> None:
    if on_skip is not None:
        on_skip(path, reason)
