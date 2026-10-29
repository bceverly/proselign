# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Collect the plain-text and Markdown documents a voice is learned from.

Documents are data, never instructions: nothing here interprets their
content beyond reading it as text.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from proselign.errors import UsageError

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})
MAX_FILE_BYTES = 4 * 1024 * 1024

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
_QUOTE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Document:
    """One ingested file."""

    path: Path
    sha256: str
    text: str


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


def load_documents(paths: list[Path]) -> list[Document]:
    """Read, normalize and de-duplicate (by content hash) every document."""
    documents: list[Document] = []
    seen: set[str] = set()
    for path in find_files(paths):
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        text = normalize(raw.decode("utf-8", errors="replace"))
        if text:
            documents.append(Document(path=path, sha256=digest, text=text))
    return documents
