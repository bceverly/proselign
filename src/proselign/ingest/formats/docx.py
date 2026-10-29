# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Text of Word 2007+ ``.docx`` files (a zip of Office Open XML parts).

Only ``word/document.xml`` is read: its paragraphs, in order.  Deleted
revisions and field instructions are not ``w:t`` text and drop out
naturally.  Office XML never carries a DTD, so any DTD is refused before
parsing, which rules out entity-expansion and external-entity attacks.
"""

from __future__ import annotations

import io
import zipfile

# The stdlib parser is safe here because any DTD is refused before parsing
# (see docx_text), which is what defusedxml would otherwise enforce.
from xml.etree import ElementTree  # nosec B405 # nosemgrep

from proselign.ingest.formats.base import MAX_OUTPUT_BYTES, ExtractError, tidy

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_PART = "word/document.xml"


def docx_text(data: bytes) -> str:
    """Return the body text of the ``.docx`` document in ``data``."""
    xml = _read_part(data)
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ExtractError("document XML contains a DTD, which Office files never do")
    try:
        root = ElementTree.fromstring(xml)  # nosec B314 # nosemgrep - DTDs refused above
    except ElementTree.ParseError as exc:
        raise ExtractError(f"document XML is malformed: {exc}") from exc
    paragraphs: list[str] = []
    try:
        _walk(root, paragraphs, [])
    except RecursionError as exc:
        raise ExtractError("document XML is nested too deeply") from exc
    return tidy(paragraphs)


def _read_part(data: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.getinfo(_PART)
            if info.file_size > MAX_OUTPUT_BYTES:
                raise ExtractError("document XML is too large")
            with archive.open(info) as handle:
                xml = handle.read(MAX_OUTPUT_BYTES + 1)
    except KeyError as exc:
        raise ExtractError(f"not a Word document (no {_PART})") from exc
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError) as exc:
        raise ExtractError(f"corrupt .docx archive: {exc}") from exc
    if len(xml) > MAX_OUTPUT_BYTES:  # the size in the zip header can lie
        raise ExtractError("document XML is too large")
    return xml


def _walk(element: ElementTree.Element, paragraphs: list[str], buffer: list[str]) -> None:
    """Collect paragraph text; nested paragraphs (text boxes) get their own."""
    for child in element:
        tag = child.tag
        if tag == _W + "p":
            inner: list[str] = []
            _walk(child, paragraphs, inner)
            paragraphs.append("".join(inner))
        elif tag == _W + "t":
            buffer.append(child.text or "")
        elif tag == _W + "tab":
            buffer.append(" ")
        elif tag in (_W + "br", _W + "cr"):
            buffer.append("\n")
        else:
            _walk(child, paragraphs, buffer)
