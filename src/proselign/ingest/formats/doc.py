# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Text of legacy Word ``.doc`` files: Word 97-2003 ([MS-DOC]) and Word 6/95.

Word 97+ locates the main document's characters through the piece table
(the ``Clx`` in the table stream), decoding each piece as cp1252 or
UTF-16.  Word 6/95 stores 8-bit text from ``fcMin``, or behind a piece
table in the document stream itself when the file was fast-saved.
Headers, footnotes and comments follow the main text and are left out;
field codes are dropped and only their displayed results kept.
"""

from __future__ import annotations

import re
import struct

from proselign.ingest.formats.base import ExtractError, tidy
from proselign.ingest.formats.ole import CompoundFile

_WORD_IDENT = 0xA5EC
_WORD95_IDENT = 0xA5DC
_WORD97_NFIB = 0x00C1
_COMPLEX = 0x0004
_ENCRYPTED = 0x0100
_TABLE_1 = 0x0200
_FC_CLX_INDEX = 33
_COMPRESSED = 0x40000000

_FIELD_BEGIN, _FIELD_SEPARATOR, _FIELD_END = "\x13", "\x14", "\x15"
# Paragraph, cell/row end, line break, page/section break.
_BREAKS = re.compile("[\r\x07\x0b\x0c]")
_CONTROL = re.compile("[\x00-\x06\x08\x0e-\x1d]")


def doc_text(data: bytes) -> str:
    """Return the body text of the Word document in ``data``."""
    container = CompoundFile(data)
    word = container.stream("WordDocument")
    if len(word) >= 2 and struct.unpack_from("<H", word, 0)[0] == _WORD95_IDENT:
        return _clean(_word95_text(word))
    ccp_text, fc_clx, lcb_clx, table_name = _read_fib(word)
    table = container.stream(table_name)
    if fc_clx + lcb_clx > len(table) or lcb_clx == 0:
        raise ExtractError("Word piece table is out of range")
    text = "".join(_pieces(word, table[fc_clx : fc_clx + lcb_clx]))[:ccp_text]
    return _clean(text)


def _word95_text(word: bytes) -> str:
    """Main text of a Word 6/95 document (always 8-bit)."""
    try:
        flags = struct.unpack_from("<H", word, 0x0A)[0]
        fc_min = struct.unpack_from("<I", word, 0x18)[0]
        ccp_text = max(struct.unpack_from("<i", word, 0x34)[0], 0)
        fc_clx, lcb_clx = struct.unpack_from("<II", word, 0x160)
    except struct.error as exc:
        raise ExtractError("Word 95 file information block is truncated") from exc
    if flags & _ENCRYPTED:
        raise ExtractError("document is password-protected")
    if flags & _COMPLEX and lcb_clx:
        if fc_clx + lcb_clx > len(word):
            raise ExtractError("Word piece table is out of range")
        clx = word[fc_clx : fc_clx + lcb_clx]
        return "".join(_pieces(word, clx, eight_bit=True))[:ccp_text]
    raw = word[fc_min : fc_min + ccp_text]
    if len(raw) != ccp_text:
        raise ExtractError("Word 95 text points outside the document")
    return raw.decode("cp1252", errors="replace")


def _read_fib(word: bytes) -> tuple[int, int, int, str]:
    """``(ccpText, fcClx, lcbClx, table stream name)`` from the FIB."""
    if len(word) < 34:
        raise ExtractError("Word document stream is truncated")
    ident, nfib = struct.unpack_from("<HH", word, 0)
    flags = struct.unpack_from("<H", word, 0x0A)[0]
    if ident != _WORD_IDENT:
        raise ExtractError("not a Word document")
    if nfib < _WORD97_NFIB:
        raise ExtractError("Word 95 and older documents are not supported")
    if flags & _ENCRYPTED:
        raise ExtractError("document is password-protected")
    try:
        pos = 32
        csw = struct.unpack_from("<H", word, pos)[0]
        pos += 2 + csw * 2
        cslw = struct.unpack_from("<H", word, pos)[0]
        rglw = pos + 2
        pos = rglw + cslw * 4
        cb_rg = struct.unpack_from("<H", word, pos)[0]
        if cslw < 4 or cb_rg <= _FC_CLX_INDEX:
            raise ExtractError("Word file information block is too short")
        ccp_text = struct.unpack_from("<i", word, rglw + 12)[0]
        fc_clx, lcb_clx = struct.unpack_from("<II", word, pos + 2 + _FC_CLX_INDEX * 8)
    except struct.error as exc:
        raise ExtractError("Word file information block is truncated") from exc
    return max(ccp_text, 0), fc_clx, lcb_clx, "1Table" if flags & _TABLE_1 else "0Table"


def _piece_table(clx: bytes) -> bytes:
    """The ``PlcPcd`` inside a ``Clx``, skipping any leading ``Prc`` blocks."""
    pos = 0
    while pos < len(clx):
        kind = clx[pos]
        if kind == 1 and pos + 3 <= len(clx):
            pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
        elif kind == 2 and pos + 5 <= len(clx):
            size = struct.unpack_from("<I", clx, pos + 1)[0]
            return clx[pos + 5 : pos + 5 + size]
        else:
            break
    raise ExtractError("Word piece table is corrupt")


def _pieces(word: bytes, clx: bytes, eight_bit: bool = False):
    """Yield the text of each piece in character-position order.

    Word 97+ marks 8-bit pieces with a flag in the file offset; in Word 6/95
    (``eight_bit``) every piece is 8-bit and the offset is used as is.
    """
    plc = _piece_table(clx)
    count = (len(plc) - 4) // 12
    if count <= 0:
        raise ExtractError("Word piece table is empty")
    cps = struct.unpack_from(f"<{count + 1}i", plc, 0)
    for index in range(count):
        chars = cps[index + 1] - cps[index]
        if chars <= 0:
            continue
        fc = struct.unpack_from("<I", plc, 4 * (count + 1) + 8 * index + 2)[0]
        if eight_bit:
            raw, encoding = word[fc : fc + chars], "cp1252"
        elif fc & _COMPRESSED:
            start = (fc & ~_COMPRESSED) // 2
            raw, encoding = word[start : start + chars], "cp1252"
        else:
            raw, encoding = word[fc : fc + 2 * chars], "utf-16-le"
        if len(raw) != (chars if encoding == "cp1252" else 2 * chars):
            raise ExtractError("Word piece points outside the document")
        yield raw.decode(encoding, errors="replace")


def _strip_fields(text: str) -> str:
    """Drop field instructions, keeping what the field displays."""
    out = []
    stack: list[bool] = []  # per open field: True once past its separator
    for char in text:
        if char == _FIELD_BEGIN:
            stack.append(False)
        elif char == _FIELD_SEPARATOR and stack:
            stack[-1] = True
        elif char == _FIELD_END and stack:
            stack.pop()
        elif all(stack):
            out.append(char)
    return "".join(out)


def _clean(text: str) -> str:
    text = _strip_fields(text)
    text = text.replace("\x1e", "-").replace("\x1f", "").replace("\xa0", " ")
    text = _CONTROL.sub("", text)
    return tidy(_BREAKS.split(text))
