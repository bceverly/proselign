# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Build small Word, OLE2 and PDF files in memory for the extractor tests.

Each builder writes just the structures the extractors read, following
the published layouts ([MS-CFB], [MS-DOC], ISO 32000), so tests need no
binary fixtures checked into the repository.
"""

# File-format writers lay out many fields at once.
# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import io
import struct
import zipfile
import zlib

SECTOR = 512
FREE, END, FATSECT, NOSTREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF


# --- OLE2 compound files ----------------------------------------------------


def make_cfb(streams: dict[str, bytes], cutoff: int = 4096) -> bytes:
    """A version-3 compound file whose root holds ``streams``.

    Streams shorter than ``cutoff`` go into the mini stream, as Word does.
    """
    sectors: list[bytes] = []
    fat: list[int] = []

    def chain(data: bytes) -> int:
        if not data:
            return END
        start = len(sectors)
        chunks = [data[i : i + SECTOR] for i in range(0, len(data), SECTOR)]
        for index, chunk in enumerate(chunks):
            sectors.append(chunk.ljust(SECTOR, b"\0"))
            fat.append(start + index + 1 if index + 1 < len(chunks) else END)
        return start

    ministream, minifat, placed = bytearray(), [], []
    for name, data in streams.items():
        if len(data) < cutoff:
            first = len(ministream) // 64
            count = max(1, -(-len(data) // 64))
            minifat.extend(first + i + 1 if i + 1 < count else END for i in range(count))
            ministream += data.ljust(count * 64, b"\0")
            placed.append((name, first, len(data)))
        else:
            placed.append((name, chain(data), len(data)))
    mini_start = chain(bytes(ministream))
    minifat_start = chain(struct.pack(f"<{len(minifat)}I", *minifat)) if minifat else END

    entries = [
        _dir_entry("Root Entry", 5, mini_start, len(ministream), child=1 if placed else NOSTREAM)
    ]
    for index, (name, start, size) in enumerate(placed):
        right = index + 2 if index + 1 < len(placed) else NOSTREAM
        entries.append(_dir_entry(name, 2, start, size, right=right))
    dir_start = chain(b"".join(entries))

    fat_count = -(-(len(sectors) + 1) // (SECTOR // 4 - 1)) or 1
    fat_sids = list(range(len(sectors), len(sectors) + fat_count))
    fat.extend([FATSECT] * fat_count)
    fat_bytes = struct.pack(f"<{len(fat)}I", *fat).ljust(fat_count * SECTOR, b"\xff")
    sectors.extend(fat_bytes[i : i + SECTOR] for i in range(0, len(fat_bytes), SECTOR))

    header = bytearray(SECTOR)
    header[0:8] = bytes.fromhex("D0CF11E0A1B11AE1")
    struct.pack_into("<HHHHH", header, 0x18, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into(
        "<8I", header, 0x2C, fat_count, dir_start, 0, cutoff, minifat_start,
        1 if minifat else 0, END, 0,
    )  # fmt: skip
    difat = fat_sids + [FREE] * (109 - len(fat_sids))
    struct.pack_into("<109I", header, 0x4C, *difat)
    return bytes(header) + b"".join(sectors)


def _dir_entry(
    name: str, kind: int, start: int, size: int, right=NOSTREAM, child=NOSTREAM
) -> bytes:
    entry = bytearray(128)
    encoded = (name + "\0").encode("utf-16-le")
    entry[: len(encoded)] = encoded
    struct.pack_into("<HBB3I", entry, 0x40, len(encoded), kind, 1, NOSTREAM, right, child)
    struct.pack_into("<IQ", entry, 0x74, start, size)
    return bytes(entry)


# --- Word 97-2003 -------------------------------------------------------------

_TEXT_AT = 1024


def make_word97(pieces: list[tuple[str, bool]], *, flags: int = 0x0200, nfib: int = 0x00C1) -> dict:
    """Streams of a Word 97 document whose main text is ``pieces``.

    Each piece is ``(text, compressed)``: cp1252 when compressed, else UTF-16.
    """
    word = bytearray(_TEXT_AT)
    struct.pack_into("<HH", word, 0, 0xA5EC, nfib)
    struct.pack_into("<H", word, 0x0A, flags)
    pos = 32
    struct.pack_into("<H", word, pos, 14)
    pos += 2 + 28
    struct.pack_into("<H", word, pos, 22)
    rglw = pos + 2
    pos = rglw + 88
    struct.pack_into("<H", word, pos, 93)
    rgfclcb = pos + 2

    cps, pcds, cp = [0], [], 0
    for text, compressed in pieces:
        offset = len(word)
        if compressed:
            word += text.encode("cp1252")
            fc = (offset * 2) | 0x40000000
        else:
            word += text.encode("utf-16-le")
            fc = offset
        cp += len(text)
        cps.append(cp)
        pcds.append(struct.pack("<HIH", 0, fc, 0))
    plc = struct.pack(f"<{len(cps)}i", *cps) + b"".join(pcds)
    clx = b"\x01" + struct.pack("<H", 2) + b"\0\0" + b"\x02" + struct.pack("<I", len(plc)) + plc
    struct.pack_into("<i", word, rglw + 12, cp)
    struct.pack_into("<II", word, rgfclcb + 33 * 8, 0, len(clx))
    table = "1Table" if flags & 0x0200 else "0Table"
    return {"WordDocument": bytes(word), table: clx}


def make_word95(text: str, *, fast_saved: bool = False, flags: int = 0) -> dict:
    """Streams of a Word 6/95 document (8-bit text, optionally fast-saved)."""
    word = bytearray(768)
    struct.pack_into("<HH", word, 0, 0xA5DC, 0x0068)
    encoded = text.encode("cp1252")
    if fast_saved:
        half = len(encoded) // 2
        second = len(word)
        word += encoded[half:]  # stored out of order, as fast saves do
        first = len(word)
        word += encoded[:half]
        plc = struct.pack("<3i", 0, half, len(encoded))
        plc += struct.pack("<HIH", 0, first, 0) + struct.pack("<HIH", 0, second, 0)
        clx = b"\x02" + struct.pack("<I", len(plc)) + plc
        struct.pack_into("<II", word, 0x160, len(word), len(clx))
        word += clx
        flags |= 0x0004
    else:
        struct.pack_into("<I", word, 0x18, len(word))
        word += encoded
    struct.pack_into("<H", word, 0x0A, flags)
    struct.pack_into("<i", word, 0x34, len(encoded))
    return {"WordDocument": bytes(word)}


# --- docx -----------------------------------------------------------------------

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(body_xml: str, *, prolog: str = "", part: str = "word/document.xml") -> bytes:
    """A .docx whose ``w:body`` contains ``body_xml``."""
    xml = f'{prolog}<w:document xmlns:w="{W_NS}"><w:body>{body_xml}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(part, xml)
    return buffer.getvalue()


def para(*runs: str) -> str:
    """A ``w:p`` with one ``w:t`` run per argument."""
    return "<w:p>" + "".join(f"<w:r><w:t>{r}</w:t></w:r>" for r in runs) + "</w:p>"


# --- PDF ------------------------------------------------------------------------

SIMPLE_FONT = (
    b"<< /Type /Font /Subtype /TrueType /BaseFont /Arial /Encoding /WinAnsiEncoding"
    b" /FirstChar 32 /LastChar 126 /Widths [" + b" 500" * 95 + b"] >>"
)


def make_pdf(pages: list[bytes], fonts: dict[str, bytes] | None = None, *, compress: bool = True,
             extra: list[bytes] = (), trailer: bytes = b"") -> bytes:  # fmt: skip
    """A PDF with one page per content stream in ``pages``.

    ``fonts`` maps resource names to font dictionaries (default: ``/F1``, a
    WinAnsi TrueType font 500 units wide).  ``extra`` objects are appended
    after the fonts, numbered on from there.
    """
    fonts = fonts or {"F1": SIMPLE_FONT}
    objects: list[bytes] = []
    font_refs = []
    for name, font in fonts.items():
        objects.append(font)
        font_refs.append(f"/{name} {len(objects)} 0 R".encode())
    objects.extend(extra)
    resources = b"<< /Font << " + b" ".join(font_refs) + b" >> >>"
    page_nums = []
    pages_num = len(objects) + 1 + 2 * len(pages)
    for content in pages:
        if compress:
            body = zlib.compress(content)
            objects.append(
                b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(body)
                + body
                + b"\nendstream"
            )
        else:
            objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        objects.append(
            b"<< /Type /Page /Parent %d 0 R /Contents %d 0 R >>" % (pages_num, len(objects))
        )
        page_nums.append(len(objects))
    kids = b" ".join(b"%d 0 R" % n for n in page_nums)
    objects.append(
        b"<< /Type /Pages /Kids ["
        + kids
        + b"] /Count %d /Resources " % len(pages)
        + resources
        + b" >>"
    )
    objects.append(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num)
    out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    for number, body in enumerate(objects, 1):
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    out += b"trailer\n<< /Root %d 0 R " % len(objects) + trailer + b">>\n%%EOF\n"
    return bytes(out)


def text_page(*lines: bytes, size: int = 12, leading: int = 14) -> bytes:
    """A content stream showing each line with ``Tj``, one per baseline."""
    ops = [b"BT /F1 %d Tf %d TL 72 720 Td" % (size, leading)]
    for line in lines:
        ops.append(b"(" + line + b") Tj T*")
    ops.append(b"ET")
    return b"\n".join(ops)
