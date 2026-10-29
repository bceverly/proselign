# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Text of ``.pdf`` files, by interpreting page content streams.

Handles the fonts word processors emit: simple fonts in WinAnsi or
MacRoman encoding (with ``/Differences``) and composite fonts with a
``/ToUnicode`` map.  Lines come from text positioning; a vertical gap
noticeably larger than the usual line spacing starts a new paragraph.
Scanned PDFs (images only) have no text to extract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from proselign.ingest.formats.base import ExtractError, tidy
from proselign.ingest.formats.pdfparse import Document, Keyword, Name, Parser, Stream

_GLYPHS = {
    "space": " ", "quoteright": "’", "quoteleft": "‘", "quotedblleft": "“",
    "quotedblright": "”", "quotesingle": "'", "quotedbl": '"', "endash": "–",
    "emdash": "—", "hyphen": "-", "bullet": "•", "ellipsis": "…",
    "fi": "fi", "fl": "fl", "period": ".", "comma": ",", "colon": ":", "semicolon": ";",
    "exclam": "!", "question": "?", "parenleft": "(", "parenright": ")", "slash": "/",
    "ampersand": "&", "percent": "%", "dollar": "$", "apostrophe": "'",
}  # fmt: skip
_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_BFCHAR = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFRANGE = re.compile(rb"beginbfrange(.*?)endbfrange", re.DOTALL)
_INLINE_IMAGE = Keyword("ID")
_KERN_SPACE = -250  # a TJ adjustment this negative (thousandths of an em) is a word gap
_PARAGRAPH_GAP = 1.4  # times the page's typical line spacing
_WORD_GAP = 0.15  # a horizontal jump wider than this (in ems) separates words


def pdf_text(data: bytes) -> str:
    """Return the text of every page of the PDF in ``data``."""
    document = Document(data)
    paragraphs: list[str] = []
    for page, resources in document.pages():
        fonts = _fonts(document, resources)
        lines = _Interpreter(fonts).run(document.contents(page))
        paragraphs.extend(_paragraphs(lines))
    text = tidy(paragraphs)
    if not text:
        raise ExtractError("PDF has no extractable text (scanned images?)")
    return text


class _Font:
    """Maps a font's character codes to Unicode and to glyph widths."""

    def __init__(self, two_byte: bool, table: dict[int, str], widths: dict[int, float]):
        self.two_byte = two_byte
        self.table = table
        self.widths = widths
        self.default_width = 1000.0 if two_byte else 500.0

    def codes(self, raw: bytes) -> list[int]:
        """The character codes in ``raw``."""
        if self.two_byte:
            return [int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw) - 1, 2)]
        return list(raw)

    def decode(self, codes: list[int]) -> str:
        """Unicode text for ``codes``."""
        return "".join(self.table.get(code, "") for code in codes)

    def width(self, code: int) -> float:
        """Advance width of ``code`` in thousandths of the font size."""
        return self.widths.get(code, self.default_width)


def _fonts(document: Document, resources: dict) -> dict[str, _Font]:
    fonts: dict[str, _Font] = {}
    font_dict = document.resolve(resources.get("Font")) if isinstance(resources, dict) else None
    if not isinstance(font_dict, dict):
        return fonts
    for name, ref in font_dict.items():
        font = document.resolve(ref)
        if isinstance(font, dict):
            fonts[name] = _load_font(document, font)
    return fonts


def _load_font(document: Document, font: dict) -> _Font:
    two_byte = font.get("Subtype") == "Type0"
    table = {} if two_byte else _simple_encoding(document, document.resolve(font.get("Encoding")))
    to_unicode = document.resolve(font.get("ToUnicode"))
    if isinstance(to_unicode, Stream):
        try:
            table.update(_parse_cmap(document.decode(to_unicode)))
        except ExtractError:
            pass
    widths = _composite_widths(document, font) if two_byte else _simple_widths(document, font)
    return _Font(two_byte, table, widths)


def _simple_widths(document: Document, font: dict) -> dict[int, float]:
    first = document.resolve(font.get("FirstChar"))
    widths = document.resolve(font.get("Widths"))
    if not isinstance(first, int) or not isinstance(widths, list):
        return {}
    return {
        first + i: float(w)
        for i, w in enumerate(document.resolve(w) for w in widths)
        if isinstance(w, (int, float))
    }


def _composite_widths(document: Document, font: dict) -> dict[int, float]:
    """Widths from a Type0 font's descendant ``/W`` array."""
    descendants = document.resolve(font.get("DescendantFonts"))
    descendant = document.resolve(descendants[0]) if isinstance(descendants, list) else None
    if not isinstance(descendant, dict):
        return {}
    spec = document.resolve(descendant.get("W"))
    widths: dict[int, float] = {}
    items = list(spec) if isinstance(spec, list) else []
    while len(items) >= 2:
        start, nxt = items[0], document.resolve(items[1])
        if isinstance(nxt, list) and isinstance(start, int):
            widths.update(
                (start + i, float(w)) for i, w in enumerate(nxt) if isinstance(w, (int, float))
            )
            items = items[2:]
        elif len(items) >= 3 and isinstance(start, int) and isinstance(nxt, int):
            last, width = items[1], items[2]
            if isinstance(width, (int, float)) and 0 <= last - start <= 0xFFFF:
                widths.update((code, float(width)) for code in range(start, last + 1))
            items = items[3:]
        else:
            break
    return widths


def _simple_encoding(document: Document, encoding) -> dict[int, str]:
    base = "mac_roman" if encoding == "MacRomanEncoding" else "cp1252"
    if isinstance(encoding, dict) and encoding.get("BaseEncoding") == "MacRomanEncoding":
        base = "mac_roman"
    table = {code: bytes([code]).decode(base, errors="replace") for code in range(32, 256)}
    if isinstance(encoding, dict):
        code = 0
        for item in document.resolve(encoding.get("Differences")) or []:
            if isinstance(item, int):
                code = item
            elif isinstance(item, Name):
                table[code] = _glyph(item)
                code += 1
    return table


def _glyph(name: str) -> str:
    if len(name) == 1:
        return name
    if name in _GLYPHS:
        return _GLYPHS[name]
    if name in _DIGITS:
        return str(_DIGITS.index(name))
    if name.startswith("uni") and len(name) == 7:
        try:
            return chr(int(name[3:], 16))
        except ValueError:
            return ""
    return ""


def _parse_cmap(data: bytes) -> dict[int, str]:
    """``bfchar`` and ``bfrange`` mappings from a ToUnicode CMap."""
    table: dict[int, str] = {}
    for block in _BFCHAR.findall(data):
        values = _values(block)
        for src, dst in zip(values[::2], values[1::2]):
            if isinstance(src, bytes) and isinstance(dst, bytes):
                table[int.from_bytes(src, "big")] = _utf16(dst)
    for block in _BFRANGE.findall(data):
        values = _values(block)
        for low, high, dst in zip(values[::3], values[1::3], values[2::3]):
            if isinstance(low, bytes) and isinstance(high, bytes):
                _add_range(table, int.from_bytes(low, "big"), int.from_bytes(high, "big"), dst)
    return table


def _add_range(table: dict[int, str], low: int, high: int, dst) -> None:
    if not 0 <= high - low <= 0xFFFF:
        return
    if isinstance(dst, list):
        for offset, item in enumerate(dst[: high - low + 1]):
            if isinstance(item, bytes):
                table[low + offset] = _utf16(item)
    elif isinstance(dst, bytes) and dst:
        start = int.from_bytes(dst, "big")
        width = len(dst)
        for offset in range(high - low + 1):
            table[low + offset] = _utf16((start + offset).to_bytes(width, "big", signed=False))


def _values(block: bytes) -> list:
    parser = Parser(block, refs=False)
    values = []
    while not parser.at_end():
        values.append(parser.value())
    return values


def _utf16(raw: bytes) -> str:
    return raw.decode("utf-16-be", errors="ignore")


@dataclass
class _TextState:
    """Text-positioning state, in user-space units."""

    h_scale: float = 1.0  # text matrix horizontal scale
    v_scale: float = 1.0  # text matrix vertical scale
    origin_x: float = 0.0  # start of the current text line
    origin_y: float = 0.0
    pen_x: float = 0.0  # where the next glyph goes
    line_y: float = 0.0  # baseline of the line being collected
    size: float = 1.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    leading: float = 0.0


class _Interpreter:
    """Runs the text operators of a content stream, collecting lines.

    The pen position is tracked from glyph widths, so a chunk that starts
    where the previous one ended continues the word, and one that starts
    noticeably further right begins a new word.
    """

    def __init__(self, fonts: dict[str, _Font]):
        self.fonts = fonts
        self.font: _Font | None = None
        self.state = _TextState()
        self.lines: list[tuple[float, list[str]]] = [(0.0, [])]

    def run(self, content: bytes) -> list[tuple[float, str]]:
        """Interpret ``content``; return ``(baseline y, text)`` per line."""
        parser = Parser(content, refs=False)
        operands: list = []
        try:
            while not parser.at_end():
                token = parser.value()
                if not isinstance(token, Keyword):
                    operands.append(token)
                    continue
                if token == _INLINE_IMAGE:  # binary image data follows: skip to EI
                    end = content.find(b"EI", parser.pos)
                    parser.pos = len(content) if end < 0 else end + 2
                else:
                    self._operator(token, operands)
                operands = []
        except (ExtractError, TypeError, ValueError, IndexError):
            pass  # keep the text gathered before the damage
        return [(y, "".join(parts)) for y, parts in self.lines if "".join(parts).strip()]

    def _operator(self, op: str, args: list) -> None:
        state = self.state
        handler = _SETTERS.get(op)
        if handler is not None and args:
            handler(state, args)
        if op == "Tf" and len(args) >= 2:
            self.font = self.fonts.get(args[0])
            state.size = float(args[1]) or 1.0
        elif op == "BT":
            state.h_scale = state.v_scale = 1.0
            state.origin_x = state.origin_y = 0.0
        elif op in ("Td", "TD") and len(args) >= 2:
            self._goto(
                state.origin_x + float(args[0]) * state.h_scale,
                state.origin_y + float(args[1]) * state.v_scale,
            )
        elif op == "Tm" and len(args) >= 6:
            state.h_scale = abs(float(args[0])) or 1.0
            state.v_scale = abs(float(args[3])) or 1.0
            self._goto(float(args[4]), float(args[5]))
        elif op in ("T*", "'", '"'):
            self._goto(state.origin_x, state.origin_y - state.leading * state.v_scale)
        if op in ("Tj", "'", '"') and args:
            self._show(args[-1])
        elif op == "TJ" and args:
            self._show_array(args[-1])

    def _show(self, raw) -> None:
        if not isinstance(raw, bytes) or self.font is None:
            return
        state = self.state
        codes = self.font.codes(raw)
        self.lines[-1][1].append(self.font.decode(codes))
        advance = 0.0
        for code in codes:
            advance += self.font.width(code) / 1000 * state.size + state.char_spacing
            if code == 32 and not self.font.two_byte:
                advance += state.word_spacing
        state.pen_x += advance * state.h_scale

    def _show_array(self, items) -> None:
        state = self.state
        for item in items if isinstance(items, list) else []:
            if isinstance(item, (int, float)):
                state.pen_x -= item / 1000 * state.size * state.h_scale
                if item < _KERN_SPACE:
                    self._space()
            else:
                self._show(item)

    def _goto(self, x: float, y: float) -> None:
        """Move the line origin (and pen) to ``(x, y)``."""
        state = self.state
        if abs(y - state.line_y) > 0.5:
            state.line_y = y
            if self.lines[-1][1]:
                self.lines.append((y, []))
            else:
                self.lines[-1] = (y, [])
        elif x - state.pen_x > _WORD_GAP * state.size * state.h_scale:
            self._space()
        state.origin_x, state.origin_y, state.pen_x = x, y, x

    def _space(self) -> None:
        parts = self.lines[-1][1]
        if parts and not parts[-1].endswith(" "):
            parts.append(" ")


def _set_leading(state: _TextState, args: list) -> None:
    state.leading = float(args[0])


def _set_leading_from_td(state: _TextState, args: list) -> None:
    if len(args) >= 2:
        state.leading = -float(args[1])


def _set_char_spacing(state: _TextState, args: list) -> None:
    state.char_spacing = float(args[0])


def _set_word_spacing(state: _TextState, args: list) -> None:
    state.word_spacing = float(args[0])


def _set_quote_spacing(state: _TextState, args: list) -> None:
    if len(args) >= 3:
        state.word_spacing, state.char_spacing = float(args[0]), float(args[1])


_SETTERS = {
    "TL": _set_leading,
    "TD": _set_leading_from_td,
    "Tc": _set_char_spacing,
    "Tw": _set_word_spacing,
    '"': _set_quote_spacing,
}


def _paragraphs(lines: list[tuple[float, str]]) -> list[str]:
    """Group lines into paragraphs by their vertical spacing."""
    if not lines:
        return []
    gaps = [prev[0] - line[0] for prev, line in zip(lines, lines[1:])]
    typical = median([g for g in gaps if g > 0] or [1.0])
    paragraphs = [[lines[0][1]]]
    for gap, (_, text) in zip(gaps, lines[1:]):
        if gap <= 0 or gap > typical * _PARAGRAPH_GAP:
            paragraphs.append([text])
        else:
            paragraphs[-1].append(text)
    return [_join(p) for p in paragraphs]


def _join(lines: list[str]) -> str:
    """Join wrapped lines, mending words hyphenated across a line break."""
    text = lines[0].strip()
    for line in lines[1:]:
        line = line.strip()
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}"
    return text
