# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Minimal PDF object reader: just enough of ISO 32000 to reach page text.

Objects are found by scanning for ``N G obj`` rather than trusting the
cross-reference table, which also copes with damaged and incrementally
updated files.  Object streams are expanded; stream filters are limited
to Flate, ASCIIHex and ASCII85.  Nesting depth, reference chains, page
trees and decompressed sizes are all capped.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

from proselign.ingest.formats.base import ExtractError, inflate

MAX_DEPTH = 64
MAX_PAGES = 5000

_WHITESPACE = b" \t\r\n\f\x00"
_DELIMITERS = b"()<>[]{}/%"
_NUMBER = re.compile(rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_REF_TAIL = re.compile(rb"\s+(\d+)\s+R(?![^\s()<>\[\]{}/%])")
# Bounded counts keep the scan linear on long runs of digits and spaces;
# object numbers never exceed 10 digits, generations 5.
_OBJECT = re.compile(rb"(?<![0-9])([0-9]{1,10})\s{1,8}([0-9]{1,5})\s{1,8}obj(?![^\s()<>\[\]{}/%])")
_ESCAPES = {ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t", ord("b"): b"\b", ord("f"): b"\f"}


class Name(str):
    """A PDF name (``/Font``), stored without the slash."""


class Keyword(str):
    """A bare PDF word: an operator in content streams."""


@dataclass(frozen=True)
class Ref:
    """An indirect reference ``N G R``."""

    num: int


@dataclass
class Stream:
    """A stream object: its dictionary and still-encoded bytes."""

    attrs: dict
    raw: bytes


class Parser:
    """Reads PDF values from ``data``; ``refs`` enables ``N G R`` references."""

    def __init__(self, data: bytes, refs: bool = True):
        self.data = data
        self.pos = 0
        self.refs = refs

    def skip(self) -> None:
        """Skip whitespace and comments."""
        data = self.data
        while self.pos < len(data):
            char = data[self.pos]
            if char in _WHITESPACE:
                self.pos += 1
            elif char == 0x25:  # %
                end = data.find(b"\n", self.pos)
                self.pos = len(data) if end < 0 else end + 1
            else:
                return

    def at_end(self) -> bool:
        """True once only whitespace and comments remain."""
        self.skip()
        return self.pos >= len(self.data)

    def value(self, depth: int = 0):
        """The next value; raises :class:`ExtractError` on malformed input."""
        if depth > MAX_DEPTH:
            raise ExtractError("PDF objects are nested too deeply")
        self.skip()
        if self.pos >= len(self.data):
            raise ExtractError("unexpected end of PDF data")
        lead = self.data[self.pos : self.pos + 2]
        if lead == b"<<":
            return self._dictionary(depth)
        handlers = {0x5B: self._array, 0x28: self._literal, 0x3C: self._hex, 0x2F: self._name}
        handler = handlers.get(lead[0])
        if handler is not None:
            return handler(depth)
        number = _NUMBER.match(self.data, self.pos)
        if number:
            return self._number(number)
        return self._keyword()

    def _dictionary(self, depth: int) -> dict:
        self.pos += 2
        result: dict = {}
        while True:
            self.skip()
            if self.data.startswith(b">>", self.pos):
                self.pos += 2
                return result
            key = self.value(depth + 1)
            if not isinstance(key, Name):
                raise ExtractError("PDF dictionary key is not a name")
            result[key] = self.value(depth + 1)

    def _array(self, depth: int) -> list:
        self.pos += 1
        result = []
        while True:
            self.skip()
            if self.data.startswith(b"]", self.pos):
                self.pos += 1
                return result
            result.append(self.value(depth + 1))

    def _literal(self, _depth: int) -> bytes:
        data, pos, out, level = self.data, self.pos + 1, bytearray(), 1
        while pos < len(data):
            char = data[pos]
            pos += 1
            if char == 0x5C and pos < len(data):  # backslash
                pos = self._escape(pos, out)
                continue
            level += (char == 0x28) - (char == 0x29)
            if level == 0:
                self.pos = pos
                return bytes(out)
            out.append(char)
        raise ExtractError("unterminated PDF string")

    def _escape(self, pos: int, out: bytearray) -> int:
        data = self.data
        char = data[pos]
        if char in _ESCAPES:
            out += _ESCAPES[char]
            return pos + 1
        if 0x30 <= char <= 0x37:
            end = pos
            while end < min(pos + 3, len(data)) and 0x30 <= data[end] <= 0x37:
                end += 1
            out.append(int(data[pos:end], 8) & 0xFF)
            return end
        if char == 0x0D:  # line continuation
            return pos + 2 if data[pos + 1 : pos + 2] == b"\n" else pos + 1
        if char != 0x0A:
            out.append(char)
        return pos + 1

    def _hex(self, _depth: int) -> bytes:
        end = self.data.find(b">", self.pos)
        if end < 0:
            raise ExtractError("unterminated PDF hex string")
        start = self.pos + 1
        self.pos = end + 1
        return unhex(self.data[start:end])

    def _name(self, _depth: int) -> Name:
        start = self.pos + 1
        end = start
        while end < len(self.data) and self.data[end] not in _WHITESPACE + _DELIMITERS:
            end += 1
        self.pos = end
        raw = re.sub(
            rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), self.data[start:end]
        )
        return Name(raw.decode("latin-1"))

    def _number(self, match: re.Match):
        self.pos = match.end()
        text = match.group(0)
        if b"." in text:
            return float(text)
        number = int(text)
        if self.refs:
            tail = _REF_TAIL.match(self.data, self.pos)
            if tail and number >= 0:
                self.pos = tail.end()
                return Ref(number)
        return number

    def _keyword(self) -> object:
        start = self.pos
        end = start
        while end < len(self.data) and self.data[end] not in _WHITESPACE + _DELIMITERS:
            end += 1
        if end == start:  # a stray delimiter such as ')' or '{'
            end += 1
        self.pos = end
        word = self.data[start:end].decode("latin-1")
        return {"true": True, "false": False, "null": None}.get(word, Keyword(word))


class Document:
    """All objects of a PDF file, with page-tree and stream helpers."""

    def __init__(self, data: bytes):
        if b"%PDF-" not in data[:1024]:
            raise ExtractError("not a PDF file")
        self.objects: dict[int, object] = {}
        self._scan(data)
        self._expand_object_streams()
        if any(isinstance(o, dict) and "Encrypt" in o for o in self.objects.values()):
            raise ExtractError("PDF is encrypted")

    def _scan(self, data: bytes) -> None:
        pos = 0
        while True:
            match = _OBJECT.search(data, pos)
            if match is None:
                return
            parser = Parser(data)
            parser.pos = match.end()
            try:
                value = parser.value()
                parser.skip()
                if isinstance(value, dict) and data.startswith(b"stream", parser.pos):
                    value, parser.pos = _read_stream(data, parser.pos + 6, value)
            except ExtractError:
                pos = match.end()
                continue
            self.objects[int(match.group(1))] = value
            pos = parser.pos

    def _expand_object_streams(self) -> None:
        for stream in [o for o in self.objects.values() if isinstance(o, Stream)]:
            if stream.attrs.get("Type") != "ObjStm":
                continue
            try:
                content = self.decode(stream)
                first = int(self.resolve(stream.attrs.get("First", 0)))
                header = Parser(content[:first], refs=False)
                numbers = []
                while not header.at_end():
                    numbers.append(header.value())
                for num, offset in zip(numbers[::2], numbers[1::2]):
                    parser = Parser(content)
                    parser.pos = first + int(offset)
                    self.objects.setdefault(int(num), parser.value())
            except (ExtractError, TypeError, ValueError):
                continue

    def resolve(self, value):
        """Follow indirect references (bounded) to a direct value."""
        for _ in range(MAX_DEPTH):
            if not isinstance(value, Ref):
                return value
            value = self.objects.get(value.num)
        raise ExtractError("PDF reference chain is too long")

    def decode(self, stream: Stream) -> bytes:
        """Apply the stream's filters; unsupported filters raise."""
        filters = self.resolve(stream.attrs.get("Filter"))
        if filters is None:
            filters = []
        elif not isinstance(filters, list):
            filters = [filters]
        data = stream.raw
        for name in (self.resolve(f) for f in filters):
            if name in ("FlateDecode", "Fl"):
                data = inflate(data)
            elif name in ("ASCIIHexDecode", "AHx"):
                data = unhex(data.split(b">")[0])
            elif name in ("ASCII85Decode", "A85"):
                data = _ascii85(data)
            else:
                raise ExtractError(f"unsupported PDF stream filter {name}")
        return data

    def pages(self) -> list[tuple[dict, dict]]:
        """``(page, inherited resources)`` for every page, in order."""
        catalog = next(
            (
                o
                for o in self.objects.values()
                if isinstance(o, dict) and o.get("Type") == "Catalog"
            ),
            None,
        )
        if catalog is None:
            raise ExtractError("PDF has no document catalog")
        found: list[tuple[dict, dict]] = []
        pending = [(self.resolve(catalog.get("Pages")), {})]
        seen: set[int] = set()
        while pending and len(found) < MAX_PAGES:
            node, resources = pending.pop()
            if not isinstance(node, dict) or id(node) in seen:
                continue
            seen.add(id(node))
            resources = self.resolve(node.get("Resources")) or resources
            if node.get("Type") == "Page" or "Kids" not in node:
                found.append((node, resources))
            else:
                kids = self.resolve(node.get("Kids")) or []
                pending.extend((self.resolve(k), resources) for k in reversed(kids))
        return found

    def contents(self, page: dict) -> bytes:
        """The page's decoded content streams, concatenated."""
        contents = self.resolve(page.get("Contents"))
        parts = contents if isinstance(contents, list) else [contents]
        decoded = []
        for part in (self.resolve(p) for p in parts):
            if isinstance(part, Stream):
                try:
                    decoded.append(self.decode(part))
                except ExtractError:
                    continue
        return b"\n".join(decoded)


def _read_stream(data: bytes, pos: int, attrs: dict) -> tuple[Stream, int]:
    """Read the stream body that starts after the ``stream`` keyword at ``pos``."""
    if data.startswith(b"\r\n", pos):
        pos += 2
    elif data[pos : pos + 1] in (b"\n", b"\r"):
        pos += 1
    length = attrs.get("Length")
    if isinstance(length, int) and 0 <= length <= len(data) - pos:
        tail = data[pos + length : pos + length + 32].lstrip(_WHITESPACE)
        if tail.startswith(b"endstream"):
            return Stream(attrs, data[pos : pos + length]), pos + length
    end = data.find(b"endstream", pos)
    if end < 0:
        raise ExtractError("unterminated PDF stream")
    body = data[pos:end]
    if body.endswith(b"\r\n"):
        body = body[:-2]
    elif body.endswith((b"\n", b"\r")):
        body = body[:-1]
    return Stream(attrs, body), end + len(b"endstream")


def unhex(text: bytes) -> bytes:
    """Decode PDF hex digits; whitespace is ignored and an odd digit padded."""
    digits = bytes(c for c in text if c not in _WHITESPACE)
    try:
        return binascii.unhexlify(digits + b"0" * (len(digits) % 2))
    except (binascii.Error, ValueError) as exc:
        raise ExtractError("malformed PDF hex string") from exc


def _ascii85(data: bytes) -> bytes:
    body = data.split(b"~>")[0].strip()
    if body.startswith(b"<~"):
        body = body[2:]
    try:
        return base64.a85decode(body, ignorechars=_WHITESPACE)
    except ValueError as exc:
        raise ExtractError("malformed ASCII85 stream") from exc
