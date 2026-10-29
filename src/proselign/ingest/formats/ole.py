# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Minimal reader for OLE2 compound files, the container of legacy ``.doc``.

Implements just enough of [MS-CFB] to read a named stream from the root
storage: header, DIFAT, FAT, mini FAT and the directory tree.  Sector
chains are followed with cycle and bounds checks.
"""

from __future__ import annotations

import struct

from proselign.ingest.formats.base import ExtractError

MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
_NOSTREAM = 0xFFFFFFFF
_SPECIAL = 0xFFFFFFFA  # sector ids at or above this are markers, not sectors
_HEADER_DIFAT = 109
_DIR_ENTRY = 128
_STREAM = 2


class CompoundFile:
    """A parsed compound file; :meth:`stream` returns root-level streams."""

    def __init__(self, data: bytes):
        if len(data) < 512 or data[:8] != MAGIC:
            raise ExtractError("not an OLE2 compound file")
        self._data = data
        sector_shift, mini_shift = struct.unpack_from("<HH", data, 0x1E)
        if sector_shift not in (9, 12) or mini_shift != 6:
            raise ExtractError("unsupported compound file sector size")
        self._size = 1 << sector_shift
        self._max_sectors = len(data) // self._size + 1
        num_fat, first_dir, _sig, self._cutoff, first_mini, _num_mini, first_difat, _num_difat = (
            struct.unpack_from("<8I", data, 0x2C)
        )
        self._fat = self._read_fat(num_fat, first_difat)
        self._entries = self._read_directory(first_dir)
        root = self._entries[0]
        self._ministream = self._read_chain(root[1], root[2])
        self._minifat = _ints(self._read_chain(first_mini, None))

    def _sector(self, sid: int) -> bytes:
        start = (sid + 1) * self._size
        if sid >= _SPECIAL or start + self._size > len(self._data):
            raise ExtractError("compound file sector out of range")
        return self._data[start : start + self._size]

    def _read_fat(self, num_fat: int, first_difat: int) -> list[int]:
        sids = list(struct.unpack_from(f"<{_HEADER_DIFAT}I", self._data, 0x4C))
        per_sector = self._size // 4 - 1
        seen: set[int] = set()
        sid = first_difat
        while sid < _SPECIAL:
            if sid in seen or len(seen) > self._max_sectors:
                raise ExtractError("compound file DIFAT chain loops")
            seen.add(sid)
            values = _ints(self._sector(sid))
            sids.extend(values[:per_sector])
            sid = values[per_sector]
        fat_sids = [s for s in sids if s < _SPECIAL][:num_fat]
        return _ints(b"".join(self._sector(s) for s in fat_sids))

    def _chain(self, start: int, fat: list[int]) -> list[int]:
        chain: list[int] = []
        seen: set[int] = set()
        sid = start
        while sid < _SPECIAL:
            if sid in seen or sid >= len(fat):
                raise ExtractError("compound file sector chain is corrupt")
            seen.add(sid)
            chain.append(sid)
            sid = fat[sid]
        return chain

    def _read_chain(self, start: int, size: int | None) -> bytes:
        data = b"".join(self._sector(s) for s in self._chain(start, self._fat))
        return data if size is None else data[:size]

    def _read_directory(self, first_dir: int) -> list[tuple]:
        raw = self._read_chain(first_dir, None)
        entries = []
        for offset in range(0, len(raw) - _DIR_ENTRY + 1, _DIR_ENTRY):
            name_len = struct.unpack_from("<H", raw, offset + 0x40)[0]
            name = raw[offset : offset + max(0, min(name_len, 64) - 2)].decode(
                "utf-16-le", errors="replace"
            )
            kind = raw[offset + 0x42]
            left, right, child = struct.unpack_from("<3I", raw, offset + 0x44)
            start, size = struct.unpack_from("<II", raw, offset + 0x74)
            entries.append((name, start, size, kind, left, right, child))
        if not entries:
            raise ExtractError("compound file has no directory")
        return entries

    def _root_children(self) -> list[tuple]:
        """Entries directly under the root storage (its red-black tree)."""
        found = []
        pending = [self._entries[0][6]]
        seen: set[int] = set()
        while pending:
            index = pending.pop()
            if index == _NOSTREAM or index in seen or index >= len(self._entries):
                continue
            seen.add(index)
            entry = self._entries[index]
            found.append(entry)
            pending.extend((entry[4], entry[5]))
        return found

    def stream(self, name: str) -> bytes:
        """Contents of the root-level stream ``name``."""
        for entry in self._root_children():
            if entry[3] == _STREAM and entry[0] == name:
                _, start, size = entry[:3]
                if size < self._cutoff:
                    return self._read_mini(start, size)
                return self._read_chain(start, size)
        raise ExtractError(f"compound file has no {name!r} stream")

    def _read_mini(self, start: int, size: int) -> bytes:
        parts = []
        for sid in self._chain(start, self._minifat):
            offset = sid * 64
            if offset + 64 > len(self._ministream):
                raise ExtractError("compound file mini sector out of range")
            parts.append(self._ministream[offset : offset + 64])
        return b"".join(parts)[:size]


def _ints(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data[: len(data) // 4 * 4]))
