# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Lookup, listing and deletion of stored voices."""

from __future__ import annotations

import shutil
from pathlib import Path

from proselign.errors import VoiceError
from proselign.paths import voices_dir
from proselign.voice.profile import MANIFEST_NAME, Voice, voice_id_for


class Registry:
    """The set of voices under one data directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else voices_dir()

    def path_for(self, name: str) -> Path:
        """Directory a voice named ``name`` lives (or would live) in."""
        return self.root / voice_id_for(name)

    def exists(self, name: str) -> bool:
        """True when a voice with this name has been stored."""
        return (self.path_for(name) / MANIFEST_NAME).is_file()

    def get(self, name: str) -> Voice:
        """Load a voice, with a helpful message when it is missing."""
        if not self.exists(name):
            raise VoiceError(
                f"unknown voice {name!r}; create it with: proselign train --voice {name} DIR"
            )
        return Voice.load(self.path_for(name))

    def save(self, voice: Voice) -> Path:
        """Persist a voice, returning its manifest path."""
        return voice.save(self.root / voice.voice_id)

    def list(self) -> list[Voice]:
        """All readable voices, sorted by identifier. Corrupt ones are skipped."""
        if not self.root.is_dir():
            return []
        voices = []
        for child in sorted(self.root.iterdir()):
            if (child / MANIFEST_NAME).is_file():
                try:
                    voices.append(Voice.load(child))
                except VoiceError:
                    continue
        return voices

    def delete(self, name: str) -> None:
        """Remove a voice and everything stored with it."""
        if not self.exists(name):
            raise VoiceError(f"unknown voice {name!r}")
        path = self.path_for(name)
        if path.is_symlink():
            raise VoiceError(f"refusing to delete {path}: it is a symbolic link")
        shutil.rmtree(path)
