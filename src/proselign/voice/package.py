# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Portable ``.proselign`` voice archives.

An archive is a gzip'd tar holding ``voice.json`` and nothing executable.
Import is defensive: archives come from other people, so members are
whitelisted by name, sizes are capped, and nothing is ever unpickled or run.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

from proselign.errors import VoiceError
from proselign.voice.profile import MANIFEST_NAME, Voice
from proselign.voice.registry import Registry

ARCHIVE_SUFFIX = ".proselign"
MAX_MEMBER_BYTES = 8 * 1024 * 1024
ALLOWED_MEMBERS = frozenset({MANIFEST_NAME})


def export_voice(
    registry: Registry, name: str, output: Path, include_sources: bool = False
) -> Path:
    """Write ``name`` to ``output`` as a ``.proselign`` archive.

    Source file paths and hashes are stripped unless ``include_sources``,
    because even metadata about a private corpus can be sensitive.
    """
    voice = registry.get(name)
    data = voice.to_dict()
    if not include_sources:
        data["sources"] = []
    if not voice.license_and_consent.get("raw_examples_exportable"):
        data["examples"] = []  # the author's own words stay on this machine
    payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        # Owner-only, like the voice store: a profile describes how one person
        # writes.  Sharing it is a deliberate act (chmod, or send the file).
        fd = os.open(output, flags, 0o600)
    except OSError as exc:
        raise VoiceError(f"cannot write {output}: {exc.strerror}") from exc
    with os.fdopen(fd, "wb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(payload))
    return output


def import_voice(registry: Registry, archive: Path, force: bool = False) -> Voice:
    """Install the voice in ``archive``; refuse to overwrite unless ``force``."""
    try:
        tar = tarfile.open(archive, "r:gz")  # pylint: disable=consider-using-with
    except (OSError, tarfile.TarError) as exc:
        raise VoiceError(f"cannot read voice archive {archive}: {exc}") from exc

    with tar:
        manifest = None
        for member in tar.getmembers():
            if member.name not in ALLOWED_MEMBERS or not member.isfile():
                raise VoiceError(f"unexpected member {member.name!r} in {archive}")
            if member.size > MAX_MEMBER_BYTES:
                raise VoiceError(f"member {member.name!r} in {archive} is too large")
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover - isfile() was checked above
                raise VoiceError(f"cannot read {member.name!r} from {archive}")
            manifest = handle.read()
    if manifest is None:
        raise VoiceError(f"{archive} has no {MANIFEST_NAME}")

    try:
        voice = Voice.from_dict(json.loads(manifest.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VoiceError(f"corrupt manifest in {archive}: {exc}") from exc

    if registry.exists(voice.voice_id) and not force:
        raise VoiceError(f"voice {voice.voice_id!r} already exists (use --force to replace it)")
    registry.save(voice)
    return voice
