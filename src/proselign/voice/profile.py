# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The voice manifest: a JSON document describing one author's style.

Quantitative measurements (``metrics``) are kept apart from interpretive
judgments (``style``) so a reader can always tell which is which.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from proselign.errors import VoiceError
from proselign.paths import ensure_private_dir, write_private_file

SCHEMA_VERSION = 1
MANIFEST_NAME = "voice.json"

_VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def voice_id_for(name: str) -> str:
    """Normalize a display name to the on-disk identifier.

    Rejects anything that could escape the voices directory, so a name is
    always safe to join onto a path.
    """
    voice_id = name.strip().lower()
    if not _VOICE_ID_RE.match(voice_id):
        raise VoiceError(
            f"invalid voice name {name!r}: use 1-64 letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return voice_id


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Voice:  # pylint: disable=too-many-instance-attributes
    """In-memory form of ``voice.json``."""

    display_name: str
    voice_id: str = ""
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    analysis_model: str = "stylometry"
    style: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    genre_profiles: dict[str, Any] = field(default_factory=dict)
    # Diction: function-word rates, word preferences, openers, conventions.
    lexicon: dict[str, Any] = field(default_factory=dict)
    # Typical passages of the author's own prose (private; see export).
    examples: list[str] = field(default_factory=list)
    corpus_summary: dict[str, Any] = field(
        default_factory=lambda: {"documents": 0, "tokens_estimated": 0}
    )
    sources: list[dict[str, Any]] = field(default_factory=list)
    license_and_consent: dict[str, Any] = field(
        default_factory=lambda: {"raw_examples_exportable": False}
    )

    def __post_init__(self) -> None:
        if not self.voice_id:
            self.voice_id = voice_id_for(self.display_name)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, keys in a stable order."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Voice:
        """Build from parsed JSON, rejecting unknown schema versions."""
        if not isinstance(data, dict):
            raise VoiceError("voice manifest must be a JSON object")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise VoiceError(
                f"unsupported voice schema_version {version!r} (expected {SCHEMA_VERSION})"
            )
        if "display_name" not in data:
            raise VoiceError("voice manifest is missing display_name")
        known = set(cls.__dataclass_fields__)  # pylint: disable=no-member
        voice = cls(**{k: v for k, v in data.items() if k in known})
        voice_id_for(voice.voice_id)  # validates a hand-edited id too
        return voice

    def save(self, directory: Path) -> Path:
        """Write ``voice.json`` into ``directory`` atomically, private to the user."""
        ensure_private_dir(directory)
        target = directory / MANIFEST_NAME
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"
        write_private_file(target, payload.encode("utf-8"))
        return target

    @classmethod
    def load(cls, directory: Path) -> Voice:
        """Read ``voice.json`` from ``directory``."""
        path = directory / MANIFEST_NAME
        try:
            data = json.loads(path.read_text("utf-8"))
        except FileNotFoundError as exc:
            raise VoiceError(f"no voice manifest at {path}") from exc
        except json.JSONDecodeError as exc:
            raise VoiceError(f"corrupt voice manifest {path}: {exc}") from exc
        return cls.from_dict(data)
