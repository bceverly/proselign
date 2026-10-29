# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Voice manifests, the registry and portable archives."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from proselign import paths
from proselign.errors import VoiceError
from proselign.voice.package import MAX_MEMBER_BYTES, export_voice, import_voice
from proselign.voice.profile import MANIFEST_NAME, SCHEMA_VERSION, Voice, voice_id_for


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", ".hidden", "x" * 65, "spaces here"])
def test_voice_id_rejects_unsafe_names(bad):
    with pytest.raises(VoiceError):
        voice_id_for(bad)


def test_voice_round_trip(tmp_path: Path):
    voice = Voice(display_name="Ada", style={"tone": ["precise"]})
    voice.save(tmp_path)
    loaded = Voice.load(tmp_path)
    assert loaded == voice
    assert loaded.voice_id == "ada"


@pytest.mark.parametrize(
    "data,match",
    [
        ([], "JSON object"),
        ({"schema_version": 99, "display_name": "x"}, "schema_version"),
        ({"schema_version": SCHEMA_VERSION}, "display_name"),
        ({"schema_version": SCHEMA_VERSION, "display_name": "x", "voice_id": "../x"}, "invalid"),
    ],
)
def test_from_dict_validation(data, match):
    with pytest.raises(VoiceError, match=match):
        Voice.from_dict(data)


def test_from_dict_ignores_unknown_fields():
    voice = Voice.from_dict(
        {"schema_version": SCHEMA_VERSION, "display_name": "x", "future_field": 1}
    )
    assert voice.voice_id == "x"


def test_load_errors(tmp_path: Path):
    with pytest.raises(VoiceError, match="no voice manifest"):
        Voice.load(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("{not json", "utf-8")
    with pytest.raises(VoiceError, match="corrupt"):
        Voice.load(tmp_path)


def test_registry_lists_skipping_corrupt(registry):
    assert registry.list() == []
    registry.save(Voice(display_name="B"))
    registry.save(Voice(display_name="A"))
    bad = registry.root / "broken"
    bad.mkdir()
    (bad / MANIFEST_NAME).write_text("nope", "utf-8")
    assert [v.voice_id for v in registry.list()] == ["a", "b"]


def test_paths_respect_xdg(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PROSELIGN_HOME")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.voices_dir() == tmp_path / "proselign" / "voices"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.data_home() == tmp_path / ".local" / "share" / "proselign"


def test_export_strips_sources_by_default(registry, tmp_path: Path):
    voice = Voice(display_name="C", sources=[{"path": "/private/diary.md", "sha256": "ab"}])
    registry.save(voice)
    archive = export_voice(registry, "C", tmp_path / "c.proselign")
    with tarfile.open(archive) as tar:
        data = json.loads(tar.extractfile(MANIFEST_NAME).read())
    assert data["sources"] == []
    archive = export_voice(registry, "C", tmp_path / "c2.proselign", include_sources=True)
    with tarfile.open(archive) as tar:
        assert json.loads(tar.extractfile(MANIFEST_NAME).read())["sources"]


def _archive(tmp_path: Path, members: dict[str, bytes], name="evil.proselign") -> Path:
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        for member, payload in members.items():
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({"../../etc/passwd": b"x"}, "unexpected member"),
        ({}, "has no"),
        ({MANIFEST_NAME: b"\xff\xfe"}, "corrupt manifest"),
    ],
)
def test_import_rejects_hostile_archives(registry, tmp_path: Path, members, message):
    archive = _archive(tmp_path, members)
    with pytest.raises(VoiceError, match=message):
        import_voice(registry, archive)


def test_import_rejects_non_archives(registry, tmp_path: Path):
    not_tar = tmp_path / "plain.proselign"
    not_tar.write_text("hello", "utf-8")
    with pytest.raises(VoiceError, match="cannot read"):
        import_voice(registry, not_tar)


def test_import_rejects_oversized_member(registry, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("proselign.voice.package.MAX_MEMBER_BYTES", 4)
    assert MAX_MEMBER_BYTES > 4
    archive = _archive(tmp_path, {MANIFEST_NAME: b"12345"})
    with pytest.raises(VoiceError, match="too large"):
        import_voice(registry, archive)
