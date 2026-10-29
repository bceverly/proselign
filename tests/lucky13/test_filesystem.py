# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Lucky 13 classes about files, permissions, links and the network."""

from __future__ import annotations

import io
import os
import re
import stat
import tarfile
from pathlib import Path

import pytest

from proselign.errors import EXIT_OK, EXIT_VOICE
from proselign.ingest.corpus import find_files
from proselign.voice.profile import MANIFEST_NAME, Voice

from ..conftest import SAMPLE_PARAGRAPHS, run
from .conftest import REPO, calls, imported_modules, parsed_sources


@pytest.fixture
def umask_zero():
    old = os.umask(0)
    yield
    os.umask(old)


def _world_writable(path: Path) -> bool:
    return bool(path.lstat().st_mode & stat.S_IWOTH)


def test_06_world_writable_cwe276_store_private_under_umask_0(corpus, isolated_home, umask_zero):
    """#6 World-writable files: even with ``umask 000`` the voice store is
    0700 and manifests 0600 -- a corpus profile is private data."""
    code, _, err = run(["train", "--voice", "Bryan", str(corpus)])
    assert code == EXIT_OK, err
    for path in [isolated_home, *isolated_home.rglob("*")]:
        mode = stat.S_IMODE(path.lstat().st_mode)
        expected = 0o700 if path.is_dir() else 0o600
        assert mode == expected, f"{path} is {oct(mode)}"


def test_06_world_writable_cwe276_exported_archive(trained, tmp_path, umask_zero):
    """#6: exported archives are owner-only (0600), even under umask 000."""
    out = tmp_path / "b.proselign"
    assert run(["export", "Bryan", "-o", str(out)], registry=trained)[0] == EXIT_OK
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_06_world_writable_cwe276_repository_tree():
    """#6: nothing shipped from the repository is world-writable, and no
    code sets permissive modes."""
    for top in ("src", "scripts", "packaging", "man", "website"):
        for path in (REPO / top).rglob("*"):
            assert not _world_writable(path), path
    for path, _ in parsed_sources():
        assert not re.search(r"0o?7[0-7]7|0o?666", path.read_text("utf-8")), path


def test_07_direct_request_cwe425_no_listening_service():
    """#7 Direct request: proselign exposes no endpoint to request directly --
    it never binds or listens, and only connects to the endpoint the user
    configured (loopback-only under --offline)."""
    for path, tree in parsed_sources():
        mods = {m.split(".")[0] for m in imported_modules(tree)}
        assert not mods & {"http.server", "socketserver", "asyncio", "xmlrpc"}, path
        assert "http.server" not in imported_modules(tree), path
        for name, _ in calls(tree):
            assert name.rsplit(".", 1)[-1] not in {
                "bind",
                "listen",
                "accept",
                "serve_forever",
            }, f"{name}() in {path}"


def test_11_symlink_following_cwe61_corpus_walk(tmp_path):
    """#11 Symlinks: a link planted in a corpus directory (to a key, to /etc,
    or to a whole directory) is never followed during ingestion."""
    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    secret = secret_dir / "id_rsa.md"
    secret.write_text("PRIVATE KEY MATERIAL", "utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "real.md").write_text("\n\n".join(SAMPLE_PARAGRAPHS), "utf-8")
    (corpus / "leak.md").symlink_to(secret)
    (corpus / "linked_dir").symlink_to(secret_dir, target_is_directory=True)
    assert [p.name for p in find_files([corpus])] == ["real.md"]


def test_11_symlink_following_cwe61_manifest_write(registry, tmp_path):
    """#11: a pre-planted link at the manifest's temp path or the manifest
    itself cannot redirect a write outside the voice directory."""
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", "utf-8")
    voice_dir = registry.root / "bryan"
    voice_dir.mkdir(parents=True)
    (voice_dir / MANIFEST_NAME).symlink_to(victim)
    registry.save(Voice(display_name="Bryan"))
    assert victim.read_text("utf-8") == "untouched"
    assert not (voice_dir / MANIFEST_NAME).is_symlink()


def test_11_symlink_following_cwe61_delete_and_archives(registry, tmp_path):
    """#11: deleting a voice whose directory is a link refuses; archive
    members that are links are refused, never resolved."""
    target = tmp_path / "precious"
    target.mkdir()
    Voice(display_name="Bryan").save(target)
    registry.root.mkdir(parents=True, exist_ok=True)
    (registry.root / "bryan").symlink_to(target, target_is_directory=True)
    code, _, err = run(["delete", "Bryan"], registry=registry)
    assert code == EXIT_VOICE and "symbolic link" in err
    assert (target / MANIFEST_NAME).exists()

    archive = tmp_path / "link.proselign"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info, io.BytesIO(b""))
    code, _, err = run(["import", str(archive), "--force"], registry=registry)
    assert code == EXIT_VOICE and "unexpected member" in err
