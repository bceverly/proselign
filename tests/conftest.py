# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Shared fixtures: an isolated data directory and a small sample corpus."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from proselign.cli import main
from proselign.voice.registry import Registry

SAMPLE_PARAGRAPHS = [
    "I don't like long meetings. We get more done when we write things down first, "
    "and then we talk about what's actually unclear.",
    "Here's the thing about design documents: they're cheap. A page of prose costs an hour. "
    "A wrong system costs a quarter. You can do that arithmetic yourself.",
    "When I review a proposal, I look for the failure modes first. What happens when the "
    "disk fills? What happens when the network drops? If the author can't answer, we wait.",
    "Short sentences help. So do concrete examples. I'd rather read one real log line than "
    "three paragraphs about observability in the abstract.",
]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own empty voice store and a clean backend config."""
    home = tmp_path / "home"
    monkeypatch.setenv("PROSELIGN_HOME", str(home))
    for var in ("PROSELIGN_BACKEND", "PROSELIGN_ENDPOINT", "PROSELIGN_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture
def registry(isolated_home: Path) -> Registry:
    return Registry(isolated_home / "voices")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A directory of .md/.txt files with enough words to train on."""
    root = tmp_path / "corpus"
    (root / "notes").mkdir(parents=True)
    (root / ".hidden").mkdir()
    for i in range(3):
        body = "\n\n".join(SAMPLE_PARAGRAPHS[i:] + SAMPLE_PARAGRAPHS[:i])
        (root / f"essay{i}.md").write_text(f"# Essay {i}\n\n{body}\n", "utf-8")
    (root / "notes" / "memo.txt").write_text("\n\n".join(SAMPLE_PARAGRAPHS) + "\n", "utf-8")
    (root / "notes" / "dup.txt").write_text("\n\n".join(SAMPLE_PARAGRAPHS) + "\n", "utf-8")
    (root / ".hidden" / "secret.md").write_text("never ingest me " * 50, "utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    return root


def run(argv: list[str], stdin: str = "", registry: Registry | None = None):
    """Invoke the CLI in-process; returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err, registry=registry)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def trained(corpus: Path, registry: Registry) -> Registry:
    """A registry holding a voice named Bryan."""
    code, _, err = run(["train", "--voice", "Bryan", str(corpus)], registry=registry)
    assert code == 0, err
    return registry
