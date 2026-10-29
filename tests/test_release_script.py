# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The pure helpers of scripts/release.py and the version gate.

Nothing here invokes version control; only the parsing and file-rewriting
logic is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import audit_deps  # noqa: E402  pylint: disable=wrong-import-position,import-error
import check_version  # noqa: E402  pylint: disable=wrong-import-position,import-error
import release  # noqa: E402  pylint: disable=wrong-import-position,import-error
import source_digest  # noqa: E402  pylint: disable=wrong-import-position,import-error

from proselign import __version__  # noqa: E402  pylint: disable=wrong-import-position


@pytest.mark.parametrize("bad", ["1.2.3", "1.2.3.4.5", "v1.2.3.4", "1.x.3.4", ""])
def test_parse_version_rejects(bad):
    with pytest.raises(ValueError):
        release.parse_version(bad)


def test_version_ordering_and_next_build():
    assert release.parse_version("0.10.0.0") > release.parse_version("0.9.9.9")
    assert release.parse_version("1.0.0.0") > release.parse_version("0.99.99.99")
    assert release.next_build("1.4.9.9") == "1.4.9.10"


def test_marker_round_trip(tmp_path: Path):
    path = tmp_path / "__init__.py"
    path.write_text('"""doc"""\n\n__version__ = "0.1.0.0"\n', "utf-8")
    assert release.read_marker(path) == "0.1.0.0"
    release.write_marker("1.0.0.0", path)
    assert release.read_marker(path) == "1.0.0.0"
    assert path.read_text("utf-8").startswith('"""doc"""')
    path.write_text("nothing\n", "utf-8")
    with pytest.raises(ValueError):
        release.read_marker(path)


def test_tags_from_refs():
    lines = [
        "abc123\trefs/tags/v0.1.0.0",
        "def456\trefs/tags/v0.1.0.0^{}",
        "0123 refs/tags/v0.2.0.1",
        "4567\trefs/tags/v0.3.0",  # three-part: not a release tag
        "9999\trefs/tags/not-a-release",
        "",
    ]
    assert release.tags_from_refs(lines) == {"0.1.0.0", "0.2.0.1"}


def test_source_marker_is_valid():
    assert release.read_marker() == __version__
    release.parse_version(__version__)


def test_check_version(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_REF_TYPE", raising=False)
    assert check_version.main([]) == 0
    assert check_version.main(["--tag", f"v{__version__}"]) == 0
    assert check_version.main(["--tag", "v999.0.0.0"]) == 1
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "v999.0.0.0")
    assert check_version.main([]) == 1
    assert "does not match" in capsys.readouterr().err


def test_choose_version():
    assert release.choose_version(None, "0.1.0.0", set()) == "0.1.0.0"  # first release
    assert release.choose_version(None, "0.1.0.0", {"0.1.0.4"}) == "0.1.0.5"
    assert release.choose_version(None, "0.2.0.0", {"0.1.0.4"}) == "0.2.0.1"
    assert release.choose_version("1.0.0.0", "0.1.0.0", {"0.2.0.0"}) == "1.0.0.0"
    for requested, tags in (
        ("0.0.9.9", set()),
        ("0.2.0.0", {"0.2.0.0"}),
        ("1.0.0", set()),
        ("bad", set()),
    ):
        with pytest.raises(release.Abort):
            release.choose_version(requested, "0.1.0.0", tags)


def test_source_digest(tmp_path: Path):
    pkg = tmp_path / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n", "utf-8")
    (pkg / "__pycache__" / "a.pyc").write_bytes(b"junk")
    assert source_digest.suffix(pkg) == "-dev"  # never released
    source_digest.record(pkg)
    assert source_digest.suffix(pkg) == ""
    (pkg / "_build.py").write_text("BUILD_SUFFIX = ''\n", "utf-8")  # excluded
    assert source_digest.suffix(pkg) == ""
    (pkg / "a.py").write_text("x = 2\n", "utf-8")
    assert source_digest.suffix(pkg) == "-dev"


def test_full_version_reads_build_stamp(monkeypatch):
    import types

    import proselign

    assert proselign.full_version() == proselign.__version__
    monkeypatch.setitem(sys.modules, "proselign._build", types.SimpleNamespace(BUILD_SUFFIX="-dev"))
    assert proselign.full_version() == proselign.__version__ + "-dev"


def test_confirm(monkeypatch):
    release.confirm("v1.0.0", assume_yes=True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(release.Abort, match="YES=1"):
        release.confirm("v1.0.0", assume_yes=False)


def test_audit_ignore_parsing():
    import datetime as dt

    today = dt.date(2026, 10, 29)
    text = (
        "# comment\n\n"
        "PYSEC-1  2027-01-01  a perfectly good reason\n"
        "PYSEC-2  2026-01-01  this one has expired already\n"
        "PYSEC-3  someday    unparseable date here\n"
        "PYSEC-4  2027-01-01  short\n"
        "PYSEC-5  2027-01-01  [requirements-x.txt]  scoped to one lock file\n"
    )
    ids, problems = audit_deps.parse_ignores(text, today)
    assert ids == [("PYSEC-1", None), ("PYSEC-5", "requirements-x.txt")]
    assert len(problems) == 3
    assert any("expired" in p for p in problems)
