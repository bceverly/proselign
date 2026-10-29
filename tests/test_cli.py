# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Golden tests for the CLI contract: clean stdout, stderr diagnostics, exit codes."""

from __future__ import annotations

import json

# Runs this package's own entry point with a fixed argv.
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from proselign import __version__
from proselign.errors import EXIT_BACKEND, EXIT_OK, EXIT_USAGE, EXIT_VALIDATION, EXIT_VOICE

from .conftest import run

DOC = """---
title: Draft
---
# Heading

We shipped 3 releases in 2026, see https://example.com/notes for details.

```python
print("code is never rewritten")
```

| a | b |
|---|---|
| 1 | 2 |

Closing paragraph with a "quoted phrase" and a citation [1].
"""


def test_filter_mode_stdout_is_content_only(trained):
    code, out, err = run(["--voice", "Bryan", "--backend", "mock"], DOC, trained)
    assert code == EXIT_OK
    assert out == DOC  # mock backend echoes; reassembly must be lossless
    assert err == ""


def test_explicit_rewrite_subcommand_and_files(trained, tmp_path: Path):
    src = tmp_path / "in.md"
    src.write_text("First file.\n", "utf-8")
    code, out, _ = run(
        ["rewrite", "--voice", "bryan", "--backend", "mock", str(src), "-"],
        "From stdin.\n",
        trained,
    )
    assert code == EXIT_OK
    assert out == "First file.\nFrom stdin.\n"


def test_missing_input_file_is_usage_error(trained, tmp_path: Path):
    code, out, err = run(
        ["--voice", "Bryan", "--backend", "mock", str(tmp_path / "nope.txt")], "", trained
    )
    assert code == EXIT_USAGE
    assert out == ""
    assert "cannot read" in err


def test_json_diagnostics_on_stderr(trained):
    code, out, err = run(
        ["--voice", "Bryan", "--backend", "mock", "--json-diagnostics"], "Hello there.\n", trained
    )
    assert code == EXIT_OK
    assert out == "Hello there.\n"
    report = json.loads(err)
    assert report["chunks"] == 1
    assert report["fallbacks"] == 0


def test_unknown_voice(registry):
    code, out, err = run(["--voice", "Nobody", "--backend", "mock"], "x", registry)
    assert code == EXIT_VOICE
    assert out == ""
    assert "proselign train --voice Nobody" in err


def test_bad_strength_is_usage_error(trained):
    code, _, err = run(["--voice", "Bryan", "--backend", "mock", "--strength", "2"], "x", trained)
    assert code == EXIT_USAGE
    assert "--strength" in err


def test_unknown_option_is_usage_error(registry):
    code, out, err = run(["--voice", "B", "--bogus"], "", registry)
    assert code == EXIT_USAGE
    assert out == ""
    assert "unrecognized arguments" in err


def test_no_arguments_prints_help(registry):
    code, out, err = run([], "", registry)
    assert code == EXIT_USAGE
    assert out == ""
    assert "usage:" in err


def test_offline_rejects_remote_endpoint(trained):
    code, _, err = run(
        ["--voice", "Bryan", "--offline", "--endpoint", "http://203.0.113.9:8080/v1"], "x", trained
    )
    assert code == EXIT_USAGE
    assert "--offline" in err


def test_unreachable_backend_is_backend_error(trained):
    code, out, err = run(
        ["--voice", "Bryan", "--endpoint", "http://127.0.0.1:9/v1", "--offline"], "Hi.\n", trained
    )
    assert code == EXIT_BACKEND
    assert out == ""
    assert "cannot reach" in err


def test_strict_failure_exits_validation(trained, monkeypatch):
    from proselign.inference.mock import MockBackend

    monkeypatch.setattr(MockBackend, "rewrite", lambda self, m, t: "numbers vanished")
    code, out, err = run(
        ["--voice", "Bryan", "--backend", "mock", "--strict"], "We had 42 cats.\n", trained
    )
    assert code == EXIT_VALIDATION
    assert out == ""
    assert "42" in err


def test_train_twice_requires_append(trained, corpus):
    code, _, err = run(["train", "--voice", "Bryan", str(corpus)], registry=trained)
    assert code == EXIT_USAGE
    assert "--append" in err
    code, _, err = run(["train", "--voice", "Bryan", "--append", str(corpus)], registry=trained)
    assert code == EXIT_OK, err


def test_voices_inspect_export_import_delete(trained, tmp_path: Path):
    code, out, _ = run(["voices"], registry=trained)
    assert code == EXIT_OK
    assert out.startswith("bryan\tBryan\t")

    code, out, _ = run(["inspect", "Bryan"], registry=trained)
    assert code == EXIT_OK
    assert json.loads(out)["voice_id"] == "bryan"

    archive = tmp_path / "b.proselign"
    code, out, err = run(["export", "Bryan", "-o", str(archive)], registry=trained)
    assert code == EXIT_OK
    assert out == ""
    assert "corpus paths omitted" in err

    code, _, err = run(["import", str(archive)], registry=trained)
    assert code == EXIT_VOICE
    assert "already exists" in err

    code, _, _ = run(["delete", "Bryan"], registry=trained)
    assert code == EXIT_OK
    code, _, _ = run(["import", str(archive)], registry=trained)
    assert code == EXIT_OK
    code, _, _ = run(["delete", "Bryan"], registry=trained)
    code, _, err = run(["delete", "Bryan"], registry=trained)
    assert code == EXIT_VOICE


def test_export_default_filename(trained, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, _, _ = run(["export", "Bryan", "--include-sources"], registry=trained)
    assert code == EXIT_OK
    assert (tmp_path / "Bryan.proselign").is_file()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_python_dash_m_entry_point(tmp_path: Path):
    """The real process boundary: exit status and stream separation."""
    env = {"PROSELIGN_HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    src = str(Path(__file__).resolve().parent.parent / "src")
    proc = subprocess.run(  # nosec B603
        [sys.executable, "-m", "proselign", "--voice", "ghost", "--backend", "mock"],
        input="text",
        capture_output=True,
        text=True,
        env={**env, "PYTHONPATH": src},
        check=False,
    )
    assert proc.returncode == EXIT_VOICE
    assert proc.stdout == ""
    assert "unknown voice" in proc.stderr


def test_train_warns_about_unreadable_documents(corpus: Path, registry):
    (corpus / "damaged.docx").write_bytes(b"PK\x03\x04 truncated")
    code, _, err = run(["train", "--voice", "Bryan", str(corpus)], registry=registry)
    assert code == EXIT_OK
    assert "warning: skipped" in err
    assert "damaged.docx: corrupt .docx archive" in err
