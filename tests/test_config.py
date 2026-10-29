# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""~/.proselignrc loading and how it combines with options and environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from proselign import config as config_mod
from proselign.errors import EXIT_OK, EXIT_USAGE, UsageError
from proselign.inference import OpenAICompatBackend, make_backend
from proselign.rewrite.style import StyleRules

from .conftest import run

REPO = Path(__file__).resolve().parent.parent


def _rc(tmp_path: Path, text: str, mode: int = 0o600) -> Path:
    path = tmp_path / "rc"
    path.write_text(text, "utf-8")
    os.chmod(path, mode)
    return path


def test_missing_file_means_defaults(tmp_path: Path):
    assert config_mod.load(tmp_path / "nope") == config_mod.Config()


def test_default_location(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PROSELIGN_CONFIG")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_mod.config_path() == tmp_path / ".proselignrc"


def test_sample_file_parses_to_defaults_with_offline(tmp_path: Path):
    sample = config_mod.load(_rc(tmp_path, (REPO / "examples" / "proselignrc").read_text("utf-8")))
    assert sample.rules == StyleRules()
    assert (sample.voice, sample.strength, sample.text_format) == ("Bryan", 0.5, "markdown")
    assert (sample.backend, sample.model, sample.offline, sample.strict) == (
        "openai-compat",
        "local",
        True,
        False,
    )
    assert not sample.warnings


def test_every_setting_and_unknown_keys(tmp_path: Path):
    path = _rc(
        tmp_path,
        "[rewrite]\nvoice = Mail  # inline note\nstrength = 0.8\nformat = text\n"
        "strict = yes\ncolour = red\n"
        "[backend]\nendpoint = http://127.0.0.1:1/v1#frag\nmodel =\n"
        "[style]\npassive_voice = off\nen_dashes = no\nem_dashes = false\n"
        "exclamation_marks = 0\nsemicolons = off\nai_artifact_filter = off\n[extra]\nx = 1\n",
    )
    settings = config_mod.load(path)
    assert (settings.voice, settings.strength, settings.text_format, settings.strict) == (
        "Mail",
        0.8,
        "text",
        True,
    )
    assert settings.endpoint == "http://127.0.0.1:1/v1#frag"
    assert settings.model is None
    assert settings.rules == StyleRules(False, False, False, False, False, False)
    assert [w.split("] ")[-1] for w in settings.warnings] == ["colour", "x"]


@pytest.mark.parametrize(
    "text,message",
    [
        ("[style]\nem_dashes = maybe\n", "em_dashes"),
        ("[rewrite]\nstrength = lots\n", "strength"),
        ("no section header\n", "cannot parse"),
    ],
)
def test_bad_values_are_usage_errors(tmp_path: Path, text, message):
    path = _rc(tmp_path, text)
    with pytest.raises(UsageError, match=message):
        config_mod.load(path)


def test_unsafe_or_odd_files_are_refused(tmp_path: Path, monkeypatch):
    path = _rc(tmp_path, "[style]\n", 0o622)
    with pytest.raises(UsageError, match="chmod go-w"):
        config_mod.load(path)
    with pytest.raises(UsageError, match="not a regular file"):
        config_mod.load(tmp_path)
    monkeypatch.setattr(config_mod, "MAX_CONFIG_BYTES", 2)
    os.chmod(path, 0o600)
    with pytest.raises(UsageError, match="larger than"):
        config_mod.load(path)
    path.write_bytes(b"\xff\xfe")
    monkeypatch.setattr(config_mod, "MAX_CONFIG_BYTES", 64)
    with pytest.raises(UsageError, match="cannot parse"):
        config_mod.load(path)


def test_unreadable_location(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("x", "utf-8")
    with pytest.raises(UsageError, match="cannot read"):
        config_mod.load(blocker / "rc")


def test_backend_precedence(monkeypatch):
    defaults = {"backend": None, "endpoint": "http://127.0.0.1:2/v1", "model": "rc-model"}
    backend = make_backend(defaults=defaults)
    assert isinstance(backend, OpenAICompatBackend)
    assert (backend.endpoint, backend.model) == ("http://127.0.0.1:2/v1", "rc-model")
    monkeypatch.setenv("PROSELIGN_MODEL", "env-model")
    assert make_backend(defaults=defaults).model == "env-model"
    assert make_backend(model="cli", defaults=defaults).model == "cli"
    assert make_backend(defaults={"backend": "mock"}).name == "mock"


def test_cli_uses_rc_voice_and_rules(trained, tmp_path: Path, monkeypatch):
    path = _rc(
        tmp_path,
        "[rewrite]\nvoice = Bryan\nstrength = 0.9\n[backend]\nbackend = mock\n"
        "[style]\nem_dashes = off\nbogus = 1\n",
    )
    monkeypatch.setenv("PROSELIGN_CONFIG", str(path))
    code, out, err = run(["-", "--json-diagnostics"], "Fast\u2014really fast.\n", registry=trained)
    assert code == EXIT_OK, err
    assert out == "Fast, really fast.\n"
    assert "unknown setting [style] bogus" in err
    assert json.loads(err.splitlines()[-1])["chunks"] == 1

    code, out, _ = run(
        ["--no-config", "--voice", "Bryan", "--backend", "mock"], "A\u2014b.\n", registry=trained
    )
    assert (code, out) == (EXIT_OK, "A\u2014b.\n")
    other = _rc(tmp_path / "..", "[rewrite]\nvoice = Nobody\n")
    code, _, err = run(["--config", str(other), "--backend", "mock", "-"], "x", registry=trained)
    assert code != EXIT_OK and "Nobody" in err


def test_cli_without_any_voice(trained):
    code, _, err = run(["rewrite", "--backend", "mock"], "x", registry=trained)
    assert code == EXIT_USAGE
    assert "no voice given" in err


@pytest.mark.parametrize(
    "value,expected",
    [("we", "plural"), ("Plural", "plural"), ("I", "singular"), ("voice", "voice")],
)
def test_first_person_setting(tmp_path: Path, value, expected):
    settings = config_mod.load(_rc(tmp_path, f"[style]\nfirst_person = {value}\n"))
    assert settings.rules.first_person == expected


def test_first_person_setting_rejects_nonsense(tmp_path: Path):
    with pytest.raises(UsageError, match="plural"):
        config_mod.load(_rc(tmp_path, "[style]\nfirst_person = they\n"))
