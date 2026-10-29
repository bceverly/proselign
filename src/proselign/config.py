# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The user's ``~/.proselignrc``: defaults and style switches for rewrites.

An INI file (see ``examples/proselignrc``).  Precedence, highest first:
command-line options, ``PROSELIGN_*`` environment variables, this file,
built-in defaults.  The file can point rewrites at an endpoint, so it is
refused when anyone but its owner could have changed it.
"""

from __future__ import annotations

import configparser
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from proselign.errors import UsageError
from proselign.rewrite.style import StyleRules

CONFIG_NAME = ".proselignrc"
MAX_CONFIG_BYTES = 64 * 1024

_STRINGS = {("rewrite", "voice"), ("backend", "backend"), ("backend", "endpoint")}
_STRINGS |= {("backend", "model"), ("rewrite", "format")}
_BOOLEANS = {("rewrite", "strict"), ("backend", "offline")}
# [style] key -> StyleRules field
_STYLE_KEYS = {
    "passive_voice": "passive_voice",
    "en_dashes": "en_dashes",
    "em_dashes": "em_dashes",
    "exclamation_marks": "exclamation_marks",
    "semicolons": "semicolons",
    "ai_artifact_filter": "ai_filter",
}


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes  # one per setting
    """Settings from the file; ``None`` means "not set there"."""

    path: Path | None = None
    voice: str | None = None
    strength: float | None = None
    text_format: str | None = None
    strict: bool = False
    backend: str | None = None
    endpoint: str | None = None
    model: str | None = None
    offline: bool = False
    rules: StyleRules = field(default_factory=StyleRules)
    warnings: list[str] = field(default_factory=list)


def config_path() -> Path:
    """``$PROSELIGN_CONFIG`` if set, else ``~/.proselignrc``."""
    override = os.environ.get("PROSELIGN_CONFIG")
    return Path(override) if override else Path.home() / CONFIG_NAME


def load(path: Path | None = None) -> Config:
    """Read the configuration; a missing file means all defaults."""
    path = path or config_path()
    try:
        info = path.stat()
    except FileNotFoundError:
        return Config()
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc.strerror}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise UsageError(f"{path} is not a regular file")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise UsageError(f"{path} is writable by other users; run: chmod go-w {path}")
    if info.st_size > MAX_CONFIG_BYTES:
        raise UsageError(f"{path} is larger than {MAX_CONFIG_BYTES // 1024} KB")
    # "key = value  # note" comments are allowed; a URL's "#fragment" has no
    # space before it, so it is left alone.
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    try:
        parser.read_string(path.read_text("utf-8"), source=str(path))
    except (OSError, UnicodeDecodeError, configparser.Error) as exc:
        raise UsageError(f"cannot parse {path}: {exc}") from exc
    return _from_parser(parser, path)


def _from_parser(parser: configparser.ConfigParser, path: Path) -> Config:
    config = Config(path=path)
    style: dict[str, bool] = {}
    for section in parser.sections():
        for key in parser[section]:
            try:
                _apply(config, style, parser, section, key)
            except ValueError as exc:
                raise UsageError(f"{path}: [{section}] {key}: {exc}") from exc
    config.rules = StyleRules(**style)
    return config


def _apply(
    config: Config,
    style: dict[str, bool],
    parser: configparser.ConfigParser,
    section: str,
    key: str,
) -> None:
    item = (section, key)
    if item in _STRINGS:
        name = "text_format" if key == "format" else key
        setattr(config, name, parser.get(section, key).strip() or None)
    elif item in _BOOLEANS:
        setattr(config, key, parser.getboolean(section, key))
    elif item == ("rewrite", "strength"):
        config.strength = parser.getfloat(section, key)
    elif section == "style" and key in _STYLE_KEYS:
        style[_STYLE_KEYS[key]] = parser.getboolean(section, key)
    else:
        config.warnings.append(f"{config.path}: ignoring unknown setting [{section}] {key}")
