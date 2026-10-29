# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""XDG-compliant storage locations, created private to the user."""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def data_home() -> Path:
    """Root of proselign's persistent data.

    ``PROSELIGN_HOME`` overrides everything (tests and portable installs use
    it); otherwise ``$XDG_DATA_HOME/proselign``, defaulting per the XDG spec.
    """
    override = os.environ.get("PROSELIGN_HOME")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "proselign"


def voices_dir() -> Path:
    """Directory holding one sub-directory per voice."""
    return data_home() / "voices"


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` and any missing parents as 0700, whatever the umask.

    ``Path.mkdir(parents=True)`` creates parents with umask-derived modes,
    so under ``umask 000`` a voice store would be world-writable; each
    missing level is created (and re-moded) explicitly instead.
    """
    missing = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        probe = probe.parent
    for directory in reversed(missing):
        directory.mkdir(mode=PRIVATE_DIR_MODE)
        # 0700 is the most restrictive usable mode; it defeats a permissive umask.
        os.chmod(directory, PRIVATE_DIR_MODE)  # nosemgrep: insecure-file-permissions
    return path


def write_private_file(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data``, readable only by the user.

    The temporary file is created exclusively and without following
    symlinks, so a planted link cannot redirect the write.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, PRIVATE_FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
