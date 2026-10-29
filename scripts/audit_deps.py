#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Dependency gate for ``make security``: pip-audit every hash-locked file.

The escape hatch is ``.pip-audit-ignore`` at the repository root, one entry
per line::

    PYSEC-2026-2447  2027-01-31  [requirements-llm.txt]  why it does not apply here

An entry needs an ID, an expiry date and a reason, and may name the one lock
file it applies to (so an exception for an isolated tool venv cannot mask
the same package elsewhere). A malformed entry is refused, and an expired
one fails the gate. An ignore is a reviewed
decision with a date on it, not a way to make a finding go away.
"""

from __future__ import annotations

import datetime as dt
import subprocess  # nosec B404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / ".pip-audit-ignore"
LOCKS = (
    "requirements-dev.txt",
    "requirements-llm.txt",
    "requirements-semgrep.txt",
    "requirements-safety.txt",
)


def parse_ignores(text: str, today: dt.date) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Return ((vulnerability ID, lock file or None for all), problems with the file)."""
    entries: list[tuple[str, str | None]] = []
    problems = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        scope = None
        if len(parts) == 3 and parts[2].startswith("["):
            scope_text, _, rest = parts[2].partition("]")
            scope, parts[2] = scope_text[1:].strip(), rest.strip()
        if len(parts) < 3 or len(parts[2]) < 10:
            problems.append(f"line {number}: need 'ID YYYY-MM-DD [lock.txt] reason (10+ chars)'")
            continue
        vuln, expires, _reason = parts
        try:
            expiry = dt.date.fromisoformat(expires)
        except ValueError:
            problems.append(f"line {number}: bad expiry date {expires!r}")
            continue
        if expiry < today:
            problems.append(f"line {number}: ignore for {vuln} expired on {expires}; re-review it")
            continue
        entries.append((vuln, scope))
    return entries, problems


def main() -> int:
    """Audit every lock file; non-zero on any unignored finding."""
    text = IGNORE_FILE.read_text("utf-8") if IGNORE_FILE.exists() else ""
    entries, problems = parse_ignores(text, dt.date.today())
    if problems:
        for problem in problems:
            print(f"audit_deps: {IGNORE_FILE.name}: {problem}", file=sys.stderr)
        return 1
    status = 0
    for lock in LOCKS:
        ids = [vuln for vuln, scope in entries if scope in (None, lock)]
        flags = [arg for vuln in ids for arg in ("--ignore-vuln", vuln)]
        print(f"--- {lock}", flush=True)
        argv = [sys.executable, "-m", "pip_audit", "--progress-spinner", "off"]
        argv += ["--require-hashes", "--disable-pip", "-r", str(ROOT / lock), *flags]
        status |= subprocess.run(argv, check=False).returncode  # nosec B603
    return status


if __name__ == "__main__":
    sys.exit(main())
