#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Fingerprint of the shipped source, used to tell releases from dev builds.

``make release`` records the digest in ``src/proselign/RELEASE_DIGEST``;
``make install`` recomputes it and, when they differ, stamps the installed
version as ``W.X.Y.Z-dev``. Content-based on purpose: it needs no version
control and works the same in a tarball.

    source_digest.py            print the current digest
    source_digest.py --suffix   print "" for a release tree, "-dev" otherwise
    source_digest.py --record   write the current digest to RELEASE_DIGEST
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "proselign"
DIGEST_FILE = PACKAGE / "RELEASE_DIGEST"
EXCLUDED = {"RELEASE_DIGEST", "_build.py"}


def digest(package: Path = PACKAGE) -> str:
    """SHA-256 over every shipped file's relative path and bytes."""
    hasher = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED or "__pycache__" in path.parts:
            continue
        hasher.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        hasher.update(path.read_bytes() + b"\0")
    return hasher.hexdigest()


def suffix(package: Path = PACKAGE) -> str:
    """``""`` when the tree matches the last release, else ``"-dev"``."""
    recorded = package / "RELEASE_DIGEST"
    try:
        released = recorded.read_text("utf-8").strip()
    except FileNotFoundError:
        return "-dev"
    return "" if released == digest(package) else "-dev"


def record(package: Path = PACKAGE) -> str:
    """Store the current digest as the release digest."""
    value = digest(package)
    (package / "RELEASE_DIGEST").write_text(value + "\n", "utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--suffix", action="store_true")
    group.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.suffix:
        print(suffix())
    elif args.record:
        print(record())
    else:
        print(digest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
