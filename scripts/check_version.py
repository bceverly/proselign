#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Version gate: the source's ``__version__`` must be a valid W.X.Y.Z and, when a
release tag is given (``--tag v1.2.3.4`` or ``$GITHUB_REF_NAME`` on a tag
build), must match it exactly. Catches a tag pushed without `make release`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release import parse_version, read_marker  # noqa: E402  pylint: disable=wrong-import-position


def main(argv: list[str] | None = None) -> int:
    """Exit 0 when the version marker is consistent."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    version = read_marker()
    try:
        parse_version(version)
    except ValueError as exc:
        print(f"check_version: {exc}", file=sys.stderr)
        return 1

    tag = args.tag
    if tag is None and os.environ.get("GITHUB_REF_TYPE") == "tag":
        tag = os.environ.get("GITHUB_REF_NAME")
    if tag and tag.lstrip("v") != version:
        print(
            f"check_version: tag {tag} does not match __version__ {version}; "
            "cut releases with `make release`",
            file=sys.stderr,
        )
        return 1
    print(f"[OK] version {version}" + (f" matches tag {tag}" if tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
