#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Cut a release: bump the version marker, COMMIT, then TAG, then push.

Tagging last is the point: the tag must name the commit that carries the
bump, because CI checks out the tag and builds exactly that tree.

    make release                      # last part + 1 (0.1.0.0 -> 0.1.0.1), prompts y/N
    make release VERSION=1.0.0.0      # explicit version, prompts y/N
    make release VERSION=1.0.0.0 DRY_RUN=1
    make release VERSION=1.0.0.0 YES=1   # non-interactive (CI / scripts)

Every git call goes through subprocess with an argument list and no shell.
"""

from __future__ import annotations

import argparse
import re

# Fixed argv lists only, never a shell.
import subprocess  # nosec B404
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# pylint: disable=wrong-import-position
from source_digest import DIGEST_FILE  # noqa: E402
from source_digest import record as record_digest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "src" / "proselign" / "__init__.py"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)$")
MARKER_RE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
TAG_RE = re.compile(r"^refs/tags/v(\d+\.\d+\.\d+\.\d+)(?:\^\{\})?$")


def parse_version(text: str) -> tuple[int, int, int, int]:
    """``"1.2.3.4"`` -> ``(1, 2, 3, 4)``; raises ValueError on anything else."""
    match = VERSION_RE.match(text.strip())
    if not match:
        raise ValueError(f"version must look like W.X.Y.Z, got {text!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def next_build(version: str) -> str:
    """The version after ``version`` when none is given: the last part + 1."""
    *head, last = parse_version(version)
    return ".".join(str(part) for part in (*head, last + 1))


def read_marker(path: Path = VERSION_FILE) -> str:
    """The ``__version__`` currently in the source tree."""
    match = MARKER_RE.search(path.read_text("utf-8"))
    if not match:
        raise ValueError(f"no __version__ line in {path}")
    return match.group(1)


def write_marker(version: str, path: Path = VERSION_FILE) -> None:
    """Rewrite the ``__version__`` line in place."""
    text = path.read_text("utf-8")
    path.write_text(MARKER_RE.sub(f'__version__ = "{version}"', text, count=1), "utf-8")


def tags_from_refs(lines: list[str]) -> set[str]:
    """Versions named by ``refs/tags/vW.X.Y.Z`` lines (ls-remote / show-ref output)."""
    found = set()
    for line in lines:
        ref = line.split()[-1] if line.split() else ""
        match = TAG_RE.match(ref)
        if match:
            found.add(match.group(1))
    return found


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in the repository root; on failure show git's own message."""
    result = subprocess.run(  # nosec B603 B607
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=120
    )
    if check and result.returncode != 0:
        sys.stderr.write(f"git {' '.join(args)} failed:\n{result.stderr}{result.stdout}")
        raise SystemExit(1)
    return result


def known_tags() -> set[str]:
    """Release versions already tagged, locally or on origin."""
    local = git("show-ref", "--tags", check=False).stdout.splitlines()
    remote = git("ls-remote", "--tags", "origin", check=False).stdout.splitlines()
    return tags_from_refs(local + remote)


def run_make(target: str) -> bool:
    """Run a make target, streaming its output; True on success."""
    result = subprocess.run(["make", target], cwd=REPO_ROOT, check=False)  # nosec B603 B607
    return result.returncode == 0


class Abort(Exception):
    """Stop the release with a message; nothing has been committed."""


def choose_version(requested: str | None, current: str, tags: set[str]) -> str:
    """Validate (or derive) the new version against the marker and all tags.

    It must be newer than every tag and not older than the marker; the very
    first release may therefore be the marker's own version.
    """
    try:
        newest_tag = max(tags, key=parse_version) if tags else None
        if requested is None:
            requested = next_build(max(current, newest_tag, key=parse_version)) if tags else current
        if parse_version(requested) < parse_version(current):
            raise ValueError(f"{requested} is older than the source version {current}")
        if newest_tag and parse_version(requested) <= parse_version(newest_tag):
            raise ValueError(f"{requested} is not newer than the last release {newest_tag}")
    except ValueError as exc:
        raise Abort(str(exc)) from exc
    return requested


def confirm(tag: str, assume_yes: bool) -> None:
    """The y/N prompt; only an explicit y/yes proceeds."""
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise Abort("no terminal to confirm on; pass YES=1")
    if input(f"Release {tag}? [y/N] ").strip().lower() not in ("y", "yes"):
        raise Abort("aborted at the prompt")


def release(args: argparse.Namespace) -> None:
    """The release flow proper; raises Abort to stop."""
    current = read_marker()
    tags = known_tags()
    version = choose_version(args.version, current, tags)
    if git("status", "--porcelain").stdout.strip():
        raise Abort("working tree is not clean; commit or stash first")

    tag = f"v{version}"
    message = args.message or f"Release {tag}"
    print(f"Current version : {current}")
    print(f"Highest tag     : {max(tags, key=parse_version) if tags else '(none)'}")
    print(f"New version     : {version}  (tag {tag})")
    print(f"Tag message     : {message}")
    if args.dry_run:
        print("Dry run: nothing changed.")
        return
    confirm(tag, args.yes)

    previous_digest = DIGEST_FILE.read_text("utf-8") if DIGEST_FILE.exists() else None
    write_marker(version)
    record_digest()
    if not args.skip_checks and not (run_make("lint") and run_make("test")):
        write_marker(current)
        if previous_digest is not None:
            DIGEST_FILE.write_text(previous_digest, "utf-8")
        raise Abort("lint/test failed; version marker restored")

    git("commit", "-a", "-m", message)
    git("tag", "-a", tag, "-m", message)
    git("push")
    git("push", "origin", tag)
    print(f"Released {tag}. The tag push triggers the build-and-release workflow.")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the release; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="W.X.Y.Z (default: last part + 1)")
    parser.add_argument("--message", help="tag message")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the y/N prompt")
    parser.add_argument("--skip-checks", action="store_true", help="skip make lint/test")
    try:
        release(parser.parse_args(argv))
    except Abort as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
