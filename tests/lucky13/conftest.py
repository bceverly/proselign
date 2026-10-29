# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""MITRE's "Lucky 13" unforgivable vulnerabilities, checked on every push.

Steve Christey (MITRE), "Unforgivable Vulnerabilities", Black Hat USA 2007:
thirteen weakness classes so well documented and so cheap to find that
shipping one is unforgivable. Each test here names its number and CWE.
CI runs this package as its own step (``pytest -m lucky13``) so a
regression is named, not buried in the suite; ``make security`` runs it too.

    1 buffer overflow (CWE-120)       8 auth bypass: authenticated=1 (CWE-472)
    2 XSS (CWE-79)                    9 grow-your-own crypto (CWE-327)
    3 directory traversal (CWE-23)   10 privilege escalation via Help (CWE-271)
    4 remote file inclusion (CWE-98) 11 symlink following (CWE-61)
    5 SQL injection (CWE-89)         12 hard-coded / default password (CWE-259)
    6 world-writable files (CWE-276) 13 integer overflow (CWE-190)
    7 direct request (CWE-425)

Proselign is a local command-line filter, not a network service, so several
classes are checked as source-hygiene invariants (the dangerous construct
must not exist) rather than as attacks against a running server.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "proselign"


def pytest_collection_modifyitems(items):
    for item in items:
        if "lucky13" in item.nodeid.split("::")[0]:
            item.add_marker(pytest.mark.lucky13)


def source_files() -> list[Path]:
    """Every shipped Python module."""
    return sorted(SRC.rglob("*.py"))


def parsed_sources() -> list[tuple[Path, ast.Module]]:
    """(path, AST) for every shipped module."""
    return [(path, ast.parse(path.read_text("utf-8"), str(path))) for path in source_files()]


def dotted(node: ast.AST) -> str:
    """``a.b.c`` for a Name/Attribute chain, else ``""``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    """Every call in ``tree`` with its dotted callee name."""
    return [(dotted(n.func), n) for n in ast.walk(tree) if isinstance(n, ast.Call)]


def imported_modules(tree: ast.Module) -> set[str]:
    """Top-level names of every module ``tree`` imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names
