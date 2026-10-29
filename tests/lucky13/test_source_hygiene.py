# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Lucky 13 classes enforced as invariants over the shipped source."""

from __future__ import annotations

import ast
import re

from .conftest import REPO, calls, dotted, imported_modules, parsed_sources

WEBSITE = REPO / "website"


def test_02_xss_cwe79_no_html_sinks():
    """#2 XSS: the CLI emits no HTML, and the website's scripts have no raw
    HTML sinks or string evaluation that untrusted data could reach."""
    for path, tree in parsed_sources():
        assert "html" not in {m.split(".")[0] for m in imported_modules(tree)}, path
    sinks = re.compile(
        r"\b(innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval)\b|new Function"
    )
    for script in WEBSITE.rglob("*.js"):
        assert not sinks.search(script.read_text("utf-8")), f"HTML/eval sink in {script}"
    for page in WEBSITE.rglob("*.html"):
        text = page.read_text("utf-8")
        assert not re.search(r"\son[a-z]+\s*=", text), f"inline event handler in {page}"
        assert "javascript:" not in text, page


DYNAMIC_CODE = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "importlib.import_module",
    "runpy.run_path",
    "runpy.run_module",
}
UNSAFE_LOADERS = {"pickle", "marshal", "shelve", "dill", "cloudpickle", "yaml", "joblib"}


def test_04_remote_file_inclusion_cwe98_no_dynamic_code():
    """#4 RFI: nothing is ever loaded as code from data -- no eval/exec, no
    dynamic imports, no pickle-family deserializers (voice archives are
    JSON-only and validated)."""
    for path, tree in parsed_sources():
        for name, _ in calls(tree):
            assert name not in DYNAMIC_CODE, f"{name}() in {path}"
        bad = {m.split(".")[0] for m in imported_modules(tree)} & UNSAFE_LOADERS
        assert not bad, f"{bad} imported in {path}"


def test_05_sql_injection_cwe89_parameterized_only():
    """#5 SQL injection: every ``.execute``/``.executemany`` passes a literal
    query string. Vacuous until the SQLite example store lands (M3), then a
    gate: f-strings, %-formatting and concatenation are rejected."""
    for path, tree in parsed_sources():
        for name, node in calls(tree):
            if name.rsplit(".", 1)[-1] in {"execute", "executemany", "executescript"}:
                query = node.args[0] if node.args else None
                assert isinstance(query, ast.Constant) and isinstance(
                    query.value, str
                ), f"non-literal SQL at {path}:{node.lineno}"


def test_09_grow_your_own_crypto_cwe327():
    """#9 Home-grown crypto: hashing uses SHA-2 from hashlib, randomness for
    anything security-relevant comes from ``secrets``; no weak digests, no
    ``random`` module, no hand-rolled XOR ciphers."""
    for path, tree in parsed_sources():
        assert "random" not in imported_modules(tree), f"random imported in {path}"
        for name, _ in calls(tree):
            assert name not in {"hashlib.md5", "hashlib.sha1"}, f"{name} in {path}"
            if name == "hashlib.new":
                raise AssertionError(f"hashlib.new() in {path}: name the algorithm explicitly")
        for node in ast.walk(tree):
            assert not isinstance(node, ast.BinOp) or not isinstance(
                node.op, ast.BitXor
            ), f"XOR at {path}:{node.lineno}; use a vetted primitive"


SHELL_OUT = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execvp",
    "os.spawnv",
    "pty.spawn",
    "webbrowser.open",
    "pydoc.pager",
}


def test_10_privilege_escalation_via_help_cwe271():
    """#10 Escalation via Help: help and man output never launch a pager,
    browser or shell; the program runs no external commands at all."""
    for path, tree in parsed_sources():
        assert "subprocess" not in imported_modules(tree), f"subprocess in {path}"
        for name, _ in calls(tree):
            assert name not in SHELL_OUT, f"{name}() in {path}"


SECRET_NAME = re.compile(r"(passw(or)?d|secret|token|api_?key|credential)", re.IGNORECASE)


def test_12_hard_coded_password_cwe259():
    """#12 Default credentials: no credential-like name is ever bound to a
    non-empty string literal, and no Authorization header is hard-coded."""
    for path, tree in parsed_sources():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                getattr(node, "value", None), ast.Constant
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = dotted(target)
                    value = node.value.value
                    if SECRET_NAME.search(name) and isinstance(value, str) and value:
                        raise AssertionError(f"hard-coded {name} at {path}:{node.lineno}")
        assert "Authorization" not in path.read_text("utf-8"), path
