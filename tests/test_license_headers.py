# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The license-header gate: year ranges, detection and repair."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_license_headers as clh  # noqa: E402  pylint: disable=wrong-import-position,import-error

Y2026 = dt.date(2026, 10, 29)
Y2028 = dt.date(2028, 1, 2)


def test_years():
    assert clh.years(Y2026) == "2026"
    assert clh.years(Y2028) == "2026-2028"


def test_good_header_passes_and_goes_stale_next_year():
    text = "# SPDX-License-Identifier: BSD-3-Clause\n# Copyright (c) 2026 Bryan C. Everly\nx = 1\n"
    assert not clh.problems(text, Y2026)
    assert "stale" in clh.problems(text, Y2028)[0]


@pytest.mark.parametrize(
    "name,original,first_lines",
    [
        ("a.py", "x = 1\n", ["# SPDX-License-Identifier: BSD-3-Clause"]),
        (
            "run.sh",
            "#!/bin/sh\necho hi\n",
            ["#!/bin/sh", "# SPDX-License-Identifier: BSD-3-Clause"],
        ),
        ("s.css", "body {}\n", ["/* SPDX-License-Identifier: BSD-3-Clause */"]),
        ("p.html", "<!doctype html>\n<p>x</p>\n", ["<!doctype html>", "<!-- SPDX-"]),
        ("t.js", "go();\n", ["// SPDX-License-Identifier: BSD-3-Clause"]),
        ("x.1", ".TH X 1\n", ['.\\" SPDX-License-Identifier: BSD-3-Clause']),
    ],
)
def test_fix_inserts_in_the_right_syntax_and_is_idempotent(tmp_path, name, original, first_lines):
    style = clh.style_for(tmp_path / name)
    fixed = clh.fix(original, style, Y2028)
    for expected, actual in zip(first_lines, fixed.splitlines()):
        assert actual.startswith(expected)
    assert "Copyright (c) 2026-2028 Bryan C. Everly" in fixed
    assert not clh.problems(fixed, Y2028)
    assert clh.fix(fixed, style, Y2028) == fixed


def test_fix_repairs_name_and_years_in_place():
    text = "# SPDX-License-Identifier: BSD-3-Clause\n# Copyright (c) 2026 Bryan Everly\nx = 1\n"
    fixed = clh.fix(text, clh.HASH, Y2028)
    assert fixed.splitlines()[1] == "# Copyright (c) 2026-2028 Bryan C. Everly"
    css = "/* SPDX-License-Identifier: BSD-3-Clause */\nbody {}\n"
    assert clh.fix(css, (".", ""), Y2026).count("SPDX") == 1


def test_style_selection(tmp_path):
    assert clh.style_for(tmp_path / "Makefile") == clh.HASH
    assert clh.style_for(tmp_path / "requirements-dev.txt") is None
    assert clh.style_for(tmp_path / "README.md") is None


def test_repository_is_clean():
    today = dt.date.today()
    bad = [p for p in clh.source_files() if clh.problems(p.read_text("utf-8"), today)]
    assert not bad
