# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The progress line: when it appears, what it says, and how it cleans up."""

from __future__ import annotations

import io

import pytest

from proselign.cli import main
from proselign.errors import EXIT_OK
from proselign.progress import NoProgress, Progress, _clock, wanted


class Terminal(io.StringIO):
    """A StringIO that claims to be an interactive UTF-8 terminal."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class Ticker:
    """A clock that only moves when told to."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_wanted_only_on_real_terminals(monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    assert wanted(Terminal())
    assert not wanted(io.StringIO())
    assert not wanted(object())

    class Closed:  # pylint: disable=too-few-public-methods
        def isatty(self):
            raise ValueError("I/O operation on closed file")

    assert not wanted(Closed())
    monkeypatch.setenv("TERM", "dumb")
    assert not wanted(Terminal())


def test_line_counts_percent_elapsed_and_estimate():
    clock = Ticker()
    progress = Progress(Terminal(), clock)
    progress.total, progress.noun = 10, "paragraphs"
    assert "0/10 paragraphs    0%  0:00 elapsed, estimating time left" in progress.line(120)
    clock.now += 40
    progress.advance(4)
    line = progress.line(120)
    assert "4/10 paragraphs   40%  0:40 elapsed, about 1:00 left" in line
    assert "█" * 12 + "░" * 18 in line
    progress.advance(99)
    assert "10/10 paragraphs  100%" in progress.line(120)
    assert "finishing" in progress.line(120)


def test_line_fits_narrow_terminals_and_plain_encodings():
    progress = Progress(io.StringIO(), Ticker())
    progress.total, progress.done = 4, 2
    assert "[" + "#" * 15 + "-" * 15 + "]" in progress.line(200)
    assert "[" not in progress.line(40)
    assert len(progress.line(20)) == 19
    progress.total = 0
    assert "100%" in progress.line(200)


def test_clock_format():
    assert (_clock(5), _clock(65), _clock(3725)) == ("0:05", "1:05", "1:02:05")


def test_animation_draws_and_erases(monkeypatch):
    monkeypatch.setattr("proselign.progress.REFRESH_SECONDS", 0.001)
    screen = Terminal()
    progress = Progress(screen)
    progress.begin(3, "chunks")
    progress.advance()
    progress.draw()
    progress.close()
    progress.close()  # closing twice is harmless
    assert "1/3 chunks" in screen.getvalue()
    assert screen.getvalue().endswith("\r\x1b[K")
    idle = Progress(Terminal())
    idle.begin(0, "chunks")  # nothing to do: no thread, nothing drawn
    idle.close()


def test_no_progress_is_silent():
    quiet = NoProgress()
    quiet.begin(5, "x")
    quiet.advance()
    quiet.close()


def _rewrite(trained, argv, monkeypatch):
    monkeypatch.setenv("TERM", "xterm")
    out, err = io.StringIO(), Terminal()
    code = main(
        ["--voice", "Bryan", "--backend", "mock", *argv],
        stdin=io.StringIO("One paragraph.\n\nAnother one.\n"),
        stdout=out,
        stderr=err,
        registry=trained,
    )
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    "argv,rc,shown",
    [([], "", True), (["--no-progress"], "", False), ([], "[rewrite]\nprogress = off\n", False)],
)
def test_cli_progress_switches(trained, monkeypatch, argv, rc, shown):
    if rc:
        path = trained.root.parent / "rc"
        path.write_text(rc, "utf-8")
        path.chmod(0o600)
        monkeypatch.setenv("PROSELIGN_CONFIG", str(path))
    code, out, err = _rewrite(trained, argv, monkeypatch)
    assert (code, out) == (EXIT_OK, "One paragraph.\n\nAnother one.\n")
    assert ("\r\x1b[K" in err) is shown
