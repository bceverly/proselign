# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""An animated progress line on stderr for long rewrites.

Shown only when stderr is an interactive terminal, so pipelines, scripts
and redirected output never see it.  A background thread redraws the line
a few times a second, which keeps the spinner moving while a model call
blocks.  The estimate is the average time per finished item so far.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from typing import TextIO

_FRAMES_UNICODE = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
_FRAMES_ASCII = "|/-\\"
REFRESH_SECONDS = 0.2


class NoProgress:
    """The silent stand-in used when no progress line is shown."""

    def begin(self, total: int, noun: str) -> None:
        """Ignore the start of a run."""

    def advance(self, count: int = 1) -> None:
        """Ignore finished items."""

    def close(self) -> None:
        """Nothing to erase."""


def wanted(stream: TextIO) -> bool:
    """Whether a progress line belongs on ``stream``."""
    try:
        interactive = stream.isatty()
    except (AttributeError, ValueError):
        return False
    return interactive and os.environ.get("TERM") != "dumb"


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Progress:
    """Tracks ``done`` of ``total`` items and draws them on ``stream``."""

    def __init__(self, stream: TextIO, clock: Callable[[], float] = time.monotonic) -> None:
        self.stream = stream
        self.clock = clock
        self.total = 0
        self.done = 0
        self.noun = "items"
        self.started = clock()
        self.frame = 0
        unicode = "utf" in (getattr(stream, "encoding", "") or "").lower()
        # (spinner frames, filled cell, empty cell)
        self.glyphs = (
            (_FRAMES_UNICODE, "\u2588", "\u2591") if unicode else (_FRAMES_ASCII, "#", "-")
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def begin(self, total: int, noun: str) -> None:
        """Start (or restart) counting toward ``total`` ``noun``."""
        self.total, self.done, self.noun = max(total, 0), 0, noun
        self.started = self.clock()
        if self._thread is None and self.total:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def advance(self, count: int = 1) -> None:
        """Record ``count`` more finished items."""
        self.done = min(self.total, self.done + count)

    def line(self, width: int) -> str:
        """The progress line, fitted to ``width`` columns."""
        frames, fill, empty = self.glyphs
        spinner = frames[self.frame % len(frames)]
        elapsed = self.clock() - self.started
        if self.done and self.done < self.total:
            remaining = elapsed / self.done * (self.total - self.done)
            eta = f"about {_clock(remaining)} left"
        elif self.done >= self.total:
            eta = "finishing"
        else:
            eta = "estimating time left"
        share = self.done / self.total if self.total else 1.0
        text = (
            f" {self.done}/{self.total} {self.noun}  {share:4.0%}  "
            f"{_clock(elapsed)} elapsed, {eta}"
        )
        bar_width = max(0, min(30, width - len(text) - 5))
        filled = round(bar_width * share)
        meter = f" [{fill * filled}{empty * (bar_width - filled)}]" if bar_width else ""
        return f"{spinner}{meter}{text}"[: max(width - 1, 0)]

    def draw(self) -> None:
        """Redraw the line in place."""
        width = shutil.get_terminal_size((80, 24)).columns
        self.stream.write("\r" + self.line(width) + "\x1b[K")
        self.stream.flush()
        self.frame += 1

    def close(self) -> None:
        """Stop animating and erase the line, leaving the cursor at column 0."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self.stream.write("\r\x1b[K")
            self.stream.flush()

    def _animate(self) -> None:
        while not self._stop.is_set():
            self.draw()
            self._stop.wait(REFRESH_SECONDS)
