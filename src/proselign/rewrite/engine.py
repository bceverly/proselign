# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The rewrite pipeline: segment, rewrite prose, validate, repair, reassemble.

Verbatim segments (fenced code, tables, HTML blocks, front matter) never
reach the model at all, which is the strongest preservation guarantee there
is. Prose is grouped into token-bounded chunks at paragraph boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from proselign.errors import UsageError, ValidationError
from proselign.inference.base import Backend
from proselign.rewrite import protect
from proselign.rewrite.prompt import build_messages
from proselign.voice.profile import Voice

MAX_CHUNK_TOKENS = 1_000_000

_FENCE_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})")


@dataclass
class Segment:
    """A run of the input that is either rewritten (prose) or copied (verbatim)."""

    text: str
    prose: bool


@dataclass
class Options:
    """Knobs for one rewrite run."""

    strength: float = 0.5
    strict: bool = False
    max_chunk_tokens: int = 1500
    max_repairs: int = 2
    text_format: str = "markdown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise UsageError("--strength must be between 0.0 and 1.0")
        if not 64 <= self.max_chunk_tokens <= MAX_CHUNK_TOKENS:
            raise UsageError(f"--max-chunk-tokens must be between 64 and {MAX_CHUNK_TOKENS}")
        if self.text_format not in ("text", "markdown"):
            raise UsageError("--format must be 'text' or 'markdown'")


@dataclass
class Report:
    """Diagnostics gathered during a run (written to stderr, never stdout)."""

    chunks: int = 0
    repaired: int = 0
    fallbacks: int = 0
    warnings: list[str] = field(default_factory=list)


def _is_verbatim_block(block: str) -> bool:
    """Markdown blocks that must be copied byte for byte."""
    stripped = block.lstrip()
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if stripped.startswith("<") and not stripped.startswith("<http"):
        return True  # raw HTML block
    if lines and all(ln.lstrip().startswith("|") for ln in lines):
        return True  # pipe table
    if lines and all(ln.startswith(("    ", "\t")) for ln in lines):
        return True  # indented code
    return False


def segment(text: str, text_format: str = "markdown") -> list[Segment]:
    """Split input into prose and verbatim segments, losslessly.

    ``"".join(s.text for s in segment(t))`` always equals ``t``.
    """
    if text_format == "text":
        return [Segment(text, True)] if text else []

    segments: list[Segment] = []
    lines = text.splitlines(keepends=True)
    i = 0
    if lines and lines[0].rstrip("\n") == "---":  # YAML front matter
        for j in range(1, len(lines)):
            if lines[j].rstrip("\n") in ("---", "..."):
                segments.append(Segment("".join(lines[: j + 1]), False))
                i = j + 1
                break

    block: list[str] = []

    def flush() -> None:
        if block:
            chunk = "".join(block)
            segments.append(Segment(chunk, not _is_verbatim_block(chunk)))
            block.clear()

    while i < len(lines):
        line = lines[i]
        fence = _FENCE_OPEN_RE.match(line)
        if fence:
            flush()
            marker = fence.group(2)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(marker):
                j += 1
            segments.append(Segment("".join(lines[i : j + 1]), False))
            i = j + 1
            continue
        if not line.strip():
            flush()
            segments.append(Segment(line, False))
        else:
            block.append(line)
        i += 1
    flush()
    return segments


def _split_edges(text: str) -> tuple[str, str, str]:
    """Separate leading/trailing whitespace so the model never alters it."""
    core = text.strip()
    if not core:
        return text, "", ""
    start = text.index(core)
    return text[:start], core, text[start + len(core) :]


class Rewriter:
    """Applies one voice to text through one backend."""

    def __init__(self, voice: Voice, backend: Backend, options: Options | None = None) -> None:
        self.voice = voice
        self.backend = backend
        self.options = options or Options()
        self.report = Report()

    def _temperature(self) -> float:
        return round(0.1 + 0.6 * self.options.strength, 2)

    def _rewrite_chunk(self, source: str) -> str:
        lead, core, trail = _split_edges(source)
        if not core:
            return source
        self.report.chunks += 1
        feedback: list[str] | None = None
        for attempt in range(self.options.max_repairs + 1):
            messages = build_messages(self.voice, core, self.options.strength, feedback)
            candidate = self.backend.rewrite(messages, self._temperature()).strip()
            problems = protect.missing(core, candidate)
            if not candidate:
                problems = ["the rewrite was empty"]
            if not problems:
                if attempt:
                    self.report.repaired += 1
                return lead + candidate + trail
            feedback = problems

        detail = "; ".join(feedback or [])
        if self.options.strict:
            raise ValidationError(f"rewrite failed fidelity checks: {detail}")
        self.report.fallbacks += 1
        self.report.warnings.append(f"kept original text for one chunk: {detail}")
        return source

    def _chunks(self, segments: list[Segment]) -> list[Segment]:
        """Merge adjacent prose (and the blank lines between) up to the token budget."""
        merged: list[Segment] = []
        budget = self.options.max_chunk_tokens
        for seg in segments:
            prev = merged[-1] if merged else None
            joinable = seg.prose or not seg.text.strip()  # prose or a blank separator
            if (
                prev is not None
                and prev.prose
                and joinable
                and self.backend.count_tokens(prev.text + seg.text) <= budget
            ):
                prev.text += seg.text
            else:
                merged.append(Segment(seg.text, seg.prose))
        return merged

    def rewrite(self, text: str) -> str:
        """Rewrite ``text``, returning the full reassembled document."""
        out = []
        for seg in self._chunks(segment(text, self.options.text_format)):
            out.append(self._rewrite_chunk(seg.text) if seg.prose else seg.text)
        return "".join(out)
