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
from proselign.progress import NoProgress
from proselign.rewrite import protect, style
from proselign.rewrite.prompt import build_language_fix, build_messages
from proselign.rewrite.style import StyleRules
from proselign.voice.profile import Voice

MAX_CHUNK_TOKENS = 1_000_000

_FENCE_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})")
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]|$)")
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_UNIT_MARK_RE = re.compile(r"\[\[(\d+)\]\]")


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
    rules: StyleRules = field(default_factory=StyleRules)

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
    language_fixes: int = 0
    warnings: list[str] = field(default_factory=list)
    unit: str = "chunk"
    kept_problems: list[list[str]] = field(default_factory=list)
    style_problems: list[list[str]] = field(default_factory=list)

    def summary(self) -> list[str]:
        """At most two lines: what was kept as written, and why; which style
        rules were not fully met.  ``warnings`` holds every detail."""
        lines = []
        for what, groups in (
            ("kept the original text of", self.kept_problems),
            ("style rules not fully met in", self.style_problems),
        ):
            if groups:
                counts: dict[str, int] = {}
                for problems in groups:
                    for category in dict.fromkeys(_category(p) for p in problems):
                        counts[category] = counts.get(category, 0) + 1
                reasons = ", ".join(f"{name} ({count})" for name, count in counts.items())
                plural = self.unit if len(groups) == 1 else f"{self.unit}s"
                lines.append(f"{what} {len(groups)} {plural}: {reasons}")
        return lines


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
    if len(lines) == 2 and _SETEXT_UNDERLINE_RE.match(lines[1]):
        return True  # setext heading (Title / =====)
    return False


def _front_matter_lines(lines: list[str]) -> int:
    """How many leading lines form a YAML front-matter block (0 if none)."""
    if not lines or lines[0].rstrip("\n") != "---":
        return 0
    for j in range(1, len(lines)):
        if lines[j].rstrip("\n") in ("---", "..."):
            return j + 1
    return 0


def _fence_end(lines: list[str], start: int, marker: str) -> int:
    """Index just past the line closing the code fence opened at ``start``."""
    j = start + 1
    while j < len(lines) and not lines[j].lstrip().startswith(marker):
        j += 1
    return j + 1


def segment(text: str, text_format: str = "markdown") -> list[Segment]:
    """Split input into prose and verbatim segments, losslessly.

    ``"".join(s.text for s in segment(t))`` always equals ``t``.
    """
    if text_format == "text":
        return [Segment(text, True)] if text else []

    lines = text.splitlines(keepends=True)
    i = _front_matter_lines(lines)
    segments = [Segment("".join(lines[:i]), False)] if i else []
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
            end = _fence_end(lines, i, fence.group(2))
            segments.append(Segment("".join(lines[i:end]), False))
            i = end
            continue
        if _ATX_HEADING_RE.match(line):
            flush()
            segments.append(Segment(line, False))  # headings stay as written
        elif line.strip():
            block.append(line)
        else:
            flush()
            segments.append(Segment(line, False))
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
        self.progress = NoProgress()  # the CLI swaps in a visible one

    def _temperature(self) -> float:
        return round(0.1 + 0.6 * self.options.strength, 2)

    def _rewrite_chunk(self, source: str) -> str:
        lead, core, trail = _split_edges(source)
        if not core:
            return source
        self.report.chunks += 1
        result, problems = self._attempt(core)
        if result is not None:
            return lead + result + trail
        detail = _summarize(problems)
        if self.options.strict:
            raise ValidationError(f"rewrite failed fidelity checks: {detail}")
        self.report.fallbacks += 1
        self.report.kept_problems.append(problems)
        self.report.warnings.append(f"kept original text for one {self.report.unit}: {detail}")
        return source

    def _attempt(self, core: str) -> tuple[str | None, list[str]]:
        """The best faithful rewrite of ``core``, or None and the last problems."""
        rules = self.options.rules
        feedback: list[str] | None = None
        best: str | None = None  # faithful, but still breaking a style rule
        for attempt in range(self.options.max_repairs + 1):
            messages = build_messages(self.voice, core, self.options.strength, feedback, rules)
            reply = self.backend.rewrite(messages, self._temperature()).strip()
            candidate = protect.repair_quotations(core, style.clean(core, reply, rules))
            problems = protect.missing(core, candidate)
            drift = protect.foreign_script(core, candidate)
            if drift and not problems:
                candidate, drift = self._fix_language(core, candidate)
            problems += drift
            if not candidate:
                problems = ["the rewrite was empty"]
            if not problems:
                problems = style.violations(core, candidate, rules)
                if not problems:
                    if attempt:
                        self.report.repaired += 1
                    return candidate, []
                best = style.punctuate(candidate, rules)
            feedback = problems

        if best is not None and not protect.missing(core, best):
            remaining = style.violations(core, best, rules)
            if remaining:
                self.report.style_problems.append(remaining)
                self.report.warnings.append(f"style rules not fully met: {_summarize(remaining)}")
            return best, []
        return None, feedback or []

    def _fix_language(self, core: str, candidate: str) -> tuple[str, list[str]]:
        """Translate the stray fragments of a rewrite that drifted into another
        language, keeping the rest of it; returns the text and any drift left."""
        reply = self.backend.rewrite(build_language_fix(candidate), 0.0).strip()
        fixed = protect.repair_quotations(core, style.clean(core, reply, self.options.rules))
        drift = protect.foreign_script(core, fixed)
        if fixed and not drift and not protect.missing(core, fixed):
            self.report.language_fixes += 1
            return fixed, []
        return candidate, protect.foreign_script(core, candidate)

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

    def rewrite_units(self, units: list[str]) -> list[str]:
        """Rewrite independent paragraphs (of a .docx), returning one result each.

        Paragraphs travel in token-bounded batches, each prefixed ``[[n]]``;
        a batch whose markers come back damaged is redone one paragraph at
        a time.  Fidelity failures keep the original paragraph.
        """
        results = list(units)
        self.report.unit = "paragraph"
        self.progress.begin(len(units), "paragraphs")
        for batch in self._unit_batches(units):
            parts = None
            if len(batch) > 1:
                text = "\n\n".join(f"[[{n}]] {units[i]}" for n, i in enumerate(batch, 1))
                rewritten, _ = self._attempt(text)
                parts = None if rewritten is None else _split_units(rewritten, len(batch))
                self.report.chunks += parts is not None
            if parts is None:
                parts = []
                for i in batch:
                    parts.append(self._rewrite_chunk(units[i]))
                    self.progress.advance()
            else:
                self.progress.advance(len(batch))
            for i, part in zip(batch, parts):
                results[i] = part
        return results

    def _unit_batches(self, units: list[str]) -> list[list[int]]:
        batches: list[list[int]] = []
        size = 0
        for i, unit in enumerate(units):
            cost = self.backend.count_tokens(unit) + 4
            if not batches or size + cost > self.options.max_chunk_tokens:
                batches.append([])
                size = 0
            batches[-1].append(i)
            size += cost
        return batches

    def rewrite(self, text: str) -> str:
        """Rewrite ``text``, returning the full reassembled document."""
        out = []
        chunks = self._chunks(segment(text, self.options.text_format))
        self.progress.begin(sum(seg.prose for seg in chunks), "chunks")
        for seg in chunks:
            if seg.prose:
                out.append(self._rewrite_chunk(seg.text))
                self.progress.advance()
            else:
                out.append(seg.text)
        return "".join(out)


def _split_units(text: str, count: int) -> list[str] | None:
    """The paragraphs of a rewritten batch, or None unless markers 1..count
    each appear once, in order."""
    marks = list(_UNIT_MARK_RE.finditer(text))
    if [int(m.group(1)) for m in marks] != list(range(1, count + 1)):
        return None
    if text[: marks[0].start()].strip():
        return None
    ends = [m.start() for m in marks[1:]] + [len(text)]
    return [text[m.end() : end].strip() for m, end in zip(marks, ends)]


def _summarize(problems: list[str], limit: int = 5) -> str:
    """The first few problems, and how many more there were."""
    shown = "; ".join(problems[:limit])
    more = len(problems) - limit
    return f"{shown}; and {more} more" if more > 0 else shown


_CATEGORIES = (
    ("rewrite these passive", "passive voice"),
    ("replace stock AI", "AI phrasing"),
    ("use the first person plural", "singular pronouns"),
    ("rephrase to use no", "forbidden punctuation"),
    ("quotation ", "quotation reworded"),
    ("number ", "number changed"),
    ("markup ", "formatting lost"),
    ("the rewrite switched to", "language drift"),
    ("the rewrite was empty", "empty reply"),
)


def _category(problem: str) -> str:
    """A short name for the kind of problem ``problem`` describes."""
    for prefix, name in _CATEGORIES:
        if problem.startswith(prefix):
            return name
    return problem.split(" ", 1)[0] + " changed"  # url, email, citation, inline_code
