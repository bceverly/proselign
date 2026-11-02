# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The rewrite pipeline: segment, rewrite prose, validate, repair, reassemble.

Verbatim segments (fenced code, tables, HTML blocks, front matter) never
reach the model at all, which is the strongest preservation guarantee there
is. Prose is grouped into small token-bounded chunks at paragraph boundaries,
and several chunks are rewritten at once (``Options.parallel``): a model
server that batches requests gets through them far faster than one by one.
With ``Options.review``, every rewrite is then checked sentence by sentence
(:mod:`proselign.rewrite.review`) and only the changes that pass are kept.
"""

from __future__ import annotations

import copy
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from proselign.errors import UsageError, ValidationError
from proselign.inference.base import Backend
from proselign.progress import NoProgress
from proselign.rewrite import protect, review, scoring, style
from proselign.rewrite.exemplars import Library
from proselign.rewrite.prompt import build_language_fix, build_messages, build_tuned_messages
from proselign.rewrite.style import StyleRules
from proselign.voice.profile import Voice

MAX_CHUNK_TOKENS = 1_000_000
MAX_CANDIDATES = 5
MAX_EXAMPLES = 8
MAX_PARALLEL = 16
# About a paragraph or two: a chunk this small fits a 4,096-token context
# with the instructions and the reply, and a failed one costs little to redo.
CHUNK_TOKENS = 400
# Parts of the voice that can be left out of a rewrite, to measure what each
# one contributes (``proselign eval --ablate``): the instruction groups, the
# tone line, the author's example passages, and the feedback that steers a
# first attempt back toward the profile.
PROMPT_PARTS = (*style.DIRECTIVE_GROUPS, "tone", "examples", "steering")

_FENCE_OPEN_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})")
_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]|$)")
_SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_UNIT_MARK_RE = re.compile(r"\[\[(\d+)\]\]")
# Pieces of a failed chunk shorter than this are kept: alone, a model has
# nothing to restyle in them and fills the reply with invented text.
MIN_PIECE_WORDS = 6
_BLANK_LINES_RE = re.compile(r"(\n[ \t]*\n\s*)")
_HEADER_LINE_RE = re.compile(
    r"^(?:to|cc|bcc|from|reply-to|date|subject|sent|re|fwd?):[ \t]", re.IGNORECASE
)
_LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+")
# A line that ends without punctuation and opens a paragraph or a list is a
# heading ("5. Intercompany paper and where the IP sits"); it stays as written.
MAX_HEADING_WORDS = 20
_UNPUNCTUATED_END_RE = re.compile(r"[\w%]\Z")
# A list item after a blank line; group 1 is its marker.
_SPACED_ITEM_RE = re.compile(r"\n[ \t]*\n\s*([ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+)")
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+")
_NUMBERED_RE = re.compile(r"^[ \t]*\d{1,3}[.)][ \t]+")


@dataclass
class Segment:
    """A run of the input that is either rewritten (prose) or copied (verbatim)."""

    text: str
    prose: bool


@dataclass
class Options:  # pylint: disable=too-many-instance-attributes  # one per knob
    """Knobs for one rewrite run."""

    strength: float = 0.5
    strict: bool = False
    max_chunk_tokens: int = CHUNK_TOKENS
    max_repairs: int = 1
    text_format: str = "markdown"
    rules: StyleRules = field(default_factory=StyleRules)
    candidates: int = 1  # rewrites generated per attempt; the closest to the voice wins
    examples: int = 3  # passages of the author's writing shown per chunk
    omit: frozenset[str] = frozenset()  # parts of the voice left out (PROMPT_PARTS)
    # Reject rewrites that use I/we more, or you differently, than the source;
    # off only for ``proselign eval``, whose neutral texts lost them on purpose.
    keep_person: bool = True
    # Reject rewrites that change a negation or modal verb, or drop or add a
    # sentence; off only for ``proselign eval``, which measures style alone.
    keep_meaning: bool = True
    parallel: int = 4  # chunks rewritten at once
    # The model was fine-tuned on this voice (``proselign tune``): it gets the
    # short tuned prompt, without profile, examples or steering.
    tuned: bool = False
    # Check each changed sentence on its own and keep the original of any
    # that fails (the CLI turns this on; see proselign.rewrite.review).
    review: bool = False
    # Change only the sentences that need work (review.needs_work): the rest
    # are left exactly as written.  Needs ``review``.
    targeted: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise UsageError("--strength must be between 0.0 and 1.0")
        if not 64 <= self.max_chunk_tokens <= MAX_CHUNK_TOKENS:
            raise UsageError(f"--max-chunk-tokens must be between 64 and {MAX_CHUNK_TOKENS}")
        if self.text_format not in ("text", "markdown"):
            raise UsageError("--format must be 'text' or 'markdown'")
        if not 1 <= self.candidates <= MAX_CANDIDATES:
            raise UsageError(f"--candidates must be between 1 and {MAX_CANDIDATES}")
        if not 0 <= self.examples <= MAX_EXAMPLES:
            raise UsageError(f"--examples must be between 0 and {MAX_EXAMPLES}")
        if not 1 <= self.parallel <= MAX_PARALLEL:
            raise UsageError(f"--parallel must be between 1 and {MAX_PARALLEL}")
        unknown = set(self.omit) - set(PROMPT_PARTS)
        if unknown:
            raise UsageError(f"unknown part(s) of a voice: {', '.join(sorted(unknown))}")


@dataclass
class Report:
    """Diagnostics gathered during a run (written to stderr, never stdout)."""

    chunks: int = 0
    repaired: int = 0
    fallbacks: int = 0
    language_fixes: int = 0
    warnings: list[str] = field(default_factory=list)
    unit: str = "passage"
    kept_problems: list[list[str]] = field(default_factory=list)
    style_problems: list[list[str]] = field(default_factory=list)
    reverted: list[list[str]] = field(default_factory=list)  # sentences kept by the review
    untouched: int = 0  # chunks with nothing that needed work

    def merge(self, other: Report) -> None:
        """Add ``other``'s counts and findings (a parallel worker's) to these."""
        self.chunks += other.chunks
        self.repaired += other.repaired
        self.fallbacks += other.fallbacks
        self.language_fixes += other.language_fixes
        self.warnings += other.warnings
        self.kept_problems += other.kept_problems
        self.style_problems += other.style_problems
        self.reverted += other.reverted
        self.untouched += other.untouched

    def summary(self) -> list[str]:
        """At most three lines: what was kept as written, and why; which
        sentences kept their original wording, and why; which style rules
        were not fully met.  ``warnings`` holds every detail."""
        lines = []
        for what, groups, unit in (
            ("kept the original text of", self.kept_problems, self.unit),
            ("kept the original wording of", self.reverted, "sentence"),
            ("style rules not fully met in", self.style_problems, self.unit),
        ):
            if groups:
                counts: dict[str, int] = {}
                for problems in groups:
                    for category in dict.fromkeys(_category(p) for p in problems):
                        counts[category] = counts.get(category, 0) + 1
                reasons = ", ".join(f"{name} ({count})" for name, count in counts.items())
                plural = unit if len(groups) == 1 else f"{unit}s"
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
    if lines and all(_HEADER_LINE_RE.match(ln) for ln in lines):
        return True  # e-mail headers (To:, Cc:, Subject:)
    return False


def _is_heading_line(lines: list[str], i: int) -> bool:
    """Whether ``lines[i]``, the first line of a block, is a plain-text heading:
    short, unpunctuated, not a bullet, and alone or followed by a list of
    another kind (a numbered heading over bullets)."""
    line = lines[i].strip()
    if not line or not _UNPUNCTUATED_END_RE.search(line) or _BULLET_RE.match(line):
        return False
    if len(line.split()) > MAX_HEADING_WORDS:
        return False
    following = lines[i + 1] if i + 1 < len(lines) else ""
    if not following.strip():
        return True
    numbered = _NUMBERED_RE.match(line)
    return bool(_BULLET_RE.match(following) if numbered else _LIST_ITEM_RE.match(following))


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
        if not block and _is_heading_line(lines, i):
            segments.append(Segment(line, False))
        elif line.strip() and not _ATX_HEADING_RE.match(line):
            block.append(line)
        else:  # blank lines, and headings, which stay as written
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

    def __init__(
        self,
        voice: Voice,
        backend: Backend,
        options: Options | None = None,
        judge: Backend | None = None,
    ) -> None:
        self.voice = voice
        self.backend = backend
        self.options = options or Options()
        # The model asked whether each change keeps the meaning (with
        # ``Options.review``); without one, only the rule checks apply.
        self.judge = judge
        self._rejected: str | None = None  # the last attempt's failed rewrite
        self.report = Report()
        self.progress = NoProgress()  # the CLI swaps in a visible one
        self._library: Library | None = None

    def _temperature(self) -> float:
        return round(0.1 + 0.6 * self.options.strength, 2)

    def _rewrite_chunk(self, source: str) -> str:
        lead, core, trail = _split_edges(source)
        if not core:
            return source
        self.report.chunks += 1
        if self._fits(core):
            return source
        pieces = _pieces(core)
        result, problems = self._attempt(core, splittable=bool(pieces) and not self.options.review)
        if result is not None:
            return lead + result + trail
        if self.options.review and self._rejected:
            # The review keeps the sentences of it that pass, and the
            # original of the rest.
            return lead + self._rejected + trail
        if pieces:
            # One dropped number anywhere fails a whole chunk; retry it piece
            # by piece, so only the pieces that still fail are kept as written.
            return lead + "".join(self._rewrite_piece(p) for p in pieces) + trail
        detail = _summarize(problems)
        if self.options.strict:
            raise ValidationError(f"rewrite failed fidelity checks: {detail}")
        self.report.fallbacks += 1
        self.report.kept_problems.append(problems)
        self.report.warnings.append(f"kept original text for one {self.report.unit}: {detail}")
        return source

    def _attempt(self, core: str, splittable: bool = False) -> tuple[str | None, list[str]]:
        """The best faithful rewrite of ``core``, or None and the last problems.
        A ``splittable`` text that loses facts is not retried whole: its
        pieces, rewritten separately, keep them far more reliably."""
        rules = self.options.rules
        masked, spans = protect.mask(core)
        feedback: list[str] | None = None
        best: str | None = None  # faithful, but still breaking a style rule
        self._rejected = None
        prompt = self._prompt(core, masked)
        for attempt in range(self.options.max_repairs + 1):
            messages = prompt(feedback)
            results = [
                self._candidate(core, spans, messages, k) for k in range(self.options.candidates)
            ]
            faithful = [text for text, problems in results if not problems]
            if not faithful:
                self._rejected, feedback = results[0]
                if splittable:
                    break
                continue
            candidate = min(faithful, key=lambda text: scoring.distance(text, self.voice))
            problems = style.violations(core, candidate, rules)
            if not attempt and not self.options.tuned and "steering" not in self.options.omit:
                # One nudge toward the profile; it is approximate.
                problems += scoring.drift(candidate, self.voice, self.options.strength)
            if not problems:
                if attempt:
                    self.report.repaired += 1
                return candidate, []
            best = style.punctuate(candidate, rules)
            feedback = problems

        return self._settle(core, best, feedback)

    def _settle(
        self, core: str, best: str | None, feedback: list[str] | None
    ) -> tuple[str | None, list[str]]:
        """After the last attempt: the faithful rewrite that still breaks a
        style rule (reported), or None and the last problems."""
        if best is not None and not protect.missing(core, best):
            remaining = style.violations(core, best, self.options.rules)
            if remaining:
                self.report.style_problems.append(remaining)
                self.report.warnings.append(f"style rules not fully met: {_summarize(remaining)}")
            return best, []
        return None, feedback or []

    def _candidate(
        self, core: str, spans: list[str], messages, variant: int
    ) -> tuple[str, list[str]]:
        """One cleaned rewrite of ``core`` (sent with ``spans`` masked) and
        its fidelity problems."""
        rules = self.options.rules
        reply = self._reply(messages, variant).strip()
        candidate, problems = protect.unmask(style.clean(core, reply, rules), spans)
        candidate = _keep_paragraphs(core, candidate)
        candidate = protect.repair_quotations(core, candidate)
        problems += protect.missing(core, candidate) + protect.length_change(core, candidate)
        problems += protect.name_changes(core, candidate)
        if self.options.keep_meaning:
            problems += protect.claim_changes(core, candidate)
            problems += protect.sentence_changes(core, candidate)
        if self.options.keep_person:
            problems += protect.person_changes(core, candidate)
        drift = protect.foreign_script(core, candidate)
        if drift and not problems:
            candidate, drift = self._fix_language(core, candidate)
        problems += drift
        if not candidate:
            problems = ["the rewrite was empty"]
        return candidate, problems

    def _reply(self, messages, variant: int) -> str:
        """The backend's reply; each extra candidate gets its own sampling seed."""
        seed = getattr(self.backend, "seed", None)
        if not variant or seed is None:
            return self.backend.rewrite(messages, self._temperature())
        self.backend.seed = (seed + variant) % 2**32  # type: ignore[attr-defined]
        try:
            return self.backend.rewrite(messages, self._temperature())
        finally:
            self.backend.seed = seed  # type: ignore[attr-defined]

    def _examples(self, core: str) -> list[str]:
        """The author's passages to show alongside ``core``."""
        if (
            self.options.tuned
            or not self.options.examples
            or not self.voice.examples
            or "examples" in self.options.omit
        ):
            return []
        if self._library is None:
            self._library = Library(self.voice.examples)
        return self._library.pick(core, self.options.examples)

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

    def _rewrite_piece(self, piece: str) -> str:
        """A piece of a failed chunk: separators and list markers stay as
        written, and the rest is rewritten on its own."""
        if len(piece.split()) < MIN_PIECE_WORDS or piece.upper() == piece:
            return piece  # a greeting, sign-off or heading: nothing to restyle
        marker = _LIST_ITEM_RE.match(piece)
        if marker:
            return marker.group(0) + self._rewrite_chunk(piece[marker.end() :])
        return self._rewrite_chunk(piece)

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

    def _map(self, work, items: list) -> list:
        """``work(rewriter, item)`` for every item, up to ``parallel`` at a
        time, results in order.  Each worker has its own copy of the backend
        (a candidate's sampling seed is set on it) and its own report, which
        are merged in order, so warnings read the same as a serial run."""
        if self.options.parallel == 1 or len(items) < 2:
            return [work(self, item) for item in items]
        if self.voice.examples and self._library is None:
            self._library = Library(self.voice.examples)  # shared, read-only
        workers = []
        for _ in items:
            worker = copy.copy(self)
            worker.backend = copy.copy(self.backend)
            worker.report = Report(unit=self.report.unit)
            workers.append(worker)
        with ThreadPoolExecutor(min(self.options.parallel, len(items))) as pool:
            results = list(pool.map(work, workers, items))
        for worker in workers:
            self.report.merge(worker.report)
        return results

    def rewrite_units(self, units: list[str]) -> list[str]:
        """Rewrite independent paragraphs (of a .docx), returning one result each.

        Paragraphs travel in token-bounded batches, each prefixed ``[[n]]``;
        a batch whose markers come back damaged is redone one paragraph at
        a time.  Fidelity failures keep the original paragraph.
        """
        results = list(units)
        self.report.unit = "paragraph"
        self.progress.begin(len(units), "paragraphs")
        batches = self._unit_batches(units)
        jobs = [[units[i] for i in batch] for batch in batches]
        for batch, parts in zip(batches, self._map(Rewriter._batch_job, jobs)):
            for i, part in zip(batch, parts):
                results[i] = part
        return self._review(units, results)

    def _batch_job(self, batch: list[str]) -> list[str]:
        """One batch of paragraphs, rewritten together or else one by one."""
        parts = None
        if self._fits("\n\n".join(batch)):
            self.progress.advance(len(batch))
            return list(batch)
        if len(batch) > 1:
            text = "\n\n".join(f"[[{n}]] {unit}" for n, unit in enumerate(batch, 1))
            rewritten, _ = self._attempt(text)
            parts = None if rewritten is None else _split_units(rewritten, len(batch))
            self.report.chunks += parts is not None
        if parts is None:
            parts = []
            for unit in batch:
                parts.append(self._rewrite_chunk(unit))
                self.progress.advance()
        else:
            self.progress.advance(len(batch))
        return parts

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
        chunks = self._chunks(segment(text, self.options.text_format))
        prose = [seg.text for seg in chunks if seg.prose]
        self.progress.begin(len(prose), "chunks")
        done = iter(self._review(prose, self._map(Rewriter._chunk_job, prose)))
        return "".join(next(done) if seg.prose else seg.text for seg in chunks)

    def _chunk_job(self, text: str) -> str:
        result = self._rewrite_chunk(text)
        self.progress.advance()
        return result

    def _prompt(self, core: str, masked: str):
        """``prompt(feedback)``: the messages for an attempt at ``core``."""
        examples, focus = self._examples(core), self._focus(masked)
        if self.options.tuned:
            return lambda feedback: build_tuned_messages(masked, feedback, focus)
        return lambda feedback: build_messages(
            self.voice,
            masked,
            self.options.strength,
            feedback,
            self.options.rules,
            examples,
            self.options.omit,
            focus,
        )

    def _focus(self, masked: str) -> list[str] | None:
        """With ``Options.targeted``, the instructions that name the only
        sentences of ``masked`` to change."""
        if not self.options.targeted:
            return None
        return review.focus(review.flagged(masked, self.voice, self.options.rules)) or None

    def _fits(self, text: str) -> bool:
        """Whether ``text`` has nothing to change: with ``Options.targeted``,
        no sentence in it needs work, so the model is not asked at all."""
        if not self.options.targeted or review.flagged(text, self.voice, self.options.rules):
            return False
        self.report.untouched += 1
        return True

    def _review(self, sources: list[str], results: list[str]) -> list[str]:
        """``results`` checked against ``sources`` sentence by sentence: a
        change is kept only if it passes the rules and then the checker."""
        if not self.options.review:
            return results
        plans = [review.align(s, r) for s, r in zip(sources, results)]
        asks = [group for groups in plans for group in groups if self._screen(group)]
        if asks:
            self.progress.begin(len(asks), "changes checked")
            with ThreadPoolExecutor(min(self.options.parallel, len(asks))) as pool:
                list(pool.map(self._ask, asks))
        for groups in plans:
            for group in groups:
                if group.problems:
                    self._kept_sentence(group)
        return [review.assemble(groups) for groups in plans]

    def _screen(  # pylint: disable=too-many-return-statements  # one per way to settle
        self, group: review.Group
    ) -> bool:
        """Settle ``group`` by the rules alone; True when it is left for the
        checker to decide."""
        if not group.changed:
            return False
        if not group.edited:
            group.problems = [f"the rewrite drops the sentence {group.before!r}"]
            return False
        if not group.source:
            group.problems = [f"the rewrite adds the sentence {group.after!r}"]
            return False
        rules = self.options.rules
        if self.options.targeted and not any(
            review.needs_work(unit, self.voice, rules) for unit in group.source
        ):
            return False  # it already fits the voice: left as written
        group.edited = [review.restore_form(group.before, group.after)]
        if not group.changed:
            return False
        group.problems = review.checks(group.before, group.after, self.options.keep_person)
        if group.problems:
            return False
        group.accepted = self.judge is None
        return not group.accepted

    def _ask(self, group: review.Group) -> None:
        """Put one change to the checker (its own copy of the backend)."""
        judge = copy.copy(self.judge)
        reply = judge.rewrite(review.judge_messages(group.before, group.after), 0.0)
        problem = review.verdict(reply)
        if problem:
            group.problems = [problem]
        else:
            group.accepted = True
        self.progress.advance()

    def _kept_sentence(self, group: review.Group) -> None:
        detail = _summarize(group.problems)
        if self.options.strict:
            raise ValidationError(f"rewrite failed fidelity checks: {detail}")
        self.report.reverted.append(group.problems)
        sentence = group.before if len(group.before) <= 80 else group.before[:77] + "..."
        self.report.warnings.append(f"kept the original of {sentence!r}: {detail}")


def _keep_paragraphs(source: str, rewrite: str) -> str:
    """``rewrite`` without the paragraph breaks the model added to a source
    that has none: list items stay on their own lines, prose is rejoined.
    In a source with breaks, a blank line the model put before a list item
    the source runs on from the line above is removed."""
    if _BLANK_LINES_RE.search(source):
        spaced = {m.group(1) for m in _SPACED_ITEM_RE.finditer(source)}
        return _SPACED_ITEM_RE.sub(
            lambda m: m.group(0) if m.group(1) in spaced else "\n" + m.group(1), rewrite
        )
    parts = _BLANK_LINES_RE.split(rewrite)[::2]
    out = parts[0]
    for part in parts[1:]:
        out += ("\n" if _LIST_ITEM_RE.match(part) else " ") + part
    return out


def _pieces(text: str) -> list[str]:
    """``text`` split into smaller pieces to rewrite separately, losslessly:
    its paragraphs, or for a single paragraph that is a list, its lines.
    Empty when there is nothing smaller to try."""
    pieces = _BLANK_LINES_RE.split(text)
    if len(pieces) > 1:
        return pieces
    lines = text.splitlines(keepends=True)
    if len(lines) > 1 and any(_LIST_ITEM_RE.match(line) for line in lines):
        return lines
    return []


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
    ("match the author's", "voice drift"),
    ("replace stock AI", "AI phrasing"),
    ("use the first person plural", "singular pronouns"),
    ("rephrase to use no", "forbidden punctuation"),
    ("quotation ", "quotation reworded"),
    ("number ", "number changed"),
    ("markup ", "formatting lost"),
    ("the rewrite switched to", "language drift"),
    ("the rewrite was empty", "empty reply"),
    ("the rewrite has ", "length changed"),
    ("placeholder ", "placeholder lost"),
    ("name ", "name changed"),
    ("the rewrite says ", "person changed"),
    ("the rewrite changes ", "claim changed"),
    ("the rewrite drops ", "sentence dropped"),
    ("the rewrite adds ", "sentence added"),
    ("the checker found", "meaning changed"),
    ("placeholder left", "placeholder lost"),
)


def _category(problem: str) -> str:
    """A short name for the kind of problem ``problem`` describes."""
    for prefix, name in _CATEGORIES:
        if problem.startswith(prefix):
            return name
    return problem.split(" ", 1)[0] + " changed"  # url, email, citation, inline_code
