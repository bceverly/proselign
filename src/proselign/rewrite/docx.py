# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Rewrite the prose of a ``.docx`` file in place, keeping everything else.

The package is copied entry by entry; only the body, footnote and endnote
parts are edited, and within them only eligible paragraphs.  Edits are
made on the original XML text, token by token, so every byte outside a
rewritten paragraph is unchanged: no XML library re-serializes the part,
which would rename namespace prefixes that Word depends on.

A paragraph is rewritten when it is plain runs of text.  Headings, titles,
captions, tables of contents, hidden text and paragraphs holding fields,
links, images, line breaks, tabs or tracked changes are left exactly as
they are, as are paragraphs that are not full sentences (subject lines, labels,
table cells).  Character
formatting inside a paragraph (bold, italic, ...) is carried through the
model as ``<s1>...</s1>`` tags around the affected words.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from proselign.errors import UsageError

MAX_PART_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MIN_WORDS = 5

_PARTS = re.compile(r"word/(?:document|footnotes|endnotes)\d*\.xml")
_TOKEN = re.compile(r"<!--.*?-->|<\?.*?\?>|<!\[CDATA\[.*?\]\]>|<![^>]*>|<[^>]+>|[^<]+", re.DOTALL)
_TAG_NAME = re.compile(r"</?([^\s/>]+)")
_SKIP_STYLES = re.compile(r'w:pStyle w:val="(?:Heading|Title|Subtitle|TOC|Caption)', re.I)
_PARAGRAPH_CHILDREN = frozenset({"w:pPr", "w:r", "w:proofErr", "w:bookmarkStart", "w:bookmarkEnd"})
_RUN_CHILDREN = frozenset({"w:rPr", "w:t", "w:lastRenderedPageBreak"})
_STYLE_TAG = re.compile(r"(</?s\d+>)")
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s*\Z")
# A typed list marker ("1.", "a)", "\u2022") that a leading run ends with a tab.
_LIST_MARKER = re.compile(
    r"[\u2022\u00b7\u25aa\u25cf\u25e6\u2013*o-]|\(?(?:\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,6})[.)]"
)
_RESERVED = re.compile(r"</?s\d+>|\[\[\d+\]\]")
_ENTITY = re.compile(r"&(#x[0-9A-Fa-f]{1,6}|#\d{1,7}|lt|gt|amp|quot|apos);")
_NAMED = {"lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'"}

RewriteUnits = Callable[[list[str]], list[str]]


@dataclass
class DocxReport:
    """What happened to the document's paragraphs."""

    rewritten: int = 0
    kept: Counter = field(default_factory=Counter)


@dataclass
class _Paragraph:
    """An eligible paragraph: where it sits and how to rebuild it."""

    start: int  # index of the <w:p> token
    end: int  # index of the </w:p> token
    keep: list[str]  # serialized non-run children, in order
    run_at: int  # position in ``keep`` where the runs go
    styles: list[tuple[str, str]]  # (run start tag, rPr XML) per style id
    text: str  # with <sN> tags around non-base formatting


def rewrite_docx(data: bytes, rewrite_units: RewriteUnits) -> tuple[bytes, DocxReport]:
    """Return ``data`` with its prose rewritten by ``rewrite_units``."""
    report = DocxReport()
    try:
        source = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UsageError(f"not a .docx file: {exc}") from exc
    out = io.BytesIO()
    total = 0
    with source, zipfile.ZipFile(out, "w") as target:
        if "word/document.xml" not in source.namelist():
            raise UsageError("not a Word document (no word/document.xml)")
        for info in source.infolist():
            total += info.file_size
            if info.file_size > MAX_PART_BYTES or total > MAX_PACKAGE_BYTES:
                raise UsageError(f"{info.filename} is too large to rewrite")
            body = _read(source, info)
            if _PARTS.fullmatch(info.filename):
                body = _rewrite_part(body, rewrite_units, report)
            target.writestr(info, body)
    return out.getvalue(), report


def _read(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        with archive.open(info) as handle:
            body = handle.read(MAX_PART_BYTES + 1)
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise UsageError(f"corrupt .docx entry {info.filename}: {exc}") from exc
    if len(body) > MAX_PART_BYTES:  # the size in the zip header can lie
        raise UsageError(f"{info.filename} is too large to rewrite")
    return body


def _rewrite_part(body: bytes, rewrite_units: RewriteUnits, report: DocxReport) -> bytes:
    try:
        xml = body.decode("utf-8")
    except UnicodeDecodeError:
        report.kept["part not UTF-8"] += 1
        return body
    if "<!DOCTYPE" in xml or "<!ENTITY" in xml:
        raise UsageError("document XML contains a DTD, which Office files never do")
    tokens = _TOKEN.findall(xml)
    ends = _match(tokens)
    paragraphs = []
    for i, token in enumerate(tokens):
        if _name(token) == "w:p" and not token.startswith("</") and ends[i] > i:
            paragraph = _eligible(tokens, ends, i, report)
            if paragraph is not None:
                paragraphs.append(paragraph)
    if not paragraphs:
        return body
    rewritten = rewrite_units([p.text for p in paragraphs])
    for paragraph, text in sorted(
        zip(paragraphs, rewritten), key=lambda pair: pair[0].start, reverse=True
    ):
        runs = _runs(paragraph, text)
        if runs is None or text.split() == paragraph.text.split():
            report.kept["unchanged by the rewrite"] += 1
            continue
        report.rewritten += 1
        inner = paragraph.keep[: paragraph.run_at] + [runs] + paragraph.keep[paragraph.run_at :]
        tokens[paragraph.start + 1 : paragraph.end] = ["".join(inner)]
    return "".join(tokens).encode("utf-8")


def _name(token: str) -> str:
    if not token.startswith("<") or token.startswith(("<?", "<!")):
        return ""
    match = _TAG_NAME.match(token)
    return match.group(1) if match else ""


def _match(tokens: list[str]) -> list[int]:
    """For each start tag, the index of its end tag (itself if self-closing)."""
    ends = list(range(len(tokens)))
    stack: list[tuple[str, int]] = []
    for i, token in enumerate(tokens):
        name = _name(token)
        if not name or token.endswith("/>"):
            continue
        if not token.startswith("</"):
            stack.append((name, i))
            continue
        while stack:
            open_name, j = stack.pop()
            if open_name == name:
                ends[j] = i
                break
        else:
            raise UsageError("document XML is malformed (unbalanced tags)")
    return ends


def _children(tokens: list[str], ends: list[int], start: int) -> list[tuple[int, int]]:
    """``(first, last)`` token ranges of the direct children of ``start``."""
    out, k = [], start + 1
    while k < ends[start]:
        last = ends[k] if _name(tokens[k]) else k
        out.append((k, last))
        k = last + 1
    return out


def _eligible(tokens, ends, start, report: DocxReport) -> _Paragraph | None:
    try:
        keep, runs, run_at = _structure(tokens, ends, start)
    except _Ineligible as exc:
        return _skip(report, str(exc))
    plain = "".join(text for _, _, text in runs)
    if not plain.strip():
        return None  # an empty or spacer paragraph: nothing to say about it
    if "w:vanish" in "".join(rpr for _, rpr, _ in runs):
        return _skip(report, "hidden text")
    if len(plain.split()) < MIN_WORDS or not _SENTENCE_END.search(plain):
        return _skip(report, "not full sentences (titles, labels, table cells)")
    if _RESERVED.search(plain):
        return _skip(report, "unusual content")
    styles, text = _tag(runs)
    return _Paragraph(start, ends[start], keep, run_at, styles, text)


class _Ineligible(Exception):
    """A paragraph that must be left as it is; the message says why."""


def _structure(tokens, ends, start) -> tuple[list[str], list[tuple[str, str, str]], int]:
    """A paragraph's non-run children, its runs, and where the runs sit."""
    keep: list[str] = []
    runs: list[tuple[str, str, str]] = []  # (start tag, rPr XML, text)
    run_at = -1
    keep_marker = False
    for first, last in _children(tokens, ends, start):
        name = _name(tokens[first])
        xml = "".join(tokens[first : last + 1])
        if not name and xml.strip():
            raise _Ineligible("unusual content")
        if name and name not in _PARAGRAPH_CHILDREN:
            raise _Ineligible("fields, links, images or tracked changes")
        if name == "w:pPr" and _SKIP_STYLES.search(xml):
            raise _Ineligible("headings and captions")
        if name == "w:r":
            run = _run(tokens, ends, first)
            if run is None:
                raise _Ineligible("tabs, breaks or embedded objects")
            if run[3]:  # ends with a tab
                # Only a leading "1.<tab>" or "\u2022<tab>" run is allowed; it is kept as is.
                if runs or keep_marker or not _LIST_MARKER.fullmatch(run[2].strip()):
                    raise _Ineligible("tabs, breaks or embedded objects")
                keep_marker = True
                keep.append(xml)
                continue
            run_at = len(keep) if run_at < 0 else run_at
            runs.append(run[:3])
        elif name != "w:proofErr":
            keep.append(xml)
    return keep, runs, run_at


def _skip(report: DocxReport, reason: str) -> None:
    report.kept[reason] += 1


def _run(tokens, ends, start) -> tuple[str, str, str, bool] | None:
    """``(start tag, rPr XML, text, ends with a tab)``, or None when the run
    holds anything but text (a tab is allowed only as its last child)."""
    rpr, text, tabbed = "", [], False
    for first, last in _children(tokens, ends, start):
        name = _name(tokens[first])
        if not name:
            continue  # whitespace between elements
        if tabbed or (name not in _RUN_CHILDREN and name != "w:tab"):
            return None
        if name == "w:rPr":
            rpr = "".join(tokens[first : last + 1])
        elif name == "w:t":
            text.extend(_unescape(t) for t in tokens[first + 1 : last] if not _name(t))
        elif name == "w:tab":
            tabbed = True
    return tokens[start], rpr, "".join(text), tabbed


def _tag(runs: list[tuple[str, str, str]]) -> tuple[list[tuple[str, str]], str]:
    """Style table and tagged text; the most-used formatting goes untagged."""
    weight: Counter = Counter()
    for _, rpr, text in runs:
        weight[rpr] += len(text)
    base = weight.most_common(1)[0][0]
    ids: dict[str, int] = {base: 0}
    styles: list[tuple[str, str]] = []
    pieces = []
    for start_tag, rpr, text in runs:
        if not text:
            continue
        if rpr not in ids:
            ids[rpr] = len(ids)
        if len(styles) <= ids[rpr]:
            styles.extend([("", "")] * (ids[rpr] + 1 - len(styles)))
        if not styles[ids[rpr]][0]:
            styles[ids[rpr]] = (start_tag, rpr)
        number = ids[rpr]
        pieces.append(text if number == 0 else f"<s{number}>{text}</s{number}>")
    # Neighbouring runs with the same formatting become one tagged span.
    return styles, re.sub(r"</s(\d+)><s\1>", "", "".join(pieces))


def _runs(paragraph: _Paragraph, text: str) -> str | None:
    """New run XML for rewritten ``text``, or None when its tags are broken."""
    text = " ".join(text.split())
    out, current = [], 0
    for piece in _STYLE_TAG.split(text):
        tag = re.fullmatch(r"<(/?)s(\d+)>", piece)
        if tag:
            number = int(tag.group(2))
            closing = tag.group(1) == "/"
            if number == 0 or number >= len(paragraph.styles):
                return None
            if closing != (current == number) or (not closing and current):
                return None
            current = 0 if closing else number
        elif piece:
            start_tag, rpr = paragraph.styles[current]
            out.append(f'{start_tag}{rpr}<w:t xml:space="preserve">{_escape(piece)}</w:t></w:r>')
    return None if current or not out else "".join(out)


def _unescape(text: str) -> str:
    def entity(match: re.Match) -> str:
        name = match.group(1)
        if name.startswith("#"):
            code = int(name[2:], 16) if name[1] in "xX" else int(name[1:])
            return chr(code) if 0 < code <= 0x10FFFF else ""
        return _NAMED[name]

    return _ENTITY.sub(entity, text)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
