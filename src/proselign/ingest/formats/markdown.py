# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The prose of a Markdown file, without its markup.

Training measures sentences and punctuation, so Markdown syntax must not
count as writing: fenced and indented code, tables, HTML tags and images
are dropped; links keep their text; emphasis markers go; headings and list
items become paragraphs of their own, as they would be in a word processor.
"""

from __future__ import annotations

import re

# Fences and headings are handled line by line: as one regex over the whole
# text they backtrack badly on unusual input.
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_INDENTED_CODE = re.compile(r"(?:^(?: {4}|\t)[^\n]*\n?)+", re.M)
_TABLE_ROW = re.compile(r"^ {0,3}\|[^\n]*$\n?", re.M)
_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\n]*\)|\[([^\]\n]+)\]\[[^\]\n]*\]")
_REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:[^\n]*$\n?", re.M)
_HTML = re.compile(r"<[^>\n]{1,200}>")
# Not inside words, so snake_case_names and 2*3*4 survive.
_EMPHASIS = re.compile(r"(?<![\w*~])(\*\*|__|\*|_|~~)(?=\S)([^\n]*?\S)\1(?![\w*~])")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n(?:---|\.\.\.)\n", re.S)
_INLINE_CODE = re.compile(r"`+([^`\n]+)`+")
_ATX = re.compile(r" {0,3}#{1,6}[ \t]+")
_UNDERLINE = re.compile(r" {0,3}(?:=+|-+)[ \t]*")
_RULE = re.compile(r"^ {0,3}(?:[-*_][ \t]*){3,}$", re.M)
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?", re.M)


def markdown_text(text: str) -> str:
    """``text`` with Markdown markup removed and block structure kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FRONT_MATTER.sub("", text)
    text = _strip_fences(text)
    text = _INDENTED_CODE.sub("", text)
    text = _TABLE_ROW.sub("", text)
    text = _REFERENCE.sub("", text)
    text = _headings(text)
    text = _RULE.sub("", text)
    text = _LIST_ITEM.sub("\n", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2), text)
    text = _HTML.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    for _ in range(3):  # nested emphasis: ***both*** or **_this_**
        text = _EMPHASIS.sub(r"\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _strip_fences(text: str) -> str:
    """``text`` with fenced code blocks replaced by a blank line."""
    out: list[str] = []
    fence = ""
    for line in text.split("\n"):
        match = _FENCE.match(line)
        if fence:
            # Closed by a fence of the same character, at least as long.
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                fence = ""
        elif match:
            fence = match.group(1)
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _headings(text: str) -> str:
    """``text`` with ATX (# Title) and setext (Title / ===) headings reduced
    to their text, each set apart as a paragraph of its own."""
    out: list[str] = []
    plain = False  # whether the last line could be a setext heading's text
    for line in text.split("\n"):
        atx = _ATX.match(line)
        if plain and _UNDERLINE.fullmatch(line):
            out[-1] = f"\n{out[-1]}\n"
            plain = False
        elif atx:
            title = line[atx.end() :].rstrip(" \t#")
            out.append(f"\n{title}\n")
            plain = False
        else:
            out.append(line)
            plain = bool(line)
    return "\n".join(out)
