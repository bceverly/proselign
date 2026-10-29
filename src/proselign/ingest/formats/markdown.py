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

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?(?:^ {0,3}\1[^\n]*$|\Z)", re.M | re.S)
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
_ATX = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*?)[ \t#]*$", re.M)
_SETEXT = re.compile(r"^([^\n]+)\n {0,3}(?:=+|-+)[ \t]*$", re.M)
_RULE = re.compile(r"^ {0,3}(?:[-*_][ \t]*){3,}$", re.M)
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?", re.M)


def markdown_text(text: str) -> str:
    """``text`` with Markdown markup removed and block structure kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FRONT_MATTER.sub("", text)
    text = _FENCE.sub("", text)
    text = _INDENTED_CODE.sub("", text)
    text = _TABLE_ROW.sub("", text)
    text = _REFERENCE.sub("", text)
    text = _SETEXT.sub(r"\n\1\n", text)
    text = _RULE.sub("", text)
    text = _ATX.sub(r"\n\1\n", text)
    text = _LIST_ITEM.sub("\n", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2), text)
    text = _HTML.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    for _ in range(3):  # nested emphasis: ***both*** or **_this_**
        text = _EMPHASIS.sub(r"\2", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
