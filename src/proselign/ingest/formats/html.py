# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The prose of a web page or blog post, without its markup.

Training measures sentences and punctuation, so only the author's running
text may count: scripts, navigation, headers and footers, comments, tables,
figures, block quotes (someone else's words) and code are dropped.  Blogs
often put shell commands and configuration in ordinary paragraphs, one per
line, so lines that read as commands or settings rather than sentences are
dropped too.  Headings and list items become paragraphs of their own.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Elements whose content is never the author's prose.
_SKIP_TAGS = frozenset(
    {
        "aside",
        "blockquote",
        "button",
        "canvas",
        "figcaption",
        "figure",
        "footer",
        "form",
        "head",
        "header",
        "iframe",
        "math",
        "nav",
        "noscript",
        "object",
        "pre",
        "script",
        "select",
        "style",
        "svg",
        "table",
        "template",
        "textarea",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "article",
        "br",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "summary",
        "ul",
    }
)
_PARAGRAPH_TAGS = frozenset({"p", "li", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "h6", "hr"})
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
# Class or id tokens of page furniture that is not the post: comment threads
# (other people's words), sharing buttons, related-post lists.
_FURNITURE = frozenset(
    {
        "comments",
        "comment-respond",
        "respond",
        "sharedaddy",
        "sharing",
        "jp-relatedposts",
        "related-posts",
        "post-navigation",
        "screen-reader-text",
    }
)

_URL = re.compile(r"<?(?:https?|ftp)://[^\s>]+>?")
# A prompt: "$ cmd", "# cmd", "PS > cmd", "boot> cmd", "fdisk: 1> cmd",
# "user@host:~$ cmd".
_PROMPT = re.compile(r"^(?:[$#%]|PS ?>|\w+(?::\s*\d+)?>|\S+@\S+?[:~$#]\S*)\s")
# What marks a line that is not a sentence as code: a command name, an
# assignment ("KEY= value", "alias ls=colorls"), braces, a variable
# reference, a redirection or pipe, an operator prose never uses, or a
# leading file path.
_COMMANDS = "sudo|doas|echo|cat|cd|ls|cp|mv|rm|mkdir|chmod|chown|curl|wget|tar|ssh|apt|apt-get"
_CODE_MARK = re.compile(
    rf"^(?:{_COMMANDS})\s"
    r"|\S=\S|^\.?[\w.%]+\s*[?+:!]?="
    r"|[{};]\s*$|\$[A-Za-z_{(]|::|=>|>>|\|\s*\w|^~?/\S*/"
)
_SENTENCE_END = re.compile("[.?!:][\"'\u201d\u2019)]*$")
MIN_SENTENCE_WORDS = 6
# Prose is mostly letters and spaces; commands, paths and settings are not.
MIN_PROSE_RATIO = 0.75
_ARTICLE = re.compile(r"<article[\s>]", re.I)
_MAIN = re.compile(r"<main[\s>]", re.I)


def html_text(markup: str, whole_page: bool = False) -> str:
    """The author's prose in ``markup``, paragraphs separated by blank lines.

    With ``whole_page``, only the first ``<article>`` (or else ``<main>``)
    is read when the page has one, so menus and sidebars stay out.
    """
    scope = None
    if whole_page:
        scope = "article" if _ARTICLE.search(markup) else "main" if _MAIN.search(markup) else None
    parser = _ProseHTML(scope)
    parser.feed(markup)
    parser.close()
    paragraphs = []
    for block in re.split(r"\n[ \t\xa0]*\n", "".join(parser.out)):
        lines = (" ".join(line.split()) for line in block.split("\n"))
        kept = [line for line in lines if line and not is_code_like(line)]
        if kept:
            paragraphs.append(" ".join(kept))
    return "\n\n".join(paragraphs)


def html_title(markup: str) -> str:
    """The text of the page's ``<title>``, or ``""``."""
    match = re.search(r"<title[^>]*>(.*?)</title>", markup, re.I | re.S)
    return html_text(match.group(1)) if match else ""


def is_code_like(line: str) -> bool:
    """True when ``line`` reads as a command, path or setting, not prose.

    A full sentence is prose even when it mentions code; only a prompt at
    its start gives it away.
    """
    text = _URL.sub("", line).strip()
    if not text:
        return True  # nothing but a link
    if _PROMPT.match(text):
        return True
    if len(text.split()) >= MIN_SENTENCE_WORDS and _SENTENCE_END.search(text):
        return False
    if _CODE_MARK.search(text):
        return True
    prose = sum(1 for c in text if c.isalpha() or c.isspace())
    return prose / len(text) < MIN_PROSE_RATIO


class _ProseHTML(HTMLParser):
    """Collects visible prose inside ``scope`` (or everywhere when ``None``)."""

    def __init__(self, scope: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.scope = scope
        self.depth = 0  # nesting inside the scope element
        self.done = False  # the scope element has been read
        self.skipping: list[str] = []

    @property
    def collecting(self) -> bool:
        """Whether text seen now belongs to the output."""
        in_scope = self.scope is None or self.depth > 0
        return in_scope and not self.done and not self.skipping

    def handle_starttag(self, tag, attrs):
        if tag == self.scope and not self.done:
            self.depth += 1
        if tag in _VOID_TAGS:
            if tag == "br" and self.collecting:
                self.out.append("\n")
            return
        marks = {t for k, v in attrs if k in ("class", "id") and v for t in v.lower().split()}
        if self.skipping or tag in _SKIP_TAGS or marks & _FURNITURE:
            self.skipping.append(tag)
        elif tag in _BLOCK_TAGS and self.collecting:
            self.out.append("\n\n" if tag in _PARAGRAPH_TAGS else "\n")

    def handle_endtag(self, tag):
        if tag in self.skipping:
            # Unwind to the matching start tag, tolerating unclosed children.
            while self.skipping and self.skipping.pop() != tag:
                pass
        elif tag in _BLOCK_TAGS and self.collecting:
            self.out.append("\n\n" if tag in _PARAGRAPH_TAGS else "\n")
        if tag == self.scope and self.depth > 0:
            self.depth -= 1
            self.done = self.depth == 0

    def handle_data(self, data):
        if self.collecting:
            self.out.append(data.replace("\xa0", " "))
