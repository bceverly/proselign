# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The author's own words from each message of a Unix mbox mailbox.

Meant for a *sent-mail* mailbox: every message is taken to be the
author's.  Each message's ``text/plain`` body is used, or its HTML body
turned into text when there is no plain one; attachments are never
decoded.  Everything that is someone else's words is dropped: quoted
lines, the original below "On ... wrote:", Outlook "From:/Sent:" headers
and forwarded messages, plus signatures.  Lines a mailbox repeats at the
end of many messages (sign-offs, signature blocks) are trimmed from each
message's tail.
"""

from __future__ import annotations

import email
import email.policy
import re
from collections import Counter
from email.message import EmailMessage
from html.parser import HTMLParser

from proselign.ingest.formats.base import ExtractError

MAX_PARTS = 500
# A tail line repeated in at least this many messages is boilerplate.
MIN_REPEATS = 2

_SEPARATOR = re.compile(rb"^From [^\r\n]*\r?\n", re.MULTILINE)
_ESCAPED_FROM = re.compile(r"^>(>*From )", re.MULTILINE)
# Where someone else's words begin; everything from here on is dropped.
_CUT = re.compile(
    r"^(?:"
    r"-{2,} ?(?i:original message|forwarded message) ?-{2,}"
    r"|Begin forwarded message:"
    r"|_{10,}"
    r"|\u2500{10,}"
    r"|[^\n]{0,80} is inviting you to a scheduled Zoom meeting\."
    r"|Microsoft Teams(?: meeting)?"
    r"|-- "
    r"|Sent from my .{1,40}"
    r"|On [^\n]{1,300}wrote:"
    r"|On [^\n]{1,200}\n[^\n]{0,200}wrote:"
    r")[ \t]*$"
    r"|^\*?From:\*? [^\n]{1,300}\n\*?(?:Sent|Date):\*? ",
    re.MULTILINE,
)
# Links and the placeholders mail clients leave for images.
_NOISE = re.compile(r"<?https?://[^\s>]+>?|\[(?:cid:|image|icon|logo)[^\]\n]{0,200}\]", re.I)
_QUOTED = re.compile(r"^[ \t]*>.*\n?", re.MULTILINE)
# A plain-text list item ("* point", "- point", "2. point") is a paragraph
# of its own, as it would be in a word processor.
_LIST_ITEM = re.compile(r"\n(?=[ \t]{0,8}(?:[*\u2022\u2013-]|\d{1,2}[.)])[ \t]+\S)")

_SKIP_TAGS = frozenset({"style", "script", "head", "title", "blockquote", "template"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "hr"}
)
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"})
# Mail clients mark the quoted original or the signature with these.
_STOP_MARKERS = ("gmail_quote", "gmail_signature", "divrplyfwdmsg", "appendonsend", "moz-cite")


def mbox_messages(data: bytes) -> list[str]:
    """The author's text of each message in the mailbox ``data``, in order."""
    if not data.startswith(b"From "):
        raise ExtractError("not an mbox mailbox (no leading 'From ' line)")
    bodies = []
    for chunk in _SEPARATOR.split(data)[1:]:
        try:
            body = _message_text(email.message_from_bytes(chunk, policy=email.policy.default))
        except (ValueError, LookupError, TypeError, AttributeError, IndexError, RecursionError):
            continue  # one damaged message must not cost the rest of the mailbox
        bodies.append(_own_words(body))
    return [text for text in _trim_repeated_tails(bodies) if text]


def mbox_text(data: bytes) -> str:
    """Every message's text as one string, messages separated by blank lines."""
    return "\n\n".join(mbox_messages(data))


def _message_text(message: EmailMessage) -> str:
    parts = list(message.walk())
    if len(parts) > MAX_PARTS:
        raise ValueError("too many MIME parts")
    plain = message.get_body(preferencelist=("plain",))
    if plain is not None:
        return _decode(plain)
    html = message.get_body(preferencelist=("html",))
    return html_text(_decode(html)) if html is not None else ""


def _decode(part: EmailMessage) -> str:
    try:
        return part.get_content()
    except LookupError:  # an unknown charset
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="replace")


def _own_words(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _ESCAPED_FROM.sub(r"\1", text)
    cut = _CUT.search(text)
    if cut:
        text = text[: cut.start()]
    text = _LIST_ITEM.sub("\n\n", _QUOTED.sub("", text))
    return _NOISE.sub("", text).strip()


def _trim_repeated_tails(bodies: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for body in bodies:
        counts.update({line.strip() for line in body.splitlines() if line.strip()})
    boilerplate = {line for line, count in counts.items() if count >= MIN_REPEATS}
    trimmed = []
    for body in bodies:
        lines = body.splitlines()
        while lines and (not lines[-1].strip() or lines[-1].strip() in boilerplate):
            lines.pop()
        trimmed.append("\n".join(lines).strip())
    return trimmed


def html_text(markup: str) -> str:
    """Readable text of an HTML mail body, without quoted or signature blocks."""
    parser = _MailHTML()
    parser.feed(markup)
    parser.close()
    lines = "\n".join(" ".join(line.split()) for line in "".join(parser.out).split("\n"))
    return re.sub(r"\n{3,}", "\n\n", lines).strip()


class _MailHTML(HTMLParser):
    """Collects visible text; stops at the first quoted-original marker."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skipping: list[str] = []
        self.stopped = False

    def handle_starttag(self, tag, attrs):
        if self.stopped:
            return
        marks = " ".join(v or "" for k, v in attrs if k in ("class", "id")).lower()
        if any(marker in marks for marker in _STOP_MARKERS):
            self.stopped = True
        elif tag in _SKIP_TAGS:
            self.skipping.append(tag)
        elif tag in _BLOCK_TAGS:
            self.out.append("\n\n" if tag == "li" else "\n")

    def handle_endtag(self, tag):
        if self.skipping and self.skipping[-1] == tag:
            self.skipping.pop()
        elif tag in _BLOCK_TAGS and tag not in _VOID_TAGS:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.stopped and not self.skipping:
            self.out.append(data)
