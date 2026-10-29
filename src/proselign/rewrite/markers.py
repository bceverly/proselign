# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Delimiters around the user's text in a prompt.

Each prompt gets a fresh random nonce in its markers, so text being
rewritten cannot forge a closing marker and smuggle "trusted" instructions
past the data boundary. Kept free of imports so backends can use it without
importing the prompt builder (which itself imports backends).
"""

from __future__ import annotations

import re
import secrets

_END_RE = re.compile(r"<<<END_PROSELIGN_INPUT_([0-9a-f]{16})>>>\s*\Z")


def make_markers() -> tuple[str, str]:
    """A matching (begin, end) marker pair with an unguessable nonce."""
    nonce = secrets.token_hex(8)
    return f"<<<PROSELIGN_INPUT_{nonce}>>>", f"<<<END_PROSELIGN_INPUT_{nonce}>>>"


def extract_input(content: str) -> str | None:
    """The text between the final marker pair, or None when there is none.

    The end marker must close the message, and the begin marker must carry
    the same nonce; forged markers inside the text therefore never match.
    """
    end = _END_RE.search(content)
    if not end:
        return None
    begin = f"<<<PROSELIGN_INPUT_{end.group(1)}>>>\n"
    start = content.find(begin)
    if start == -1:
        return None
    return content[start + len(begin) : end.start()].removesuffix("\n")
