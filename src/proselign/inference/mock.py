# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""A deterministic stand-in backend for tests and pipeline dry runs."""

from __future__ import annotations

from proselign.inference.base import Backend, Message
from proselign.rewrite.markers import extract_input


class MockBackend(Backend):
    """Echoes the text to be rewritten, unchanged.

    That makes every pipeline stage around inference (chunking, protection,
    validation, stdout discipline) testable without a model.
    """

    name = "mock"

    def rewrite(self, messages: list[Message], temperature: float) -> str:
        content = messages[-1].content
        text = extract_input(content)
        return content if text is None else text
