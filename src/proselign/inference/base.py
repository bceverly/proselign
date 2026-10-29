# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""The contract every inference backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """One chat message: ``role`` is ``system`` or ``user``."""

    role: str
    content: str


class Backend(ABC):
    """A text-generation engine. Implementations must not print to stdout."""

    name = "abstract"

    @abstractmethod
    def rewrite(self, messages: list[Message], temperature: float) -> str:
        """Return the model's reply to ``messages``."""

    def count_tokens(self, text: str) -> int:
        """Approximate token count; backends with a tokenizer may do better."""
        return max(1, len(text) // 4)

    def health(self) -> str:
        """Short human-readable status, raising BackendError when unusable."""
        return "ok"
