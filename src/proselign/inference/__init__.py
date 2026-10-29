# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Inference backends behind one small, engine-independent interface."""

from __future__ import annotations

import os

from proselign.errors import UsageError
from proselign.inference.base import Backend
from proselign.inference.mock import MockBackend
from proselign.inference.openai_compat import OpenAICompatBackend

BACKENDS = ("openai-compat", "mock")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1"


def make_backend(
    name: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    offline: bool = False,
    seed: int | None = None,
) -> Backend:
    """Build a backend from explicit arguments, falling back to environment.

    ``PROSELIGN_BACKEND``, ``PROSELIGN_ENDPOINT`` and ``PROSELIGN_MODEL``
    supply defaults so a shell profile can configure every invocation.
    """
    name = name or os.environ.get("PROSELIGN_BACKEND", "openai-compat")
    if name == "mock":
        return MockBackend()
    if name == "openai-compat":
        return OpenAICompatBackend(
            endpoint=endpoint or os.environ.get("PROSELIGN_ENDPOINT", DEFAULT_ENDPOINT),
            model=model or os.environ.get("PROSELIGN_MODEL", "local"),
            offline=offline,
            seed=seed,
        )
    raise UsageError(f"unknown backend {name!r}; choose one of: {', '.join(BACKENDS)}")


__all__ = ["BACKENDS", "Backend", "MockBackend", "OpenAICompatBackend", "make_backend"]
