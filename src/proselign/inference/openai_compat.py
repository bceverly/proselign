# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Backend for any server speaking the OpenAI-compatible chat API.

llama.cpp's ``llama-server``, vLLM, Ollama and LM Studio all expose this
endpoint shape, so one small client covers them all. Standard library only.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from proselign.errors import BackendError, UsageError
from proselign.inference.base import Backend, Message

TIMEOUT_SECONDS = 600
MAX_SEED = 2**32 - 1  # llama.cpp and vLLM take an unsigned 32-bit seed


def is_loopback(endpoint: str) -> bool:
    """True when ``endpoint`` resolves only to loopback addresses."""
    host = urlparse(endpoint).hostname
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_loopback for i in infos)


class OpenAICompatBackend(Backend):
    """POSTs to ``{endpoint}/chat/completions``."""

    name = "openai-compat"

    def __init__(
        self, endpoint: str, model: str, offline: bool = False, seed: int | None = None
    ) -> None:
        scheme = urlparse(endpoint).scheme
        if scheme not in ("http", "https"):
            raise UsageError(f"endpoint must be an http(s) URL, got {endpoint!r}")
        if seed is not None and not 0 <= seed <= MAX_SEED:
            raise UsageError(f"--seed must be between 0 and {MAX_SEED}")
        if offline and not is_loopback(endpoint):
            raise UsageError(f"--offline forbids the non-local endpoint {endpoint}")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.seed = seed

    def rewrite(self, messages: list[Message], temperature: float) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        # The scheme was restricted to http(s) in __init__.
        request = urllib.request.Request(  # nosec B310
            f"{self.endpoint}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # nosec B310 # nosemgrep
                request, timeout=TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendError(f"cannot reach inference endpoint {self.endpoint}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendError(f"inference endpoint returned invalid JSON: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "inference endpoint reply has no choices[0].message.content"
            ) from exc
