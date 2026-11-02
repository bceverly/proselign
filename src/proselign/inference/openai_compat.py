# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Backend for any server speaking the OpenAI-compatible chat API.

llama.cpp's ``llama-server``, vLLM, Ollama and LM Studio all expose this
endpoint shape, so one small client covers them all. Standard library only.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from proselign.errors import BackendError, UsageError
from proselign.inference.base import Backend, Message

TIMEOUT_SECONDS = 600
# Replies are capped, so a model that never stops (Ollama keeps generating
# past its context window) fails fast instead of running into the timeout.
MIN_REPLY_TOKENS = 512
MAX_REPLY_TOKENS = 8192
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
            "max_tokens": reply_budget(messages),
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
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"cannot reach inference endpoint {self.endpoint}: {exc}") from exc
        except TimeoutError as exc:
            raise BackendError(
                f"model {self.model!r} at {self.endpoint} did not answer within "
                f"{TIMEOUT_SECONDS} seconds; it may be too large for this machine, or the "
                "server may be busy with another request"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise BackendError(
                f"the connection to {self.endpoint} failed mid-request: {exc}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendError(f"inference endpoint returned invalid JSON: {exc}") from exc
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "inference endpoint reply has no choices[0].message.content"
            ) from exc
        # Cut off at the cap: a partial rewrite is no rewrite (the caller retries).
        return "" if choice.get("finish_reason") == "length" else content

    def _http_error(self, exc: urllib.error.HTTPError) -> BackendError:
        """The server answered but refused: say why, as plainly as it allows."""
        detail = _error_detail(exc)
        if exc.code == 404:
            # Ollama and llama.cpp answer 404 for a model they don't have.
            return BackendError(
                f"model {self.model!r} was not found at {self.endpoint}"
                + (f" ({detail})" if detail else "")
                + f"; with Ollama, download it first: ollama pull {self.model}"
            )
        return BackendError(
            f"inference endpoint {self.endpoint} returned HTTP {exc.code}"
            + (f": {detail}" if detail else "")
        )


def reply_budget(messages: list[Message]) -> int:
    """A reply-length cap: about twice the request's length in tokens (at
    roughly four characters a token), within fixed bounds."""
    request_chars = sum(len(m.content) for m in messages)
    return max(MIN_REPLY_TOKENS, min(MAX_REPLY_TOKENS, request_chars // 2))


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The server's own error message, if its reply body carries one."""
    try:
        body = json.loads(exc.read(4096).decode("utf-8"))
    except (OSError, ValueError):
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        error = error.get("message")
    if not isinstance(error, str):
        return ""
    return " ".join(error.split())[:200]
