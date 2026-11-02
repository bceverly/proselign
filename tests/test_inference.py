# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Backend selection and the OpenAI-compatible client, against a local fake server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from proselign.errors import BackendError, UsageError
from proselign.inference import MockBackend, OpenAICompatBackend, make_backend, openai_compat
from proselign.inference.base import Message
from proselign.inference.openai_compat import (
    MAX_REPLY_TOKENS,
    MIN_REPLY_TOKENS,
    is_loopback,
    reply_budget,
)


class _Handler(BaseHTTPRequestHandler):
    reply: bytes = b""
    status: int = 200
    delay: float = 0.0
    seen: list = []

    def do_POST(self):  # pylint: disable=invalid-name  # http.server API
        length = int(self.headers["Content-Length"])
        type(self).seen.append(json.loads(self.rfile.read(length)))
        time.sleep(type(self).delay)
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).reply)

    def log_message(self, *args):  # silence the test output
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(_Handler, "seen", [])
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _endpoint(httpd) -> str:
    return f"http://127.0.0.1:{httpd.server_address[1]}/v1/"


def test_successful_completion(server, monkeypatch):
    reply = json.dumps({"choices": [{"message": {"content": "rewritten"}}]}).encode()
    monkeypatch.setattr(_Handler, "reply", reply)
    backend = OpenAICompatBackend(_endpoint(server), "m", offline=True, seed=7)
    assert backend.rewrite([Message("user", "hi")], 0.3) == "rewritten"
    sent = _Handler.seen[0]
    assert sent["model"] == "m"
    assert sent["seed"] == 7
    assert sent["temperature"] == 0.3
    assert sent["max_tokens"] == MIN_REPLY_TOKENS


def test_reply_budget_scales_with_the_request():
    assert reply_budget([Message("user", "x" * 4000)]) == 2000
    assert reply_budget([Message("user", "x" * 100_000)]) == MAX_REPLY_TOKENS


def test_a_reply_cut_off_at_the_cap_counts_as_empty(server, monkeypatch):
    choice = {"message": {"content": "and so on and so on"}, "finish_reason": "length"}
    monkeypatch.setattr(_Handler, "reply", json.dumps({"choices": [choice]}).encode())
    backend = OpenAICompatBackend(_endpoint(server), "m")
    assert backend.rewrite([Message("user", "hi")], 0.3) == ""


def test_a_timeout_is_reported_plainly(server, monkeypatch):
    monkeypatch.setattr(openai_compat, "TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(_Handler, "delay", 1.0)
    backend = OpenAICompatBackend(_endpoint(server), "big")
    with pytest.raises(BackendError, match="did not answer within 0.2 seconds"):
        backend.rewrite([Message("user", "hi")], 0.3)


def test_a_dropped_connection_is_reported_plainly(monkeypatch):
    def drop(*_args, **_kwargs):
        raise ConnectionResetError("reset by peer")

    monkeypatch.setattr("urllib.request.urlopen", drop)
    backend = OpenAICompatBackend("http://127.0.0.1:9/v1", "m")
    with pytest.raises(BackendError, match="failed mid-request: reset by peer"):
        backend.rewrite([Message("user", "hi")], 0.3)


@pytest.mark.parametrize("reply", [b"not json", b'{"choices": []}', b"\xff"])
def test_bad_replies(server, reply, monkeypatch):
    monkeypatch.setattr(_Handler, "reply", reply)
    backend = OpenAICompatBackend(_endpoint(server), "m")
    messages = [Message("user", "hi")]
    with pytest.raises(BackendError):
        backend.rewrite(messages, 0.3)


@pytest.mark.parametrize(
    ("status", "reply", "expected"),
    [
        (
            404,
            b'{"error": {"message": "model \\"big\\" not found, try pulling it first"}}',
            "model 'big' was not found at http://127.0.0.1",
        ),
        (404, b"not json", "ollama pull big"),
        (500, b'{"error": "out of memory"}', "returned HTTP 500: out of memory"),
        (503, b'{"error": 7}', "returned HTTP 503"),
    ],
)
def test_http_errors_say_why(server, monkeypatch, status, reply, expected):
    monkeypatch.setattr(_Handler, "status", status)
    monkeypatch.setattr(_Handler, "reply", reply)
    backend = OpenAICompatBackend(_endpoint(server), "big")
    with pytest.raises(BackendError) as caught:
        backend.rewrite([Message("user", "hi")], 0.3)
    message = str(caught.value)
    assert expected in message
    assert "cannot reach" not in message


def test_endpoint_validation():
    with pytest.raises(UsageError, match="http"):
        OpenAICompatBackend("file:///etc/passwd", "m")
    with pytest.raises(UsageError, match="offline"):
        OpenAICompatBackend("http://198.51.100.1/v1", "m", offline=True)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8080", True),
        ("http://127.0.0.1", True),
        ("http://[::1]:80", True),
        ("http://198.51.100.1", False),
        ("not a url", False),
        ("http://name.invalid", False),
    ],
)
def test_is_loopback(url, expected):
    assert is_loopback(url) is expected


def test_is_loopback_resolves_names(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda host, port: [(0, 0, 0, "", ("127.0.0.2", 0))])
    assert is_loopback("http://my-gpu-box") is True


def test_make_backend(monkeypatch):
    assert isinstance(make_backend("mock"), MockBackend)
    monkeypatch.setenv("PROSELIGN_ENDPOINT", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("PROSELIGN_MODEL", "qwen")
    backend = make_backend()
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.endpoint.endswith(":1234/v1")
    assert backend.model == "qwen"
    monkeypatch.setenv("PROSELIGN_BACKEND", "mock")
    assert isinstance(make_backend(), MockBackend)
    with pytest.raises(UsageError):
        make_backend("gpt")
