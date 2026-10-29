# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Backend selection and the OpenAI-compatible client, against a local fake server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from proselign.errors import BackendError, UsageError
from proselign.inference import MockBackend, OpenAICompatBackend, make_backend
from proselign.inference.base import Message
from proselign.inference.openai_compat import is_loopback


class _Handler(BaseHTTPRequestHandler):
    reply: bytes = b""
    seen: list = []

    def do_POST(self):  # pylint: disable=invalid-name  # http.server API
        length = int(self.headers["Content-Length"])
        type(self).seen.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(type(self).reply)

    def log_message(self, *args):  # silence the test output
        pass


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def _endpoint(httpd) -> str:
    return f"http://127.0.0.1:{httpd.server_address[1]}/v1/"


def test_successful_completion(server):
    _Handler.reply = json.dumps({"choices": [{"message": {"content": "rewritten"}}]}).encode()
    backend = OpenAICompatBackend(_endpoint(server), "m", offline=True, seed=7)
    assert backend.rewrite([Message("user", "hi")], 0.3) == "rewritten"
    sent = _Handler.seen[0]
    assert sent["model"] == "m" and sent["seed"] == 7 and sent["temperature"] == 0.3


@pytest.mark.parametrize("reply", [b"not json", b'{"choices": []}', b"\xff"])
def test_bad_replies(server, reply):
    _Handler.reply = reply
    backend = OpenAICompatBackend(_endpoint(server), "m")
    with pytest.raises(BackendError):
        backend.rewrite([Message("user", "hi")], 0.3)


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
    assert backend.endpoint.endswith(":1234/v1") and backend.model == "qwen"
    monkeypatch.setenv("PROSELIGN_BACKEND", "mock")
    assert isinstance(make_backend(), MockBackend)
    with pytest.raises(UsageError):
        make_backend("gpt")
