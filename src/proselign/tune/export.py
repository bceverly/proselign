# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Turn a trained adapter into an Ollama model.

The adapter (``adapter_model.safetensors`` and ``adapter_config.json``, as
peft saves them) is rewritten as a GGUF LoRA adapter: each low-rank pair
``lora_A`` / ``lora_B`` is stored under the name of the base tensor it
modifies (``blk.7.attn_q.weight.lora_a``), with the scaling ``alpha``.
Ollama then builds a model from the base it already has plus the adapter.

Converting needs ``numpy``, ``safetensors`` and ``gguf`` (the training
extras); registering uses Ollama's own HTTP API.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path
from urllib.parse import urlparse

from proselign.errors import UsageError

ADAPTER_FILE = "adapter.gguf"
CONTEXT_TOKENS = 8192
# peft name -> GGUF name, for the Llama-style families (Qwen2, Llama, Mistral)
_PROJECTIONS = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_ARCHITECTURES = {"qwen2": "qwen2", "llama": "llama", "mistral": "llama"}
_LORA_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(\w+\.\w+)\.lora_([AB])(?:\.default)?\.weight$")
_MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}(?::[a-z0-9._-]{1,64})?$")


def gguf_name(peft_name: str) -> str | None:
    """The GGUF tensor name for one of peft's LoRA tensors, or None for a
    tensor this converter does not know."""
    match = _LORA_NAME_RE.search(peft_name)
    if not match or match.group(2) not in _PROJECTIONS:
        return None
    layer, projection, side = match.groups()
    return f"blk.{layer}.{_PROJECTIONS[projection]}.weight.lora_{side.lower()}"


def architecture(base_config: dict) -> str:
    """The GGUF architecture name for a base model's ``config.json``."""
    kind = str(base_config.get("model_type", ""))
    if kind not in _ARCHITECTURES:
        raise UsageError(
            f"cannot convert an adapter for a {kind or 'unknown'!r} model; "
            f"supported: {', '.join(sorted(_ARCHITECTURES))}"
        )
    return _ARCHITECTURES[kind]


def to_gguf(adapter_dir: Path, base_config: dict, out_path: Path) -> Path:
    """Write the adapter in ``adapter_dir`` as a GGUF LoRA file."""
    try:
        gguf = import_module("gguf")
        numpy = import_module("numpy")
        safe = import_module("safetensors.numpy")
    except ImportError as exc:
        raise UsageError(
            "converting the adapter needs numpy, safetensors and gguf: "
            "pip install 'proselign[tune]'"
        ) from exc
    config = json.loads((adapter_dir / "adapter_config.json").read_text("utf-8"))
    tensors = safe.load_file(str(adapter_dir / "adapter_model.safetensors"))
    named = {}
    for name, value in tensors.items():
        target = gguf_name(name)
        if target is None:
            raise UsageError(f"the adapter has a tensor this converter does not know: {name}")
        named[target] = value.astype(numpy.float16)
    if not named:
        raise UsageError(f"no LoRA tensors in {adapter_dir}")
    writer = gguf.GGUFWriter(str(out_path), architecture(base_config))
    writer.add_string(gguf.Keys.General.TYPE, gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(config.get("lora_alpha", 16)))
    for target in sorted(named):
        writer.add_tensor(target, named[target])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    out_path.chmod(0o600)
    return out_path


def _post(url: str, body: bytes, content_type: str, timeout: int) -> dict:
    """POST to the Ollama API; its JSON reply, or a plain error."""
    request = urllib.request.Request(  # nosec B310 - the scheme was checked
        url, body, {"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 # nosemgrep
            reply = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = " ".join(exc.read(2000).decode("utf-8", "replace").split())
        raise UsageError(f"the model server refused {url}: HTTP {exc.code} {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise UsageError(f"cannot reach the model server at {url}: {exc}") from exc
    try:
        return json.loads(reply) if reply.strip() else {}
    except ValueError:
        return {}


def register(name: str, base_model: str, adapter: Path, endpoint: str) -> None:
    """Create Ollama model ``name`` from ``base_model`` plus the adapter,
    through Ollama's API: upload the adapter as a blob, then create."""
    for model in (name, base_model):
        if not _MODEL_NAME_RE.match(model):
            raise UsageError(f"not a valid Ollama model name: {model!r}")
    root = ollama_root(endpoint)
    data = adapter.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    _post(f"{root}/api/blobs/{digest}", data, "application/octet-stream", 600)
    spec = {
        "model": name,
        "from": base_model,
        "adapters": {ADAPTER_FILE: digest},
        "parameters": {"num_ctx": CONTEXT_TOKENS},
        "stream": False,
    }
    reply = _post(f"{root}/api/create", json.dumps(spec).encode("utf-8"), "application/json", 600)
    if reply.get("error"):
        raise UsageError(f"the model server could not create {name}: {reply['error']}")


def ollama_root(endpoint: str) -> str:
    """The Ollama server's own API root for an OpenAI-compatible endpoint."""
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise UsageError(f"endpoint must be an http(s) URL, got {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def unload_models(endpoint: str) -> list[str]:
    """Ask Ollama to free the GPU (training needs all of it); returns the
    models it unloaded.  A server that is not Ollama is left alone."""
    root = ollama_root(endpoint)
    try:
        listing = f"{root}/api/ps"
        with urllib.request.urlopen(listing, timeout=10) as response:  # nosec B310 # nosemgrep
            loaded = [m["name"] for m in json.loads(response.read()).get("models", [])]
        for model in loaded:
            body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
            request = urllib.request.Request(  # nosec B310 - the scheme was checked
                f"{root}/api/generate", body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=60):  # nosec B310 # nosemgrep
                pass
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        return []
    return loaded
