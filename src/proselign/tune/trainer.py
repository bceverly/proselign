# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Train a LoRA adapter on (neutral, author) pairs: QLoRA on one GPU.

The base model is loaded in 4-bit (NF4) so a 14B model fits a 16 GB card;
only the adapter's low-rank matrices are trained, on the attention and MLP
projections.  Each example is the tuned rewrite prompt followed by the
author's passage, and the loss counts only the passage, so the model
learns to answer that prompt the way the author writes.

This module needs ``torch``, ``transformers``, ``peft`` and
``bitsandbytes`` (``pip install 'proselign[tune]'``), imported only when
training starts so the rest of proselign never depends on them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from proselign.errors import ProselignError, UsageError
from proselign.rewrite.prompt import build_tuned_messages

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXTRAS = ("torch", "transformers", "peft", "bitsandbytes")
GIB = 1 << 30
# Activations (with checkpointing), the adapter, its optimizer state, the
# 4-bit layers' dequantization buffers and the CUDA context, on top of the
# weights themselves.
TRAINING_OVERHEAD = 3 * GIB // 2


@dataclass
class Settings:
    """Training knobs; the defaults suit a few hundred passages."""

    epochs: int = 2
    rank: int = 16
    alpha: int = 32
    learning_rate: float = 1e-4
    max_tokens: int = 1024  # longer examples are cut (passages are far shorter)
    accumulate: int = 8  # examples per optimizer step

    def __post_init__(self) -> None:
        if not 1 <= self.epochs <= 10:
            raise UsageError("--epochs must be between 1 and 10")
        if self.rank not in (4, 8, 16, 32, 64):
            raise UsageError("--rank must be 4, 8, 16, 32 or 64")


def require_extras() -> None:
    """Fail plainly when the training libraries are missing."""
    missing = [name for name in EXTRAS if find_spec(name) is None]
    if missing:
        raise UsageError(
            f"fine-tuning needs {', '.join(missing)}: install the training extras with "
            "pip install 'proselign[tune]' (and a CUDA build of torch)"
        )


def memory_needed(config: dict) -> int:
    """Bytes of GPU memory that training the model ``config`` describes takes,
    roughly: its linear layers in 4-bit (with their scales), the embeddings
    and output head in 16-bit, and ``TRAINING_OVERHEAD``."""
    hidden = int(config.get("hidden_size", 0))
    heads = int(config.get("num_attention_heads", 0)) or 1
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv = int(config.get("num_key_value_heads", heads)) * head_dim
    layer = (
        2 * hidden * heads * head_dim  # q, o
        + 2 * hidden * kv  # k, v
        + 3 * hidden * int(config.get("intermediate_size", 0))  # gate, up, down
    )
    linear = layer * int(config.get("num_hidden_layers", 0))
    tables = 1 if config.get("tie_word_embeddings") else 2
    embeddings = tables * int(config.get("vocab_size", 0)) * hidden
    return int(linear * 0.5625) + embeddings * 2 + TRAINING_OVERHEAD


def free_memory() -> tuple[int, int]:  # pragma: no cover - needs a CUDA GPU
    """Free and total bytes on the GPU training would use."""
    torch = import_module("torch")
    if not torch.cuda.is_available():
        raise UsageError("fine-tuning needs a CUDA GPU; none is available to torch")
    free, total = torch.cuda.mem_get_info(0)
    return int(free), int(total)


def check_memory(needed: int, free: int, total: int, holders: list[str]) -> None:
    """Refuse to start training that would run out of GPU memory."""
    if free >= needed:
        return
    if needed > total:
        raise ProselignError(
            f"training this base model needs about {needed / GIB:.1f} GiB of GPU memory; "
            f"the GPU has {total / GIB:.1f} GiB (try a smaller --base and --model)"
        )
    held = (
        f" Ollama still has {', '.join(holders)} loaded (another program asked for it)."
        if holders
        else ""
    )
    raise ProselignError(
        f"training needs about {needed / GIB:.1f} GiB of free GPU memory, but only "
        f"{free / GIB:.1f} of {total / GIB:.1f} GiB is free.{held} Stop whatever is using "
        "the GPU and run tune again; the training pairs are kept, so it starts training "
        "straight away"
    )


def chat(pair: dict[str, str]) -> tuple[list[dict[str, str]], str]:
    """The prompt messages and the target reply for one pair."""
    messages = [
        {"role": m.role, "content": m.content} for m in build_tuned_messages(pair["source"])
    ]
    return messages, pair["target"]


def encode(tokenizer, pair: dict[str, str], max_tokens: int) -> tuple[list[int], list[int]]:
    """Token ids and labels for one pair: the prompt's labels are -100, so
    only the author's passage (and the end-of-turn token) is learned."""
    messages, target = chat(pair)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    end = tokenizer.eos_token or ""
    reply_ids = tokenizer(target + end, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + reply_ids)[:max_tokens]
    labels = ([-100] * len(prompt_ids) + reply_ids)[:max_tokens]
    return ids, labels


def train(  # pragma: no cover  # pylint: disable=too-many-locals  # needs a CUDA GPU
    pairs: list[dict[str, str]],
    base: str,
    out_dir: Path,
    settings: Settings,
    on_step: Callable[[int, int, float], None] | None = None,
) -> Path:
    """Train an adapter for ``base`` (a Hugging Face model id or directory)
    on ``pairs`` and save it in ``out_dir``; ``on_step(done, total, loss)``
    hears about each optimizer step."""
    require_extras()
    torch = import_module("torch")
    transformers = import_module("transformers")
    peft = import_module("peft")
    if not torch.cuda.is_available():
        raise UsageError("fine-tuning needs a CUDA GPU; none is available to torch")

    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    quantized = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quantized, dtype=torch.bfloat16, device_map={"": 0}
    )
    # Not peft.prepare_model_for_kbit_training: it casts the embeddings and
    # output head to 32-bit, about 6 GB more for a 14B model than a 16 GB
    # card has.  Checkpointing alone keeps activations small.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=settings.rank,
            lora_alpha=settings.alpha,
            lora_dropout=0.05,
            target_modules=list(TARGET_MODULES),
            task_type="CAUSAL_LM",
        ),
    )
    examples = [encode(tokenizer, pair, settings.max_tokens) for pair in pairs]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate
    )
    total = math.ceil(len(examples) / settings.accumulate) * settings.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 10) * max(0.0, 1 - step / max(total, 1))
    )
    generator = torch.Generator().manual_seed(0)
    model.train()
    step, running = 0, 0.0
    for _ in range(settings.epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        for n, index in enumerate(order, 1):
            ids, labels = examples[index]
            batch = {
                "input_ids": torch.tensor([ids], device="cuda"),
                "labels": torch.tensor([labels], device="cuda"),
            }
            loss = model(**batch).loss / settings.accumulate
            loss.backward()
            running += loss.item()
            if n % settings.accumulate == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if on_step is not None:
                    on_step(step, total, running)
                running = 0.0
    model.save_pretrained(str(out_dir))
    return out_dir
