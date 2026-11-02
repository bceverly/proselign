# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""``proselign tune``: fine-tune a local model on a voice.

Prompting a model with a description of a voice and a few passages gets
only part of the way to the author: the model keeps its own habits. A
small LoRA adapter trained on the author's own writing moves the model
itself.  The steps:

1. ``pairs``: each passage of the author's documents is flattened into
   plain, neutral prose by the current model, giving (neutral, author)
   pairs; facts are masked as placeholders on both sides, exactly as in a
   rewrite, so the tuned model learns to keep them.
2. ``trainer``: a LoRA adapter is trained (QLoRA, 4-bit base) to turn the
   neutral text back into the author's.  This step needs a CUDA GPU and the
   training extras (``pip install 'proselign[tune]'``); everything else is
   standard library.
3. ``export``: the adapter is converted to GGUF and registered with Ollama
   as a model of its own, on top of the same base model.

Rewrites in a tuned voice then use that model with a short prompt.
"""
