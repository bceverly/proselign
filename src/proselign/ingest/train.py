# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Build (or extend) a voice from a corpus of the author's documents."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from proselign.errors import UsageError
from proselign.ingest.corpus import load_documents
from proselign.ingest.stylometry import describe, measure, words
from proselign.voice.profile import Voice

MIN_WORDS = 200


def build_voice(
    name: str,
    paths: list[Path],
    base: Voice | None = None,
    on_skip: Callable[[Path, str], None] | None = None,
) -> Voice:
    """Measure every document under ``paths`` and return the resulting voice.

    With ``base`` (``train --append``), previously recorded sources whose
    files are still readable are re-measured together with the new ones, so
    the profile always describes the whole corpus, never a running average.
    ``on_skip`` hears about each file that could not be used.
    """
    sources = list(paths)
    if base is not None:
        sources.extend(
            Path(s["path"]) for s in base.sources if "path" in s and Path(s["path"]).is_file()
        )
    documents = load_documents(sources, on_skip)
    if not documents:
        raise UsageError("no readable documents found (.txt, .md, .docx, .doc, .pdf, .mbox)")

    corpus = "\n\n".join(doc.text for doc in documents)
    total_words = len(words(corpus))
    if total_words < MIN_WORDS:
        raise UsageError(
            f"corpus has only {total_words} words; at least {MIN_WORDS} are needed "
            "for a meaningful profile"
        )

    metrics = measure(corpus)
    voice = base or Voice(display_name=name)
    voice.metrics = metrics
    voice.style = describe(metrics)
    voice.corpus_summary = {
        "documents": sum(doc.parts for doc in documents),
        "words": total_words,
        "tokens_estimated": int(total_words * 1.3),
    }
    voice.sources = [{"path": str(doc.path.resolve()), "sha256": doc.sha256} for doc in documents]
    voice.updated_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return voice
