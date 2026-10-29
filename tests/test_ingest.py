# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Corpus discovery, normalization, stylometry and training."""

from __future__ import annotations

from pathlib import Path

import pytest

from proselign.errors import UsageError
from proselign.ingest import corpus as corpus_mod
from proselign.ingest.corpus import find_files, load_documents, normalize
from proselign.ingest.stylometry import describe, measure, split_sentences
from proselign.ingest.train import build_voice


def test_find_files_filters_and_skips_hidden(corpus: Path):
    names = sorted(p.name for p in find_files([corpus]))
    assert names == ["dup.txt", "essay0.md", "essay1.md", "essay2.md", "memo.txt"]


def test_find_files_accepts_explicit_file_and_rejects_missing(corpus: Path):
    png = corpus / "image.png"
    assert find_files([png]) == [png]
    with pytest.raises(UsageError):
        find_files([corpus / "missing"])


def test_load_documents_dedupes_and_skips_huge(corpus: Path, monkeypatch):
    docs = load_documents([corpus])
    assert len(docs) == 4  # dup.txt has the same bytes as memo.txt
    monkeypatch.setattr(corpus_mod, "MAX_FILE_BYTES", 10)
    assert not load_documents([corpus])


def test_normalize_strips_non_author_material():
    raw = "---\ntitle: x\n---\nMine.\r\n\n> quoted\n\n```\ncode\n```\n\n\n\nAlso mine.\n"
    assert normalize(raw) == "Mine.\n\nAlso mine."


def test_measure_and_describe():
    text = (
        "I don't think so. You can't be serious! We'll see what's next; maybe nothing.\n\n"
        "It was decided by the board. Short one."
    )
    metrics = measure(text)
    assert metrics["sentences"] == 5
    assert metrics["paragraphs"] == 2
    assert metrics["contractions_per_1k"] > 100
    assert metrics["passive_per_1k"] > 0
    assert metrics["punctuation_per_1k"]["semicolons"] > 0
    style = describe(metrics)
    assert "conversational" in style["tone"]
    assert style["sentence_length"] == "short"
    assert style["paragraphing"] == "compact"
    assert style["uses_contractions"] is True


def test_measure_empty_and_long_sentences():
    assert measure("")["sentence_length_words"]["mean"] == 0.0
    long_text = " ".join(["extraordinarily"] * 30) + "."
    style = describe(measure(long_text))
    assert style["sentence_length"] == "long"
    assert "technical" in style["tone"]
    assert describe(measure(""))["paragraphing"] == "extended"


def test_split_sentences():
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_build_voice_and_append(corpus: Path, tmp_path: Path):
    voice = build_voice("Bryan", [corpus])
    assert voice.corpus_summary["documents"] == 4
    assert voice.style
    assert voice.metrics["words"] > 200
    extra = tmp_path / "extra.md"
    extra.write_text("A brand new note with different words entirely. " * 10, "utf-8")
    grown = build_voice("Bryan", [extra], base=voice)
    assert grown.corpus_summary["documents"] == 5


def test_build_voice_rejects_thin_corpus(tmp_path: Path):
    (tmp_path / "tiny.md").write_text("Too short.", "utf-8")
    with pytest.raises(UsageError, match="only"):
        build_voice("x", [tmp_path])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UsageError, match="no readable"):
        build_voice("x", [empty])


def test_load_documents_reads_word_and_pdf_and_reports_skips(tmp_path: Path, monkeypatch):
    from .conftest import SAMPLE_PARAGRAPHS  # pylint: disable=import-outside-toplevel
    from .docbuilders import (  # pylint: disable=import-outside-toplevel
        make_cfb,
        make_docx,
        make_pdf,
        make_word97,
        para,
        text_page,
    )

    (tmp_path / "a.docx").write_bytes(make_docx("".join(para(p) for p in SAMPLE_PARAGRAPHS)))
    (tmp_path / "b.doc").write_bytes(make_cfb(make_word97([("\r".join(SAMPLE_PARAGRAPHS), True)])))
    (tmp_path / "c.pdf").write_bytes(make_pdf([text_page(b"A short PDF.")]))
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 nothing here")
    (tmp_path / "blank.md").write_text("\n\n", "utf-8")
    skipped: list[tuple[str, str]] = []
    docs = load_documents([tmp_path], lambda path, why: skipped.append((path.name, why)))
    assert sorted(d.path.name for d in docs) == ["a.docx", "b.doc", "c.pdf"]
    assert sorted(skipped) == [
        ("blank.md", "no text"),
        ("broken.pdf", "PDF has no document catalog"),
    ]

    monkeypatch.setattr(corpus_mod, "MAX_FILE_BYTES", 10)
    skipped.clear()
    load_documents([tmp_path / "a.docx"], lambda path, why: skipped.append(why))
    assert skipped == ["larger than 0 MB"]


def test_mailbox_messages_count_as_documents(tmp_path: Path, monkeypatch):
    from .conftest import SAMPLE_PARAGRAPHS  # pylint: disable=import-outside-toplevel

    box = "".join(
        f"From me@example.com Mon Jan  5 10:00:00 2026\nSubject: {i}\n\n{text}\n"
        for i, text in enumerate(SAMPLE_PARAGRAPHS)
    )
    (tmp_path / "Sent.mbox").write_text(box, "utf-8")
    (tmp_path / "note.md").write_text("\n\n".join(SAMPLE_PARAGRAPHS), "utf-8")
    docs = load_documents([tmp_path])
    assert [(d.path.name, d.parts) for d in docs] == [
        ("Sent.mbox", len(SAMPLE_PARAGRAPHS)),
        ("note.md", 1),
    ]
    voice = build_voice("Bryan", [tmp_path])
    assert voice.corpus_summary["documents"] == len(SAMPLE_PARAGRAPHS) + 1
    assert len(voice.sources) == 2

    monkeypatch.setattr(corpus_mod, "MAX_FILE_BYTES", 10)
    assert [d.path.name for d in load_documents([tmp_path])] == ["Sent.mbox"]
    monkeypatch.setattr(corpus_mod, "MAX_MAILBOX_BYTES", 10)
    skipped: list[str] = []
    assert not load_documents([tmp_path / "Sent.mbox"], lambda path, why: skipped.append(why))
    assert skipped == ["larger than 0 MB"]
