# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Text extraction from .docx, .doc (Word 97 and 95) and .pdf documents."""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import pytest

from proselign.ingest import formats as formats_mod
from proselign.ingest.formats import ExtractError
from proselign.ingest.formats import base as base_mod
from proselign.ingest.formats import extract_parts, extract_text
from proselign.ingest.formats import mbox as mbox_mod
from proselign.ingest.formats.doc import doc_text
from proselign.ingest.formats.docx import docx_text
from proselign.ingest.formats.mbox import html_text, mbox_messages, mbox_text
from proselign.ingest.formats.ole import CompoundFile
from proselign.ingest.formats.pdf import pdf_text
from proselign.ingest.formats.pdfparse import Document, Keyword, Name, Parser, Ref

from .docbuilders import (
    SIMPLE_FONT,
    make_cfb,
    make_docx,
    make_pdf,
    make_word95,
    make_word97,
    para,
    text_page,
)

# --- dispatch ---------------------------------------------------------------


def test_plain_text_is_decoded_as_utf8():
    assert extract_text(Path("a.md"), "café".encode()) == "café"


def test_unexpected_parser_errors_become_extract_errors(monkeypatch):
    def broken(_data):
        raise IndexError("boom")

    monkeypatch.setitem(formats_mod._EXTRACTORS, ".pdf", broken)  # pylint: disable=protected-access
    with pytest.raises(ExtractError, match="unreadable .pdf file"):
        extract_text(Path("x.PDF"), b"")


def test_inflate_refuses_bombs_and_garbage():
    bomb = zlib.compress(b"\0" * 10_000)
    with pytest.raises(ExtractError, match="beyond the size limit"):
        base_mod.inflate(bomb, limit=100)


def test_inflate_refuses_garbage():
    with pytest.raises(ExtractError, match="corrupt compressed"):
        base_mod.inflate(b"not zlib")


# --- docx -------------------------------------------------------------------


def test_docx_paragraphs_runs_tabs_and_breaks():
    body = (
        para("Hello, ", "world.") + "<w:p><w:r><w:t>a</w:t><w:tab/><w:t>b</w:t><w:br/></w:r></w:p>"
    )
    assert docx_text(make_docx(body)) == "Hello, world.\n\na b"


def test_docx_ignores_deleted_text_and_field_codes():
    body = (
        "<w:p><w:del><w:r><w:delText>gone</w:delText></w:r></w:del>"
        "<w:r><w:instrText>PAGE</w:instrText></w:r><w:r><w:t>kept</w:t></w:r></w:p>"
    )
    assert docx_text(make_docx(body)) == "kept"


def test_docx_text_box_paragraph_is_separate():
    box = f"<w:r><w:pict><w:txbxContent>{para('In the box.')}</w:txbxContent></w:pict></w:r>"
    body = f"<w:p><w:r><w:t>Outside.</w:t></w:r>{box}</w:p>"
    assert docx_text(make_docx(body)) == "In the box.\n\nOutside."


def test_docx_refuses_dtd_entity_expansion():
    prolog = '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa">]>'
    data = make_docx(para("&a;"), prolog=prolog)
    with pytest.raises(ExtractError, match="DTD"):
        docx_text(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"not a zip", "corrupt .docx"),
        (make_docx(para("x"), part="word/other.xml"), "no word/document.xml"),
    ],
)
def test_docx_rejects_non_documents(data, message):
    with pytest.raises(ExtractError, match=message):
        docx_text(data)


def test_docx_rejects_malformed_xml():
    data = make_docx("<w:p>")
    with pytest.raises(ExtractError, match="malformed"):
        docx_text(data)


def test_docx_rejects_oversized_part(monkeypatch):
    import proselign.ingest.formats.docx as docx_mod  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(docx_mod, "MAX_OUTPUT_BYTES", 10)
    data = make_docx(para("much more than ten bytes"))
    with pytest.raises(ExtractError, match="too large"):
        docx_text(data)


def test_docx_rejects_deep_nesting():
    deep = "<w:sdt>" * 5000 + para("x") + "</w:sdt>" * 5000
    data = make_docx(deep)
    with pytest.raises(ExtractError):
        docx_text(data)


# --- OLE2 and Word ------------------------------------------------------------


def test_cfb_reads_regular_and_mini_streams():
    big, small = b"B" * 5000, b"small stream"
    container = CompoundFile(make_cfb({"Big": big, "Small": small}))
    assert container.stream("Big") == big
    assert container.stream("Small") == small


def test_cfb_missing_stream():
    container = CompoundFile(make_cfb({"Present": b"x"}))
    with pytest.raises(ExtractError, match="no 'Absent' stream"):
        container.stream("Absent")


def test_cfb_rejects_non_ole():
    with pytest.raises(ExtractError, match="not an OLE2"):
        CompoundFile(b"PK" + b"\0" * 600)


def test_cfb_rejects_odd_sector_sizes():
    data = bytearray(make_cfb({"S": b"x"}))
    struct.pack_into("<H", data, 0x1E, 7)
    raw = bytes(data)
    with pytest.raises(ExtractError, match="sector size"):
        CompoundFile(raw)


def test_cfb_detects_looping_chains():
    data = bytearray(make_cfb({"Big": b"B" * 5000}))  # regular sectors, not the mini stream
    fat_sid = struct.unpack_from("<I", data, 0x4C)[0]
    fat_offset = (fat_sid + 1) * 512
    struct.pack_into("<I", data, fat_offset + 4, 0)  # sector 1 -> sector 0 -> 1 ...
    container = CompoundFile(bytes(data))
    with pytest.raises(ExtractError, match="chain"):
        container.stream("Big")


def test_cfb_out_of_range_sector():
    data = bytearray(make_cfb({"S": b"x"}))
    struct.pack_into("<I", data, 0x30, 9999)  # directory starts past the end
    raw = bytes(data)
    with pytest.raises(ExtractError):
        CompoundFile(raw)


def test_word97_compressed_and_unicode_pieces():
    streams = make_word97([("Plain cp1252 text’s fine.\r", True), ("Unicode Łódź.\r", False)])
    assert doc_text(make_cfb(streams)) == "Plain cp1252 text’s fine.\n\nUnicode Łódź."


def test_word97_fields_keep_results_and_drop_instructions():
    text = (
        'Date: \x13 DATE \\@ "MMMM" \x14July\x15. Nested \x13 IF \x13 X \x14x\x15 \x14shown\x15.\r'
    )
    assert doc_text(make_cfb(make_word97([(text, True)]))) == "Date: July. Nested shown."


def test_word97_special_characters():
    text = "non\x1ebreaking soft\x1fhyphen\xa0space\x01pic\x07cell\x0bline\x0cpage"
    assert doc_text(make_cfb(make_word97([(text, True)]))) == (
        "non-breaking softhyphen spacepic\n\ncell\n\nline\n\npage"
    )


def test_word97_zero_table_stream():
    streams = make_word97([("Zero table.", True)], flags=0)
    assert doc_text(make_cfb(streams)) == "Zero table."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"flags": 0x0200 | 0x0100}, "password-protected"),
        ({"nfib": 0x0065}, "Word 95 and older"),
    ],
)
def test_word97_refusals(kwargs, message):
    data = make_cfb(make_word97([("x", True)], **kwargs))
    with pytest.raises(ExtractError, match=message):
        doc_text(data)


@pytest.mark.parametrize(
    ("word", "message"),
    [(b"\0" * 64, "not a Word document"), (b"\xec\xa5", "truncated")],
)
def test_word97_bad_identity_and_truncation(word, message):
    data = make_cfb({"WordDocument": word})
    with pytest.raises(ExtractError, match=message):
        doc_text(data)


def test_word97_piece_outside_document():
    streams = make_word97([("text", True)])
    word = bytearray(streams["WordDocument"])
    streams["WordDocument"] = bytes(word[:-2])
    data = make_cfb(streams)
    with pytest.raises(ExtractError, match="outside"):
        doc_text(data)


def test_word97_corrupt_piece_table():
    streams = make_word97([("text", True)])
    streams["1Table"] = b"\x07" + streams["1Table"][1:]
    data = make_cfb(streams)
    with pytest.raises(ExtractError, match="piece table is corrupt"):
        doc_text(data)


def test_word97_piece_table_out_of_range():
    streams = make_word97([("text", True)])
    streams["1Table"] = b""
    data = make_cfb(streams)
    with pytest.raises(ExtractError, match="out of range"):
        doc_text(data)


def test_word95_plain_and_fast_saved():
    text = "Written in Word 95.\rSecond paragraph."
    expected = "Written in Word 95.\n\nSecond paragraph."
    assert doc_text(make_cfb(make_word95(text))) == expected
    assert doc_text(make_cfb(make_word95(text, fast_saved=True))) == expected


def _short_word95() -> dict:
    streams = make_word95("some text")
    streams["WordDocument"] = streams["WordDocument"][:-3]
    return streams


@pytest.mark.parametrize(
    ("streams", "message"),
    [
        (make_word95("x", flags=0x0100), "password-protected"),
        (_short_word95(), "outside"),
        ({"WordDocument": b"\xdc\xa5" + b"\0" * 40}, "truncated"),
    ],
)
def test_word95_refusals(streams, message):
    data = make_cfb(streams)
    with pytest.raises(ExtractError, match=message):
        doc_text(data)


# --- PDF parser ---------------------------------------------------------------


def _values(data: bytes, refs: bool = True) -> list:
    parser = Parser(data, refs)
    values = []
    while not parser.at_end():
        values.append(parser.value())
    return values


def test_pdf_parser_value_types():
    values = _values(b"<< /A [1 -2.5 .5 true null] /B#20C (s) >> 12 0 R /N % comment\n RG")
    assert values == [
        {"A": [1, -2.5, 0.5, True, None], "B C": b"s"},
        Ref(12),
        Name("N"),
        Keyword("RG"),
    ]


def test_pdf_literal_string_escapes():
    raw = rb"(a\(b\) (nested) \n\t\\ \101\60x \q end\
cont)"
    assert _values(raw) == [b"a(b) (nested) \n\t\\ A0x q endcont"]


def test_pdf_hex_strings_and_content_mode():
    assert _values(b"<48 65 6>", refs=False) == [b"He`"]
    assert _values(b"1 0 0 1 0 0 cm", refs=False)[-1] == Keyword("cm")


@pytest.mark.parametrize(
    "bad", [b"(unterminated", b"<abc", b"<zz>", b"<< 1 2 >>", b"[" * 100 + b"]" * 100]
)
def test_pdf_parser_rejects_malformed(bad):
    with pytest.raises(ExtractError):
        _values(bad)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"hello", "not a PDF"),
        (make_pdf([text_page(b"x")], extra=[b"<< /Encrypt 1 0 R >>"]), "encrypted"),
    ],
)
def test_pdf_document_refusals(data, message):
    with pytest.raises(ExtractError, match=message):
        Document(data)


def test_pdf_without_catalog():
    document = Document(b"%PDF-1.4\n1 0 obj << /Type /Page >> endobj")
    with pytest.raises(ExtractError, match="catalog"):
        document.pages()


def test_pdf_reference_loops_are_bounded():
    doc = Document(b"%PDF-1.4\n1 0 obj 1 0 R endobj")
    with pytest.raises(ExtractError, match="too long"):
        doc.resolve(Ref(1))


# --- PDF text -------------------------------------------------------------------


def test_pdf_lines_join_into_paragraphs():
    page = text_page(b"The first line of a para-", b"graph wraps here.", b"", b"A new paragraph.")
    page = page.replace(b"(A new", b"0 -14 Td (A new")
    assert (
        pdf_text(make_pdf([page]))
        == "The first line of a paragraph wraps here.\n\nA new paragraph."
    )


def test_pdf_uncompressed_and_multiple_pages():
    pdf = make_pdf([text_page(b"Page one."), text_page(b"Page two.")], compress=False)
    assert pdf_text(pdf) == "Page one.\n\nPage two."


def test_pdf_chunks_on_one_line_join_by_position():
    # "infor" is 5 glyphs * 500/1000 * 12pt = 30pt wide: a chunk at +30 continues
    # the word, one at +40 starts a new word.
    page = b"BT /F1 12 Tf 72 720 Td (infor) Tj 30 0 Td (mation) Tj 42 0 Td (system) Tj ET"
    assert pdf_text(make_pdf([page])) == "information system"


def test_pdf_tj_arrays_kerning_and_spacing_operators():
    page = (
        b"BT /F1 12 Tf 2 Tc 1 Tw 14 TL 1 0 0 1 72 720 Tm [(Ker)20(ned)-600(gap)] TJ "
        b"0 -14 TD (next) Tj 1 2 (quoted) \" (moved) ' ET"
    )
    assert pdf_text(make_pdf([page])) == "Kerned gap next quoted moved"


def test_pdf_type0_font_with_tounicode_cmap():
    cmap = (
        b"begincmap 1 beginbfchar <0001> <0048> endbfchar "
        b"2 beginbfrange <0002> <0003> <0069> <0004> <0005> [<0021> <2014>] endbfrange endcmap"
    )
    fonts = {
        "F1": b"<< /Type /Font /Subtype /Type0 /ToUnicode 2 0 R /DescendantFonts [3 0 R] >>",
    }
    extra = [
        b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream",
        b"<< /W [1 [600 500] 3 5 400] >>",
    ]
    page = b"BT /F1 10 Tf 72 720 Td <00010002000300040005> Tj ET"
    assert pdf_text(make_pdf([page], fonts, extra=extra)) == "Hij!—"


def test_pdf_differences_and_macroman_encodings():
    differences = SIMPLE_FONT.replace(
        b"/WinAnsiEncoding",
        b"<< /BaseEncoding /MacRomanEncoding"
        b" /Differences [65 /quoteright /B /uni00E9 /seven /bogus /uniZZZZ] >>",
    )
    # A-F come from /Differences (unknown glyph names map to nothing); G and
    # 0x8E fall back to the MacRoman base encoding.
    page = b"BT /F1 12 Tf 72 720 Td (ABCDEFG\x8e) Tj ET"
    assert pdf_text(make_pdf([page], {"F1": differences})) == "’Bé7Gé"


def test_pdf_object_streams_and_ascii_filters():
    font_dict = SIMPLE_FONT
    header = b"7 0 "  # object 7 lives inside object stream 1
    objstm = header + font_dict
    hex_body = base64.b16encode(
        zlib.compress(b"BT /F2 12 Tf 72 720 Td (In an object stream.) Tj ET")
    )
    a85_body = base64.a85encode(b"BT /F2 12 Tf 72 700 Td (ASCII85 too.) Tj ET") + b"~>"
    objstm_dict = b"<< /Type /ObjStm /N 1 /First %d /Length %d >>" % (len(header), len(objstm))
    pdf = (
        b"%PDF-1.5\n"
        b"1 0 obj " + objstm_dict + b"\nstream\n" + objstm + b"\nendstream endobj\n"
        b"2 0 obj << /Filter [/AHx /FlateDecode] /Length 9 0 R >>\nstream\n"
        + hex_body
        + b">\nendstream endobj\n"
        b"3 0 obj << /Filter /A85 >>\nstream\r\n" + a85_body + b"\r\nendstream endobj\n"
        b"4 0 obj << /Type /Page /Contents [2 0 R 3 0 R]"
        b" /Resources << /Font << /F2 7 0 R >> >> >> endobj\n"
        b"5 0 obj << /Type /Pages /Kids [4 0 R] >> endobj\n"
        b"6 0 obj << /Type /Catalog /Pages 5 0 R >> endobj\n"
    )  # fmt: skip
    assert pdf_text(pdf) == "In an object stream. ASCII85 too."


def test_pdf_unsupported_filters_and_images_leave_no_text():
    page = b"<< /Filter /DCTDecode /Length 4 >>\nstream\nJPEG\nendstream"
    pdf = make_pdf([text_page(b"x")]).replace(b"/Contents 3 0 R", b"/Contents 2 0 R")
    pdf = pdf.replace(b"2 0 obj\n", b"2 0 obj\n" + page + b"\nendobj\n99 0 obj\n", 1)
    with pytest.raises(ExtractError, match="no extractable text"):
        pdf_text(pdf)


def test_pdf_inline_images_and_damaged_content_are_survived():
    page = (
        b"BT /F1 12 Tf 72 720 Td (Before.) Tj ET BI /W 1 /H 1 ID \x00\xffEI "
        b"BT 72 700 Td (After.) Tj ET ("
    )
    assert pdf_text(make_pdf([page])) == "Before. After."


def test_pdf_page_tree_inherits_resources_and_ignores_cycles():
    pdf = make_pdf([text_page(b"Inherited font.")])
    pdf = pdf.replace(b"/Kids [3 0 R]", b"/Kids [3 0 R 4 0 R]")  # 4 is the Pages node itself
    assert pdf_text(pdf) == "Inherited font."


# --- mbox -------------------------------------------------------------------


def _mail(body: str, ctype: str = "text/plain", extra: str = "") -> str:
    return (
        "From author@example.com Mon Jan  5 10:00:00 2026\n"
        f"From: Author <author@example.com>\nSubject: s\n{extra}"
        f'Content-Type: {ctype}; charset="utf-8"\n\n{body}\n'
    )


SIGNED = "\n\nThanks,\nAuthor Name\nChief Example Officer\n"


def test_mbox_splits_messages_and_drops_quotes_and_signatures():
    box = (
        _mail("First reply here." + SIGNED + "\nOn Mon, Jan 5, 2026, Someone wrote:\n> their words")
        + _mail("Second message.\n>From the escaped line.\n> quoted inline\nStill mine." + SIGNED)
        + _mail("Third.\n\n-- \nsig block")
    ).encode()
    assert mbox_messages(box) == [
        "First reply here.",
        "Second message.\nFrom the escaped line.\nStill mine.",
        "Third.",
    ]
    assert mbox_text(box).startswith("First reply here.\n\nSecond message.")


@pytest.mark.parametrize(
    "tail",
    [
        "-----Original Message-----\nFrom: x",
        "---------- Forwarded message ---------\nFrom: x",
        "Begin forwarded message:\n\nx",
        "From: Other <o@example.com>\nSent: Monday\nTo: me",
        "________________________________\nFrom: x",
        "Sent from my iPhone",
        "-------- Forwarded Message --------\nFrom: x",
        "\u2500" * 20 + "\nBryan is inviting you to a scheduled Zoom meeting.\nMeeting ID: 1",
        "Bryan is inviting you to a scheduled Zoom meeting.\nPasscode: 2",
        "Microsoft Teams meeting\nJoin on your computer",
        "On Mon, Jan 5, 2026 at 10:00 AM Someone <s@example.com>\nwrote:\nx",
    ],
)
def test_mbox_cuts_at_the_start_of_someone_elses_words(tail):
    assert mbox_messages(_mail(f"Mine.\n\n{tail}").encode()) == ["Mine."]


def test_mbox_removes_links_and_image_placeholders():
    body = "See [cid:image001.png@01] the page<https://example.com/x> or https://example.org now."
    assert mbox_messages(_mail(body).encode()) == ["See  the page or  now."]


def test_mbox_html_body_without_quotes_styles_or_signature():
    html = (
        "<html><head><style>p{}</style></head><body><div>Hello&nbsp;there,</div>"
        "<div><br></div><p>Line <b>two</b>.</p><blockquote>quoted</blockquote>"
        '<img src="x"><div class="gmail_signature">sig</div><p>after</p></body></html>'
    )
    assert mbox_messages(_mail(html, "text/html").encode()) == ["Hello there,\n\nLine two."]
    assert html_text("<div id=divRplyFwdMsg>From: x</div>") == ""


def test_mbox_prefers_plain_over_html_and_skips_attachments():
    message = (
        "From a Mon Jan  5 10:00:00 2026\nMIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="B"\n\n'
        '--B\nContent-Type: multipart/alternative; boundary="A"\n\n'
        "--A\nContent-Type: text/plain\n\nPlain words.\n"
        "--A\nContent-Type: text/html\n\n<p>Html words.</p>\n--A--\n"
        "--B\nContent-Type: application/pdf\nContent-Transfer-Encoding: base64\n\nJVBERi0=\n--B--\n"
    )
    assert mbox_messages(message.encode()) == ["Plain words."]


def test_mbox_unknown_charset_and_empty_messages():
    odd = _mail("Caf\xe9 talk.").replace('charset="utf-8"', 'charset="x-unknown"')
    attachment_only = "From a Mon Jan  5 10:00:00 2026\nContent-Type: application/zip\n\nUEsDBA==\n"
    assert mbox_messages((odd + attachment_only).encode()) == ["Café talk."]


def test_mbox_damaged_and_oversized_messages_are_skipped(monkeypatch):
    monkeypatch.setattr(mbox_mod, "MAX_PARTS", 3)
    parts = "".join(f"--B\nContent-Type: text/plain\n\npart {i}\n" for i in range(5))
    many = f'From a Mon Jan  5 2026\nContent-Type: multipart/mixed; boundary="B"\n\n{parts}--B--\n'
    assert mbox_messages((many + _mail("Kept.")).encode()) == ["Kept."]


def test_mbox_rejects_non_mailboxes():
    with pytest.raises(ExtractError, match="not an mbox"):
        mbox_messages(b"Subject: hi\n\nbody")


def test_extract_parts_splits_only_mailboxes():
    box = (_mail("One.") + _mail("Two.")).encode()
    assert extract_parts(Path("Sent.MBOX"), box) == ["One.", "Two."]
    assert extract_parts(Path("a.txt"), b"whole file") == ["whole file"]
    assert extract_text(Path("a.mbox"), box) == "One.\n\nTwo."


def test_extract_parts_wraps_unexpected_mailbox_errors(monkeypatch):
    def broken(_data):
        raise TypeError("boom")

    monkeypatch.setattr(formats_mod, "mbox_messages", broken)
    with pytest.raises(ExtractError, match="unreadable mailbox"):
        extract_parts(Path("a.mbox"), b"From x\n")


def test_mbox_list_items_become_paragraphs():
    body = "Points:\n * first point\n   wraps here\n- second\n2) third\nNot a list - just text."
    assert mbox_messages(_mail(body).encode()) == [
        "Points:\n\n * first point\n   wraps here\n\n- second\n\n2) third\nNot a list - just text."
    ]
    assert html_text("<p>Intro</p><ul><li>one</li><li>two</li></ul>") == "Intro\n\none\n\ntwo"
