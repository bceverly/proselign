# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Rewriting .docx files in place: what is edited, what is left untouched."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from proselign.errors import EXIT_OK, EXIT_USAGE, UsageError
from proselign.ingest.formats.docx import docx_text
from proselign.rewrite import docx as docx_mod
from proselign.rewrite.docx import rewrite_docx
from proselign.rewrite.engine import Options, Rewriter, _split_units, _summarize
from proselign.rewrite.prompt import build_messages
from proselign.voice.profile import Voice

from .conftest import run
from .test_rewrite import ScriptedBackend

NS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" mc:Ignorable="w14"'
)
SENTENCE = "This plain paragraph has more than enough words to count."
BOLD = "<w:rPr><w:b/></w:rPr>"


def _p(*runs: str, ppr: str = "", extra: str = "") -> str:
    return f'<w:p w14:paraId="1">{ppr}{extra}{"".join(runs)}</w:p>'


def _r(text: str, rpr: str = "") -> str:
    return f'<w:r w:rsidR="00A1">{rpr}<w:t xml:space="preserve">{text}</w:t></w:r>'


def _docx(body: str, **parts: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8"?>\n<w:document {NS}><w:body>{body}'
            "<w:sectPr/></w:body></w:document>",
        )
        for name, xml in parts.items():
            archive.writestr(name.replace("__", "/").replace("_xml", ".xml"), xml)
        archive.writestr("word/media/image1.png", b"\x89PNG binary")
    return buffer.getvalue()


def _upper(units: list[str]) -> list[str]:
    return [u.upper().replace("<S", "<s").replace("</S", "</s") for u in units]


def _entries(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {i.filename: archive.read(i) for i in archive.infolist()}


def test_only_eligible_paragraphs_change_and_everything_else_is_byte_identical():
    body = "".join(
        [
            _p(_r(SENTENCE)),
            _p(
                _r("A Heading That Is Long Enough."),
                ppr='<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>',
            ),
            _p(_r("Too short.")),
            _p(_r("A title with plenty of words but no stop")),
            _p(_r("Tab"), "<w:r><w:tab/></w:r>", _r(" then more words follow here.")),
            _p(_r("Hidden words are never rewritten at all.", "<w:rPr><w:vanish/></w:rPr>")),
            _p('<w:hyperlink r:id="x">' + _r("Linked text goes here and on.") + "</w:hyperlink>"),
            _p(_r("Odd [[1]] markers in this one sentence.")),
            _p("stray text" + _r("Stray text sits between the runs here.")),
            "<w:p/>",
            _p(),
        ]
    )
    header = f"<w:hdr {NS}>{_p(_r('Header words that stay exactly as they are.'))}</w:hdr>"
    source = _docx(body, word__header1_xml=header)
    result, report = rewrite_docx(source, _upper)
    before, after = _entries(source), _entries(result)
    assert list(before) == list(after)
    assert [name for name in before if before[name] != after[name]] == ["word/document.xml"]
    assert SENTENCE.upper() in docx_text(result)
    assert report.rewritten == 1
    assert report.kept == {
        "headings and captions": 1,
        "not full sentences (titles, labels, table cells)": 2,
        "tabs, breaks or embedded objects": 1,
        "hidden text": 1,
        "fields, links, images or tracked changes": 1,
        "unusual content": 2,
    }
    xml = after["word/document.xml"].decode()
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>\n<w:document ' + NS)
    assert 'w14:paraId="1"' in xml


def test_formatting_survives_as_tags_and_comes_back_as_runs():
    paragraph = _p(
        _r("Revenue was "),
        _r("up sharply", BOLD),
        _r(" this quarter, "),
        _r("and", BOLD),
        _r(" "),
        _r("costs", BOLD),
        _r(" fell &amp; held."),
        ppr='<w:pPr><w:jc w:val="both"/></w:pPr>',
        extra='<w:bookmarkStart w:id="0" w:name="b"/><w:proofErr w:type="spellStart"/>',
    )
    seen: list[str] = []

    def rewrite(units):
        seen.extend(units)
        return ["We saw revenue <s1>rise sharply</s1> & costs <s1>fall</s1> <this> quarter."]

    result, report = rewrite_docx(_docx(paragraph), rewrite)
    assert seen == [
        "Revenue was <s1>up sharply</s1> this quarter, <s1>and</s1> <s1>costs</s1> fell & held."
    ]
    assert report.rewritten == 1
    xml = _entries(result)["word/document.xml"].decode()
    assert '<w:pPr><w:jc w:val="both"/></w:pPr><w:bookmarkStart w:id="0" w:name="b"/>' in xml
    assert "proofErr" not in xml
    assert f'<w:r w:rsidR="00A1">{BOLD}<w:t xml:space="preserve">rise sharply</w:t></w:r>' in xml
    assert "&amp; costs " in xml and "&lt;this&gt;" in xml


@pytest.mark.parametrize(
    "reply",
    [
        "Tags <s1>left open.",
        "Tags </s1>closed first.",
        "Nested <s1>a <s1>b</s1></s1>.",
        "Unknown <s7>style</s7> here.",
        "Zero <s0>style</s0> here.",
        "",
        SENTENCE.replace("words", "wordz") + "<s1>x</s1><s2>",
    ],
)
def test_broken_tags_keep_the_original_paragraph(reply):
    source = _docx(_p(_r("Plain words around a "), _r("bold bit", BOLD), _r(" in a sentence.")))
    result, report = rewrite_docx(source, lambda units: [reply])
    assert report.rewritten == 0
    assert report.kept == {"unchanged by the rewrite": 1}
    assert _entries(result)["word/document.xml"] == _entries(source)["word/document.xml"]


def test_footnotes_are_rewritten():
    foot = f"<w:footnotes {NS}>{_p(_r(SENTENCE))}</w:footnotes>"
    result, report = rewrite_docx(_docx(_p(_r(SENTENCE)), word__footnotes_xml=foot), _upper)
    assert SENTENCE.upper().encode() in _entries(result)["word/footnotes.xml"]
    assert report.rewritten == 2


def test_non_utf8_part_is_copied():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", b"\xff\xfe<w:document/>")
    result, report = rewrite_docx(buffer.getvalue(), _upper)
    assert report.kept == {"part not UTF-8": 1}
    assert _entries(result)["word/document.xml"] == b"\xff\xfe<w:document/>"


@pytest.mark.parametrize(
    "data,message",
    [
        (b"not a zip", "not a .docx"),
        (_docx("").replace(b"word/document.xml", b"word/documenX.xml"), "no word/document.xml"),
        (_docx('<!DOCTYPE x [<!ENTITY a "b">]>'), "DTD"),
        (_docx("<w:p><w:r></w:p></w:r></w:x>"), "unbalanced"),
    ],
)
def test_refusals(data, message):
    with pytest.raises(UsageError, match=message):
        rewrite_docx(data, _upper)


def test_size_limits(monkeypatch):
    monkeypatch.setattr(docx_mod, "MAX_PART_BYTES", 10)
    with pytest.raises(UsageError, match="too large"):
        rewrite_docx(_docx(_p(_r(SENTENCE))), _upper)


def test_numeric_entities_round_trip():
    text = "Caf&#233; and &#x2603; stay readable in this sentence."
    seen: list[str] = []
    rewrite_docx(_docx(_p(_r(text))), lambda units: seen.extend(units) or units)
    assert seen == ["Caf\u00e9 and \u2603 stay readable in this sentence."]
    assert docx_mod._unescape("a&#0;b&#x110000;c") == "abc"  # pylint: disable=protected-access


# --- engine batching --------------------------------------------------------


VOICE = Voice(display_name="Bryan")


def test_units_are_batched_and_split_back():
    backend = ScriptedBackend(["[[1]] One done.\n\n[[2]] Two done.", "Three done."])
    rewriter = Rewriter(VOICE, backend, Options(max_chunk_tokens=64))
    assert rewriter.rewrite_units(["One.", "Two.", "Three." * 60]) == [
        "One done.",
        "Two done.",
        "Three done.",
    ]
    assert len(backend.prompts) == 2
    assert "The input has 2 paragraphs" in backend.prompts[0][1].content
    assert "Format requirement" not in backend.prompts[1][1].content
    assert rewriter.report.chunks == 2


def test_failed_batch_is_retried_one_paragraph_at_a_time():
    backend = ScriptedBackend(["dropped markers", "still none", "nope", "A2.", "B2."])
    rewriter = Rewriter(VOICE, backend, Options())
    assert rewriter.rewrite_units(["A.", "B."]) == ["A2.", "B2."]
    assert rewriter.report.fallbacks == 0


def test_format_note_names_tags():
    note = build_messages(VOICE, "[[1]] a <s1>b</s1> <s2>c</s2>", 0.5)[1].content
    assert "Words wrapped in <s1>, <s2> ... </s1>, </s2>" in note


def test_damaged_markers_fall_back_to_one_at_a_time():
    backend = ScriptedBackend(
        ["[[2]] B. [[1]] A.", "[[2]] B. [[1]] A.", "[[2]] B. [[1]] A.", "A2.", "B2."]
    )
    rewriter = Rewriter(VOICE, backend, Options())
    assert rewriter.rewrite_units(["A.", "B."]) == ["A2.", "B2."]


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("[[1]] a\n\n[[2]] b", 2, ["a", "b"]),
        ("intro [[1]] a", 1, None),
        ("[[1]] a", 2, None),
        ("[[1]] a [[1]] b", 2, None),
    ],
)
def test_split_units(text, count, expected):
    assert _split_units(text, count) == expected


# --- command line -----------------------------------------------------------


def test_cli_rewrites_docx_to_output(trained, tmp_path: Path):
    source = tmp_path / "in.docx"
    source.write_bytes(_docx(_p(_r(SENTENCE.replace("words", "words—really")))))
    out = tmp_path / "out.docx"
    code, stdout, err = run(
        ["--voice", "Bryan", "--backend", "mock", str(source), "-o", str(out)], registry=trained
    )
    assert code == EXIT_OK, err
    assert not stdout
    assert "rewrote 0 paragraph(s)" in err and "unchanged by the rewrite" in err
    assert docx_text(out.read_bytes()) == docx_text(source.read_bytes())


@pytest.mark.parametrize(
    "extra,message",
    [
        ([], "needs --output"),
        (["-o", "{src}"], "must not be the input"),
        (["{src}", "-o", "x.docx"], "one .docx"),
    ],
)
def test_cli_docx_usage_errors(trained, tmp_path: Path, extra, message):
    source = tmp_path / "in.docx"
    source.write_bytes(_docx(""))
    argv = ["--voice", "Bryan", "--backend", "mock", str(source)]
    argv += [a.format(src=source, missing=tmp_path / "gone.docx") for a in extra]
    code, _, err = run(argv, registry=trained)
    assert code == EXIT_USAGE
    assert message in err


def test_cli_docx_unreadable_and_too_large(trained, tmp_path: Path, monkeypatch):
    argv = ["--voice", "Bryan", "--backend", "mock", str(tmp_path / "gone.docx"), "-o", "o.docx"]
    code, _, err = run(argv, registry=trained)
    assert (code, "cannot read" in err) == (EXIT_USAGE, True)
    source = tmp_path / "big.docx"
    source.write_bytes(_docx(""))
    monkeypatch.setattr("proselign.cli.MAX_PACKAGE_BYTES", 10)
    argv[4] = str(source)
    code, _, err = run(argv, registry=trained)
    assert (code, "too large" in err) == (EXIT_USAGE, True)


def test_cli_text_output_file(trained, tmp_path: Path):
    out = tmp_path / "out.md"
    code, stdout, _ = run(
        ["--voice", "Bryan", "--backend", "mock", "-o", str(out)], "Hello.\n", registry=trained
    )
    assert (code, stdout, out.read_text("utf-8")) == (EXIT_OK, "", "Hello.\n")


def test_summarize_caps_long_problem_lists():
    assert _summarize(["a", "b"]) == "a; b"
    assert _summarize([str(i) for i in range(8)], 3) == "0; 1; 2; and 5 more"


def test_typed_list_markers_are_kept_and_the_item_rewritten():
    marker = "<w:r><w:t>1.</w:t><w:tab/></w:r>"
    item = _p(marker, _r("Lead in", BOLD), _r(" and then the rest of the sentence."))
    bullet = _p("<w:r><w:t>\u2022</w:t><w:tab/></w:r>", _r(SENTENCE))
    refused = [
        _p("<w:r><w:t>Not a marker</w:t><w:tab/></w:r>", _r(SENTENCE)),
        _p(_r(SENTENCE), "<w:r><w:t>2.</w:t><w:tab/></w:r>"),
        _p(marker, marker, _r(SENTENCE)),
        _p("<w:r><w:tab/><w:t>1.</w:t></w:r>", _r(SENTENCE)),
    ]
    result, report = rewrite_docx(_docx(item + bullet + "".join(refused) + _p(_r("   "))), _upper)
    assert report.rewritten == 2
    assert report.kept == {"tabs, breaks or embedded objects": 4}
    xml = _entries(result)["word/document.xml"].decode()
    assert (
        f'<w:p w14:paraId="1">{marker}<w:r w:rsidR="00A1">{BOLD}<w:t xml:space="preserve">LEAD IN'
        in xml
    )
    assert "THEN THE REST OF THE SENTENCE." in xml and SENTENCE.upper() in xml
