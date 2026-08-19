from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.common import DocumentParseError
from parsers.html_parser import HTMLParser
from parsers.pdf_parser import PDFParser
from retrieval.errors import IndexCorruptionError
from retrieval.index_manager import IndexBuilder, IndexManager


def _minimal_text_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(output)


def test_html_parser_extracts_visible_text_and_metadata(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    html = source / "rules.html"
    html.write_text(
        "<html><head><title>Rules</title><script>secret()</script></head>"
        "<body><h1>Permit Rules</h1><p>Applications require a receipt.</p></body></html>",
        encoding="utf-8",
    )
    chunks = HTMLParser(allowed_root=source).parse(html)
    assert chunks[0]["metadata"]["title"] == "Rules"
    assert "Applications require a receipt." in chunks[0]["text"]
    assert "secret" not in chunks[0]["text"]


def test_pdf_parser_extracts_text_and_page_metadata(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    pdf = source / "rules.pdf"
    pdf.write_bytes(_minimal_text_pdf("Permit hours are nine to four."))
    chunks = PDFParser(allowed_root=source).parse(pdf)
    assert "Permit hours" in chunks[0]["text"]
    assert chunks[0]["metadata"]["page"] == 1
    assert chunks[0]["metadata"]["source_type"] == "pdf"


def test_parser_rejects_invalid_pdf_limits_and_path_traversal(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    invalid = source / "bad.pdf"
    invalid.write_bytes(b"not a PDF")
    with pytest.raises(DocumentParseError):
        PDFParser(allowed_root=source).parse(invalid)

    large = source / "large.pdf"
    large.write_bytes(_minimal_text_pdf("x" * 2_000))
    with pytest.raises(DocumentParseError, match="max_file_bytes"):
        PDFParser(allowed_root=source, max_file_bytes=1_024).parse(large)

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(_minimal_text_pdf("outside"))
    with pytest.raises(DocumentParseError, match="escapes"):
        PDFParser(allowed_root=source).parse(source / ".." / "outside.pdf")


def test_index_rebuild_is_idempotent_and_detects_corruption(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "fixture.html").write_text(
        "<html><body><p>Permit applications require a written receipt.</p></body></html>",
        encoding="utf-8",
    )
    index = tmp_path / "index"
    builder = IndexBuilder(sources, index)
    first = builder.build()
    first_manifest = (index / "manifest.json").read_bytes()
    second = builder.build()
    assert first == second
    assert (index / "manifest.json").read_bytes() == first_manifest
    assert IndexManager(index).retrieve("written receipt")

    bm25 = index / "bm25" / "index.json"
    payload = json.loads(bm25.read_text(encoding="utf-8"))
    payload["documents"][0]["text"] = "tampered"
    bm25.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IndexCorruptionError):
        IndexManager(index)
