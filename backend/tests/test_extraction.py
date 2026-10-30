from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument

from app.services.document_extraction import DocumentExtractor, ExtractionError


def test_extracts_pdf_text_and_preserves_page_number(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Quarterly report contents")
    pdf.save(pdf_path)
    pdf.close()

    sections = DocumentExtractor(1000).extract(pdf_path, ".pdf")

    assert sections[0].text.strip() == "Quarterly report contents"
    assert sections[0].page_number == 1
    assert sections[0].source_location == "page 1"


def test_extracts_docx_paragraphs_and_table_content(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    document = DocxDocument()
    document.add_paragraph("DOCX overview")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    document.save(docx_path)

    sections = DocumentExtractor(1000).extract(docx_path, ".docx")

    assert "DOCX overview" in sections[0].text
    assert "Name | Value" in sections[0].text
    assert sections[0].page_number is None


@pytest.mark.parametrize("extension", [".txt", ".md", ".markdown"])
def test_extracts_text_and_markdown(tmp_path: Path, extension: str) -> None:
    file_path = tmp_path / f"sample{extension}"
    file_path.write_bytes(b"# Heading\n\nMeaningful body text.")

    sections = DocumentExtractor(1000).extract(file_path, extension)

    assert sections[0].text == "# Heading\n\nMeaningful body text."
    assert sections[0].source_location == "document"


def test_rejects_empty_or_unextractable_document(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text(" \n\t ", encoding="utf-8")

    with pytest.raises(ExtractionError, match="no extractable text"):
        DocumentExtractor(1000).extract(file_path, ".txt")


def test_enforces_extracted_text_limit(tmp_path: Path) -> None:
    file_path = tmp_path / "large.txt"
    file_path.write_text("x" * 30, encoding="utf-8")

    with pytest.raises(ExtractionError, match="configured document limit"):
        DocumentExtractor(20).extract(file_path, ".txt")
