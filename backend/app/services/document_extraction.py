from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument


class ExtractionError(ValueError):
    """A supported file could not be converted to text."""


@dataclass(frozen=True)
class ExtractedSection:
    text: str
    page_number: int | None = None
    source_location: str | None = None


class DocumentExtractor:
    def __init__(self, max_extracted_chars: int, max_document_pages: int = 2000) -> None:
        self.max_extracted_chars = max_extracted_chars
        self.max_document_pages = max_document_pages

    def extract(self, path: Path, extension: str) -> list[ExtractedSection]:
        if extension == ".pdf":
            sections = self._extract_pdf(path)
        elif extension == ".docx":
            sections = self._extract_docx(path)
        elif extension in {".txt", ".md", ".markdown"}:
            sections = self._extract_text_file(path)
        else:
            raise ExtractionError("Unsupported document format")
        if sum(len(section.text) for section in sections) > self.max_extracted_chars:
            raise ExtractionError("Extracted content exceeds the configured document limit")
        if not any(section.text.strip() for section in sections):
            raise ExtractionError("Document contains no extractable text")
        return sections

    def _extract_pdf(self, path: Path) -> list[ExtractedSection]:
        try:
            with pymupdf.open(path) as document:
                if document.is_encrypted:
                    raise ExtractionError("Password-protected PDFs are not supported")
                if document.page_count > self.max_document_pages:
                    raise ExtractionError("PDF exceeds the configured page limit")
                sections = []
                extracted_chars = 0
                for index, page in enumerate(document):
                    page_number = index + 1
                    text = page.get_text("text")
                    extracted_chars += len(text)
                    if extracted_chars > self.max_extracted_chars:
                        raise ExtractionError("Extracted content exceeds the configured document limit")
                    sections.append(
                        ExtractedSection(text, page_number=page_number, source_location=f"page {page_number}")
                    )
                return sections
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError("PDF text extraction failed") from exc

    def _extract_docx(self, path: Path) -> list[ExtractedSection]:
        try:
            document = DocxDocument(path)
            content = []
            extracted_chars = 0

            def add_content(value: str) -> None:
                nonlocal extracted_chars
                if value.strip():
                    extracted_chars += len(value) + 1
                    if extracted_chars > self.max_extracted_chars:
                        raise ExtractionError("Extracted content exceeds the configured document limit")
                    content.append(value)

            for paragraph in document.paragraphs:
                add_content(paragraph.text)
            for table_index, table in enumerate(document.tables, start=1):
                for row_index, row in enumerate(table.rows, start=1):
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        add_content(f"[Table {table_index}, row {row_index}] " + " | ".join(cells))
            return [ExtractedSection("\n".join(content), source_location="document")]
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError("DOCX text extraction failed") from exc

    def _extract_text_file(self, path: Path) -> list[ExtractedSection]:
        try:
            data = path.read_bytes()
            if b"\x00" in data:
                raise ExtractionError("Text documents cannot contain binary data")
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("cp1252")
            return [ExtractedSection(text, source_location="document")]
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError("Text document decoding failed") from exc
