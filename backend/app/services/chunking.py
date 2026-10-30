from dataclasses import dataclass
from hashlib import sha256
from uuid import UUID, uuid5

from app.services.document_extraction import ExtractedSection
from app.services.text_processing import clean_text


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    document_id: str
    index: int
    text: str
    source_filename: str
    page_number: int | None
    source_location: str


def chunk_sections(
    sections: list[ExtractedSection],
    document_id: UUID,
    source_filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk settings require size > 0 and overlap between 0 and size")
    chunks: list[DocumentChunk] = []
    for section in sections:
        text = clean_text(section.text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary_start = start + max(1, chunk_size // 2)
                boundary = max(text.rfind("\n", boundary_start, end), text.rfind(" ", boundary_start, end))
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                first_line = text.count("\n", 0, start) + 1
                last_line = text.count("\n", 0, end) + 1
                location = section.source_location or "document"
                if section.page_number is None and "lines" in location:
                    location = f"{location}; lines {first_line}-{last_line}"
                elif section.page_number is None and location == "document":
                    location = f"lines {first_line}-{last_line}"
                index = len(chunks)
                stable_key = f"{index}:{sha256(chunk_text.encode('utf-8')).hexdigest()}"
                chunk_id = str(uuid5(document_id, stable_key))
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=str(document_id),
                        index=index,
                        text=chunk_text,
                        source_filename=source_filename,
                        page_number=section.page_number,
                        source_location=location,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - chunk_overlap)
            while start < end and text[start].isspace():
                start += 1
    return chunks
