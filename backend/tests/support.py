from uuid import UUID
from pathlib import Path

from app.services.chunking import chunk_sections
from app.services.document_extraction import DocumentExtractor
from app.services.document_processor import ProcessingResult


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.vectors: dict[str, list[str]] = {}
        self.fail_next = False

    def process(self, document) -> ProcessingResult:
        key = str(document.id)
        self.calls[key] = self.calls.get(key, 0) + 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated processing issue")
        sections = DocumentExtractor(100_000).extract(
            Path(document.storage_path), document.file_extension
        )
        chunks = chunk_sections(sections, UUID(key), document.original_filename, 60, 10)
        self.vectors[key] = [chunk.id for chunk in chunks]
        return ProcessingResult(chunk_count=len(chunks), embedding_model="test-embedder")

    def delete_vectors(self, document_id: str) -> None:
        self.vectors.pop(document_id, None)
