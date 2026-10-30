from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.core.config import Settings
from app.models.knowledge_document import KnowledgeDocument
from app.services.chunking import DocumentChunk, chunk_sections
from app.services.document_extraction import DocumentExtractor
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.file_storage import FileStorage
from app.services.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class ProcessingResult:
    chunk_count: int
    embedding_model: str


class DocumentProcessor:
    def __init__(
        self,
        storage: FileStorage,
        extractor: DocumentExtractor,
        embedder: SentenceTransformerEmbedder,
        vector_store: ChromaVectorStore,
        settings: Settings,
    ) -> None:
        self.storage = storage
        self.extractor = extractor
        self.embedder = embedder
        self.vector_store = vector_store
        self.settings = settings

    def process(self, document: KnowledgeDocument) -> ProcessingResult:
        if not document.storage_path or not document.file_extension or not document.original_filename:
            raise ValueError("Document storage metadata is incomplete")
        sections = self.extractor.extract(Path(document.storage_path), document.file_extension)
        chunks: list[DocumentChunk] = chunk_sections(
            sections,
            UUID(str(document.id)),
            document.original_filename,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document produced no usable text chunks")
        embeddings: list[list[float]] = []
        batch_size = self.embedder.batch_size
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            embeddings.extend(self.embedder.embed([chunk.text for chunk in batch]))
        self.vector_store.replace_document_chunks(str(document.id), chunks, embeddings)
        return ProcessingResult(len(chunks), self.embedder.model_name)

    def delete_vectors(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
