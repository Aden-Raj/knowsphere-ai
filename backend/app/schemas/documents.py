from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.knowledge_document import DocumentStatus, KnowledgeDocument


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    original_filename: str | None
    file_extension: str | None
    mime_type: str | None
    file_size_bytes: int | None
    status: DocumentStatus
    error_message: str | None
    chunk_count: int
    embedding_model: str | None
    created_at: datetime
    updated_at: datetime
    processing_completed_at: datetime | None

    @classmethod
    def from_document(cls, document: KnowledgeDocument) -> "DocumentRead":
        result = cls.model_validate(document)
        return result.model_copy(update={"mime_type": document.content_type})


class UploadFailure(BaseModel):
    filename: str | None
    detail: str


class MultipleUploadResponse(BaseModel):
    accepted: list[DocumentRead]
    rejected: list[UploadFailure]
