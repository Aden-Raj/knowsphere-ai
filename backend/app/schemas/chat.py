from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    document_ids: list[UUID] | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=50)
    conversation_id: UUID | None = None

    @model_validator(mode="after")
    def non_blank_question(self):
        if not self.question.strip():
            raise ValueError("question must not be blank")
        return self


class RetrievedChunkRead(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    chunk_index: int
    page_number: int | None
    text: str
    relevance_score: float
    included_in_context: bool


class CitationRead(BaseModel):
    document_id: str
    filename: str
    page_number: int | None
    chunk_id: str
    quoted_text: str


class QueryResponse(BaseModel):
    conversation_id: UUID | None
    answer: str | None
    citations: list[CitationRead]
    retrieved_chunks: list[RetrievedChunkRead]
    provider: str | None
    model: str | None
    generation_status: str
    retrieval_metadata: dict


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    role: str
    content: str
    citations: list[dict]
    provider: str | None
    model: str | None
    generation_status: str | None
    created_at: datetime


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRead] = Field(default_factory=list)
