"""Typed state passed between KnowSphere's LangGraph RAG nodes."""

from typing import Literal, TypedDict

from app.services.rag.pipeline import RetrievedChunk

TaskType = Literal["direct", "research", "comparison", "synthesis"]


class PlannedQuery(TypedDict):
    query: str
    subject: str | None
    aspect: str


class TaskEvidence(TypedDict):
    query: str
    subject: str | None
    aspect: str
    chunk_ids: list[str]


class EvidenceCoverage(TypedDict):
    sufficient: bool
    covered_aspects: list[str]
    missing_aspects: list[str]
    evidence_counts: dict[str, int]
    adjustment_recommended: bool


class QueryAnalysisResult(TypedDict):
    """An existing deterministic early-route result from ``RAGPipeline``."""

    answer: str | None
    citations: list[dict[str, object]]
    generation_status: str


class RAGGraphState(TypedDict, total=False):
    """Per-invocation data only; never contains sessions, providers, or secrets."""

    question: str
    resolved_question: str
    resolution_status: str
    task_type: TaskType
    subqueries: list[PlannedQuery]
    adjustment_queries: list[PlannedQuery]
    task_evidence: list[TaskEvidence]
    retrieval_round: int
    evidence_coverage: EvidenceCoverage
    document_ids: list[str] | None
    top_k: int | None
    retrieval_only: bool
    query_analysis: QueryAnalysisResult | None
    retrieved_chunks: list[RetrievedChunk]
    selected_chunks: list[RetrievedChunk]
    assembled_context: str
    retrieval_metadata: dict[str, object]
    provider: str | None
    model: str | None
    generated_answer: str | None
    citation_chunk_ids: list[str]
    validated_citations: list[dict[str, object]]
    generation_status: str
    error_category: str | None
    final_result: dict[str, object]
