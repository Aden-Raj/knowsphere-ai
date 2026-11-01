"""LangGraph composition for the existing deterministic and generated RAG stages."""

from collections.abc import Callable, Hashable
from typing import TypeVar, cast

from langgraph.graph import END, START, StateGraph

from app.services.rag.pipeline import RAGPipeline
from app.services.rag.state import QueryAnalysisResult, RAGGraphState

_T = TypeVar("_T")


class RAGGraphStageError(RuntimeError):
    """Safe graph failure that reports stage metadata without leaking exception details."""

    category = "stage_failure"

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"RAG graph stage failed: {stage}.")


def _run_stage(stage: str, operation: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    try:
        return operation(*args, **kwargs)
    except Exception:
        raise RAGGraphStageError(stage) from None


def _route_after_analysis(state: RAGGraphState) -> str:
    """Route existing early decisions directly to finalization; otherwise generate."""
    return "finalize" if state.get("query_analysis") is not None else "generate"


def build_rag_graph(pipeline: RAGPipeline):
    """Build a compiled graph that delegates all RAG behavior to ``RAGPipeline``."""

    def retrieve_node(state: RAGGraphState) -> dict[str, object]:
        chunks, selected, context, metadata = _run_stage(
            "retrieve", pipeline.retrieve_context,
            state["question"], state.get("document_ids"), state.get("top_k"),
        )
        return {
            "retrieved_chunks": chunks,
            "selected_chunks": selected,
            "assembled_context": context,
            "retrieval_metadata": metadata,
        }

    def analyze_node(state: RAGGraphState) -> dict[str, object]:
        decision = _run_stage(
            "analyze", pipeline.analyze_query,
            state["question"], state.get("selected_chunks", []), state.get("retrieval_only", False),
        )
        if decision is None:
            return {"query_analysis": None}
        answer, citations, status = decision
        analysis: QueryAnalysisResult = {
            "answer": answer,
            "citations": citations,
            "generation_status": status,
        }
        return {"query_analysis": analysis}

    def generate_node(state: RAGGraphState) -> dict[str, object]:
        answer, citations, status = _run_stage(
            "generate", pipeline.generate_response,
            state["question"], state.get("selected_chunks", []), state.get("assembled_context", ""),
        )
        citation_ids = [
            citation["chunk_id"]
            for citation in citations
            if isinstance(citation.get("chunk_id"), str)
        ]
        return {
            "generated_answer": answer,
            "citation_chunk_ids": citation_ids,
            "validated_citations": citations,
            "generation_status": status,
            "error_category": status if status in {"provider_error", "invalid_response"} else None,
        }

    def finalize_node(state: RAGGraphState) -> dict[str, object]:
        analysis = state.get("query_analysis")
        if analysis is not None:
            answer = analysis["answer"]
            citations = analysis["citations"]
            status = analysis["generation_status"]
        else:
            answer = state.get("generated_answer")
            citations = state.get("validated_citations", [])
            status = state.get("generation_status", "invalid_response")

        result = _run_stage(
            "finalize", pipeline.build_result,
            state.get("retrieved_chunks", []),
            state.get("selected_chunks", []),
            state.get("assembled_context", ""),
            state.get("retrieval_metadata", {}),
            answer,
            citations,
            status,
        )
        provider = getattr(pipeline.provider, "name", None)
        model = getattr(pipeline.provider, "model", None)
        return {
            "provider": provider,
            "model": model,
            "citation_chunk_ids": [
                citation["chunk_id"]
                for citation in citations
                if isinstance(citation.get("chunk_id"), str)
            ],
            "validated_citations": citations,
            "generation_status": status,
            "error_category": status if status in {"provider_error", "invalid_response"} else None,
            "final_result": result,
        }

    builder = StateGraph(RAGGraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "analyze")
    builder.add_conditional_edges(
        "analyze",
        _route_after_analysis,
        cast(dict[Hashable, str], {"generate": "generate", "finalize": "finalize"}),
    )
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
