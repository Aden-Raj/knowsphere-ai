import logging
from uuid import UUID
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.conversation import ChatMessage, Conversation
from app.schemas.chat import ConversationRead, QueryRequest, QueryResponse
from app.services.conversation_memory import (
    CLARIFICATION_RESPONSE,
    load_recent_conversation_history,
)
from app.services.follow_up_resolver import resolve_follow_up
from app.services.rag.factory import get_rag_graph, get_rag_pipeline
from app.services.rag.graph import RAGGraphStageError
from app.services.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _execute_query(
    pipeline: RAGPipeline, request: QueryRequest, *, retrieval_only: bool = False,
    resolved_question: str | None = None, resolution_status: str = "standalone",
) -> dict:
    document_ids = [str(item) for item in request.document_ids] if request.document_ids else None
    retrieval_question = resolved_question or request.question
    if pipeline.settings.rag_execution_mode == "langgraph":
        state = get_rag_graph(pipeline).invoke({
            "question": request.question,
            "resolved_question": retrieval_question,
            "resolution_status": resolution_status,
            "document_ids": document_ids,
            "top_k": request.top_k,
            "retrieval_only": retrieval_only,
        })
        return state["final_result"]
    if retrieval_only:
        return pipeline.query(retrieval_question, document_ids, request.top_k, retrieval_only=True)
    return pipeline.query(retrieval_question, document_ids, request.top_k)


def _ambiguous_reference_result(pipeline: RAGPipeline, top_k: int | None) -> dict:
    """Return a safe clarification without retrieval or model generation."""
    metadata = {
        "candidate_count": 0,
        "filtered_count": 0,
        "semantic_candidate_count": 0,
        "lexical_candidate_count": 0,
        "top_k": top_k or pipeline.settings.rag_top_k,
        "ranking_method": "reciprocal_rank_fusion",
    }
    return pipeline.build_result([], [], "", metadata, CLARIFICATION_RESPONSE, [], "no_evidence")


def _log_graph_failure(exc: RAGGraphStageError) -> None:
    logger.error("RAG graph execution failed stage=%s category=%s", exc.stage, exc.category)


def _validate_query_conversation(
    request: QueryRequest, db: Session = Depends(get_db),
) -> Conversation | None:
    if request.conversation_id is None:
        return None
    conversation = db.get(Conversation, request.conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def _persist(
    db: Session, result: dict, request: QueryRequest,
    existing_conversation: Conversation | None = None,
    clarification: bool = False,
) -> UUID:
    conversation = existing_conversation if request.conversation_id else None
    if request.conversation_id and conversation is None:
        conversation = db.get(Conversation, request.conversation_id)
    if request.conversation_id and conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = Conversation(title=request.question.strip()[:300])
        db.add(conversation)
        db.flush()
    else:
        conversation.updated_at = datetime.now(timezone.utc)
    citations = result["citations"]
    user_message_time = datetime.now(timezone.utc)
    db.add(ChatMessage(conversation_id=conversation.id, role="user", content=request.question,
                       citations=[], generation_status=None, created_at=user_message_time))
    assistant_content = result["answer"] or (
        "Answer generation is unavailable because no response could be generated. "
        "Retrieved evidence is included with this result." if result["generation_status"] == "provider_unavailable"
        else "No answer was generated. See generation_status and retrieved evidence."
    )
    db.add(ChatMessage(conversation_id=conversation.id, role="assistant", content=assistant_content,
                       citations=citations,
                       provider=None if clarification else result["provider"],
                       model=None if clarification else result["model"],
                       generation_status="clarification" if clarification else result["generation_status"],
                       created_at=user_message_time + timedelta(microseconds=1)))
    db.commit()
    return conversation.id


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db),
          existing_conversation: Conversation | None = Depends(_validate_query_conversation),
          pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> QueryResponse:
    try:
        history = (
            load_recent_conversation_history(db, request.conversation_id, pipeline.settings)
            if request.conversation_id is not None and existing_conversation is not None
            else []
        )
        resolution = resolve_follow_up(request.question, history)
        clarification = resolution.status in {"ambiguous_reference", "unresolved_reference", "no_history"}
        if clarification:
            result = _ambiguous_reference_result(pipeline, request.top_k)
        else:
            result = _execute_query(
                pipeline, request,
                resolved_question=resolution.retrieval_question,
                resolution_status=resolution.status,
            )
    except RAGGraphStageError as exc:
        _log_graph_failure(exc)
        raise HTTPException(status_code=503, detail="Query processing is temporarily unavailable") from None
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Query processing is temporarily unavailable") from exc
    conversation_id = _persist(db, result, request, existing_conversation, clarification)
    return QueryResponse(conversation_id=conversation_id, **result)


@router.post("/retrieve", response_model=QueryResponse)
def retrieve(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> QueryResponse:
    try:
        result = _execute_query(pipeline, request, retrieval_only=True)
    except RAGGraphStageError as exc:
        _log_graph_failure(exc)
        raise HTTPException(status_code=503, detail="Retrieval is temporarily unavailable") from None
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Retrieval is temporarily unavailable") from exc
    return QueryResponse(conversation_id=None, **result)


@router.get("/conversations", response_model=list[ConversationRead])
def list_conversations(offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100),
                       db: Session = Depends(get_db)) -> list[Conversation]:
    return list(db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()).offset(offset).limit(limit)).all())


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(conversation_id: UUID, db: Session = Depends(get_db)) -> Conversation:
    conversation = db.scalar(select(Conversation).options(selectinload(Conversation.messages)).where(Conversation.id == conversation_id))
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: UUID, db: Session = Depends(get_db)) -> None:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
