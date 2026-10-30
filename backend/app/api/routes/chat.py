from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.session import get_db
from app.models.conversation import ChatMessage, Conversation
from app.schemas.chat import ConversationRead, QueryRequest, QueryResponse
from app.services.rag.factory import get_rag_pipeline
from app.services.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


def _persist(db: Session, result: dict, request: QueryRequest) -> UUID:
    conversation = db.get(Conversation, request.conversation_id) if request.conversation_id else None
    if request.conversation_id and conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation is None:
        conversation = Conversation(title=request.question.strip()[:300])
        db.add(conversation)
        db.flush()
    else:
        conversation.updated_at = datetime.now(timezone.utc)
    citations = result["citations"]
    db.add(ChatMessage(conversation_id=conversation.id, role="user", content=request.question,
                       citations=[], generation_status=None))
    assistant_content = result["answer"] or (
        "Answer generation is unavailable because no response could be generated. "
        "Retrieved evidence is included with this result." if result["generation_status"] == "provider_unavailable"
        else "No answer was generated. See generation_status and retrieved evidence."
    )
    db.add(ChatMessage(conversation_id=conversation.id, role="assistant", content=assistant_content,
                       citations=citations, provider=result["provider"], model=result["model"],
                       generation_status=result["generation_status"]))
    db.commit()
    return conversation.id


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db),
          pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> QueryResponse:
    try:
        result = pipeline.query(request.question, [str(i) for i in request.document_ids] if request.document_ids else None,
                                request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Query processing is temporarily unavailable") from exc
    conversation_id = _persist(db, result, request)
    return QueryResponse(conversation_id=conversation_id, **result)


@router.post("/retrieve", response_model=QueryResponse)
def retrieve(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> QueryResponse:
    try:
        result = pipeline.query(request.question, [str(i) for i in request.document_ids] if request.document_ids else None,
                                request.top_k, retrieval_only=True)
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
