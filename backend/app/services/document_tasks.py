import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.knowledge_document import DocumentStatus, KnowledgeDocument
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
_active_tasks: set[UUID] = set()
_active_tasks_lock = threading.Lock()


def reserve_document_task(document_id: UUID) -> bool:
    with _active_tasks_lock:
        if document_id in _active_tasks:
            return False
        _active_tasks.add(document_id)
        return True


def release_document_task(document_id: UUID) -> None:
    with _active_tasks_lock:
        _active_tasks.discard(document_id)


def is_document_task_active(document_id: UUID) -> bool:
    with _active_tasks_lock:
        return document_id in _active_tasks


def process_document_task(
    document_id: UUID,
    session_factory: Callable[[], Session],
    processor: DocumentProcessor,
) -> None:
    try:
        with session_factory() as db:
            document = db.get(KnowledgeDocument, document_id)
            if document is None:
                return
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            document.processing_completed_at = None
            db.commit()
            try:
                result = processor.process(document)
                document = db.get(KnowledgeDocument, document_id)
                if document is None:
                    processor.delete_vectors(str(document_id))
                    return
                document.status = DocumentStatus.COMPLETED
                document.chunk_count = result.chunk_count
                document.embedding_model = result.embedding_model
                document.error_message = None
                document.processing_completed_at = datetime.now(UTC)
                db.commit()
            except Exception:
                logger.exception("Document processing failed for document_id=%s", document_id)
                try:
                    processor.delete_vectors(str(document_id))
                except Exception:
                    logger.exception("Vector cleanup failed for document_id=%s", document_id)
                db.rollback()
                document = db.get(KnowledgeDocument, document_id)
                if document is not None:
                    document.status = DocumentStatus.FAILED
                    document.error_message = (
                        "Document processing failed. Correct the source or service issue and reindex."
                    )
                    document.chunk_count = 0
                    document.embedding_model = None
                    document.processing_completed_at = None
                    db.commit()
    finally:
        release_document_task(document_id)
