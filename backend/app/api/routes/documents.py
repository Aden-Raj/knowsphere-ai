import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import get_db
from app.models.knowledge_document import DocumentStatus, KnowledgeDocument
from app.schemas.documents import DocumentRead, MultipleUploadResponse, UploadFailure
from app.services.document_processor import DocumentProcessor
from app.services.document_tasks import (
    is_document_task_active,
    process_document_task,
    release_document_task,
    reserve_document_task,
)
from app.services.factory import get_document_processor, get_file_storage
from app.services.file_storage import FileStorage, FileTooLargeError, UploadValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _http_upload_error(exc: UploadValidationError) -> HTTPException:
    if isinstance(exc, FileTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


async def _create_document(
    upload: UploadFile,
    db: Session,
    storage: FileStorage,
) -> KnowledgeDocument:
    stored = await storage.save_upload(upload)
    document = KnowledgeDocument(
        id=uuid4(),
        title=stored.original_filename[:300],
        original_filename=stored.original_filename,
        storage_path=str(stored.path),
        file_extension=stored.extension,
        content_type=stored.mime_type,
        file_size_bytes=stored.size_bytes,
        status=DocumentStatus.UPLOADED,
        chunk_count=0,
    )
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception:
        db.rollback()
        storage.delete_file(str(stored.path))
        raise
    return document


def _schedule_processing(
    background_tasks: BackgroundTasks,
    db: Session,
    document: KnowledgeDocument,
    processor: DocumentProcessor,
) -> None:
    if not reserve_document_task(document.id):
        raise HTTPException(status_code=409, detail="Document processing is already scheduled")
    try:
        task_session_factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)
        background_tasks.add_task(process_document_task, document.id, task_session_factory, processor)
    except Exception:
        release_document_task(document.id)
        raise


def _get_document(db: Session, document_id: UUID) -> KnowledgeDocument:
    document = db.get(KnowledgeDocument, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: FileStorage = Depends(get_file_storage),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> DocumentRead:
    try:
        document = await _create_document(file, db, storage)
    except UploadValidationError as exc:
        raise _http_upload_error(exc) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Unable to store uploaded document") from exc
    _schedule_processing(background_tasks, db, document, processor)
    return DocumentRead.from_document(document)


@router.post("/upload/multiple", response_model=MultipleUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_multiple_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    storage: FileStorage = Depends(get_file_storage),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> MultipleUploadResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if len(files) > settings.max_upload_files:
        raise HTTPException(status_code=413, detail=f"At most {settings.max_upload_files} files may be uploaded at once")

    accepted: list[DocumentRead] = []
    rejected: list[UploadFailure] = []
    for upload in files:
        try:
            document = await _create_document(upload, db, storage)
        except UploadValidationError as exc:
            rejected.append(UploadFailure(filename=upload.filename, detail=str(exc)))
            continue
        except OSError:
            logger.exception("Unable to store uploaded document")
            rejected.append(UploadFailure(filename=upload.filename, detail="Unable to store uploaded document"))
            continue
        _schedule_processing(background_tasks, db, document, processor)
        accepted.append(DocumentRead.from_document(document))
    return MultipleUploadResponse(accepted=accepted, rejected=rejected)


@router.get("", response_model=list[DocumentRead])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[DocumentRead]:
    documents = db.scalars(
        select(KnowledgeDocument)
        .order_by(KnowledgeDocument.created_at.desc(), KnowledgeDocument.id)
        .offset(offset)
        .limit(limit)
    ).all()
    return [DocumentRead.from_document(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: UUID, db: Session = Depends(get_db)) -> DocumentRead:
    return DocumentRead.from_document(_get_document(db, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    storage: FileStorage = Depends(get_file_storage),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> None:
    document = _get_document(db, document_id)
    if is_document_task_active(document.id):
        raise HTTPException(status_code=409, detail="A processing document cannot be deleted")
    try:
        processor.delete_vectors(str(document.id))
    except Exception as exc:
        logger.exception("Unable to remove vectors for document_id=%s", document_id)
        raise HTTPException(status_code=503, detail="Document vectors are temporarily unavailable") from exc
    storage_path = document.storage_path
    db.delete(document)
    db.commit()
    try:
        storage.delete_file(storage_path)
    except (OSError, ValueError):
        logger.exception("Unable to remove stored source for document_id=%s", document_id)


@router.post("/{document_id}/reindex", response_model=DocumentRead, status_code=status.HTTP_202_ACCEPTED)
def reindex_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> DocumentRead:
    document = _get_document(db, document_id)
    if is_document_task_active(document.id):
        raise HTTPException(status_code=409, detail="Document processing is already in progress")
    if not document.storage_path or not Path(document.storage_path).is_file():
        raise HTTPException(status_code=409, detail="The stored source file is unavailable")
    document.status = DocumentStatus.UPLOADED
    document.error_message = None
    document.chunk_count = 0
    document.embedding_model = None
    document.processing_completed_at = None
    db.commit()
    db.refresh(document)
    _schedule_processing(background_tasks, db, document, processor)
    return DocumentRead.from_document(document)
