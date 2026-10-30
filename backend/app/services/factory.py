from functools import lru_cache

from app.core.config import get_settings
from app.services.document_extraction import DocumentExtractor
from app.services.document_processor import DocumentProcessor
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.file_storage import FileStorage
from app.services.vector_store import ChromaVectorStore


@lru_cache
def get_file_storage() -> FileStorage:
    settings = get_settings()
    return FileStorage(
        root=settings.upload_dir,
        max_upload_size_bytes=settings.max_upload_size_bytes,
        max_uncompressed_bytes=settings.max_uncompressed_bytes,
    )


@lru_cache
def get_document_processor() -> DocumentProcessor:
    settings = get_settings()
    storage = get_file_storage()
    return DocumentProcessor(
        storage=storage,
        extractor=DocumentExtractor(settings.max_extracted_chars, settings.max_document_pages),
        embedder=SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            cache_folder=settings.huggingface_cache_dir,
        ),
        vector_store=ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        ),
        settings=settings,
    )
