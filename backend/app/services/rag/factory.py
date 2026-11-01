from functools import lru_cache

from app.core.config import get_settings
from app.services.factory import get_document_processor
from app.services.rag.graph import build_rag_graph
from app.services.rag.pipeline import RAGPipeline


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    processor = get_document_processor()
    return RAGPipeline(processor.embedder, processor.vector_store, get_settings())


@lru_cache(maxsize=8)
def get_rag_graph(pipeline: RAGPipeline):
    """Reuse one compiled graph for each cached pipeline instance."""
    return build_rag_graph(pipeline)
