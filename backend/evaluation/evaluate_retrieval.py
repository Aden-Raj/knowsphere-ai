"""Evaluate hybrid retrieval against a temporary, isolated Chroma collection."""

import json
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import get_settings
from app.services.chunking import DocumentChunk
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.rag.pipeline import RAGPipeline
from app.services.vector_store import ChromaVectorStore


def main() -> None:
    dataset = json.loads((Path(__file__).with_name("dataset.json")).read_text(encoding="utf-8"))
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_device,
        cache_folder=settings.huggingface_cache_dir,
    )
    document_id = "hybrid-evaluation-document"
    chunks = [
        DocumentChunk(
            id=row["expected_chunk_id"] or f"eval_irrelevant_{index}",
            document_id=document_id,
            index=index,
            text=row["document_text"],
            source_filename="evaluation.txt",
            page_number=None,
            source_location=f"lines {index + 1}-{index + 1}",
        )
        for index, row in enumerate(dataset)
    ]
    vectors = embedder.embed([chunk.text for chunk in chunks])

    with TemporaryDirectory(prefix="knowsphere-rag-eval-") as temp_directory:
        store = ChromaVectorStore(Path(temp_directory) / "chroma", "hybrid_evaluation")
        store.replace_document_chunks(document_id, chunks, vectors)
        pipeline = RAGPipeline(embedder, store, settings)

        relevant = [row for row in dataset if row["expected_chunk_id"]]
        irrelevant = [row for row in dataset if not row["expected_chunk_id"]]
        hits = citation_integrity = 0
        for row in relevant:
            result = pipeline.query(row["question"], [document_id], retrieval_only=True)
            retrieved = {chunk["chunk_id"]: chunk for chunk in result["retrieved_chunks"]}
            expected_id = row["expected_chunk_id"]
            found = expected_id in retrieved
            hits += int(found)
            citation_integrity += int(
                found and any(
                    citation["chunk_id"] == expected_id
                    and citation["document_id"] == document_id
                    and citation["quoted_text"] == retrieved[expected_id]["text"]
                    for citation in result["citations"]
                )
            )

        rejected = 0
        for row in irrelevant:
            result = pipeline.query(row["question"], [document_id], retrieval_only=True)
            rejected += int(result["generation_status"] == "no_evidence")
        store.close()
        del pipeline, store
        gc.collect()

    print(json.dumps({
        "retrieval_method": "semantic_plus_bm25_rrf",
        "relevant_queries": len(relevant),
        "recall_at_k": hits / len(relevant) if relevant else 1.0,
        "citation_integrity": citation_integrity / len(relevant) if relevant else 1.0,
        "irrelevant_queries": len(irrelevant),
        "irrelevant_rejection_rate": rejected / len(irrelevant) if irrelevant else 1.0,
    }, indent=2))


if __name__ == "__main__":
    main()
