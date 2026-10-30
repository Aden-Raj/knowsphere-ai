"""Small natural-language retrieval and citation integrity evaluation in isolated Chroma."""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.services.embeddings import SentenceTransformerEmbedder


def main() -> None:
    dataset = json.loads((Path(__file__).with_name("dataset.json")).read_text(encoding="utf-8"))
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_device,
        cache_folder=settings.huggingface_cache_dir,
    )
    chunk_ids = [row["expected_chunk_id"] for row in dataset]
    documents = [row["document_text"] for row in dataset]
    vectors = embedder.embed(documents)

    client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
    collection = client.create_collection("evaluation")
    collection.add(
        ids=chunk_ids,
        embeddings=vectors,
        documents=documents,
        metadatas=[
            {
                "document_id": "eval_doc",
                "chunk_id": chunk_id,
                "chunk_index": index,
                "source_filename": "evaluation.txt",
            }
            for index, chunk_id in enumerate(chunk_ids)
        ],
    )

    hit = citation_ok = 0
    for row in dataset:
        query_embedding = embedder.embed([row["question"]])[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where={"document_id": "eval_doc"},
            include=["documents", "metadatas"],
        )
        found_id = result["ids"][0][0] if result["ids"][0] else None
        found_text = result["documents"][0][0] if found_id else None
        if found_id == row["expected_chunk_id"]:
            hit += 1
        citation_ok += int(
            found_id in chunk_ids and found_text == documents[chunk_ids.index(found_id)]
        )

    total = len(dataset)
    print(
        json.dumps(
            {"queries": total, "recall_at_1": hit / total, "citation_integrity": citation_ok / total},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
