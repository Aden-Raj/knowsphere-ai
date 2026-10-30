from pathlib import Path

from app.services.chunking import DocumentChunk


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_document_chunks(self, document_id: str, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        self.delete_document(document_id)
        batch_size = 128
        for offset in range(0, len(chunks), batch_size):
            chunk_batch = chunks[offset : offset + batch_size]
            embedding_batch = embeddings[offset : offset + batch_size]
            self._collection.upsert(
                ids=[chunk.id for chunk in chunk_batch],
                embeddings=embedding_batch,
                documents=[chunk.text for chunk in chunk_batch],
                metadatas=[self._metadata(chunk) for chunk in chunk_batch],
            )

    def get_document_chunks(self, document_id: str) -> list[dict]:
        result = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        records = []
        for index, vector_id in enumerate(result["ids"]):
            records.append(
                {
                    "id": vector_id,
                    "text": result["documents"][index],
                    "metadata": result["metadatas"][index],
                }
            )
        return records

    def query(self, embedding: list[float], top_k: int, document_ids: list[str] | None = None) -> list[dict]:
        if self._collection.count() == 0:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[embedding], n_results=top_k,
            where=where, include=["documents", "metadatas", "distances"],
        )
        records = []
        for index, vector_id in enumerate(result["ids"][0]):
            records.append({
                "id": vector_id,
                "text": result["documents"][0][index],
                "metadata": result["metadatas"][0][index],
                "distance": result["distances"][0][index],
            })
        return records

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "chunk_index": chunk.index,
            "source_filename": chunk.source_filename,
            "source_location": chunk.source_location,
        }
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        return metadata
