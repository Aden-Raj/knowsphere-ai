import math
from pathlib import Path
import re

from app.services.chunking import DocumentChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.IGNORECASE)
_PERSON_NAME_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z'’-]{1,}|[A-Z][a-z'’-]{1,})"
    r"(?:\s+(?:[A-Z][A-Z'’-]{1,}|[A-Z][a-z'’-]{1,})){1,3}\b"
)
_NAME_ROLE_PATTERN = re.compile(r"\b(person|candidate|applicant|author|individual)\b", re.IGNORECASE)
_NAME_ACTION_PATTERN = re.compile(r"\b(name|named|mentioned|listed|identified)\b", re.IGNORECASE)
_WHO_DOCUMENT_PATTERN = re.compile(
    r"\bwho\b.*\b(mentioned|named|listed|identified)\b.*\b(document|file|resume|cv|uploaded)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "their", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with", "you", "your", "mentioned",
    "uploaded", "document",
}


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
        collection_count = self._collection.count()
        if collection_count == 0 or top_k < 1:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self._collection.query(
            query_embeddings=[embedding], n_results=min(top_k, collection_count),
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

    def lexical_query(self, query: str, top_k: int, document_ids: list[str] | None = None) -> list[dict]:
        """Rank persisted chunk text with BM25, with a name-entity fallback for identity questions."""
        if top_k < 1:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        documents = result.get("documents") or []
        if not documents:
            return []

        terms = [token.casefold() for token in _TOKEN_PATTERN.findall(query)
                 if token.casefold() not in _STOPWORDS]
        name_intent = bool(
            (_NAME_ROLE_PATTERN.search(query) and _NAME_ACTION_PATTERN.search(query))
            or _WHO_DOCUMENT_PATTERN.search(query)
        )
        if not terms and not name_intent:
            return []

        tokenized = [[token.casefold() for token in _TOKEN_PATTERN.findall(text or "")]
                     for text in documents]
        lengths = [len(tokens) for tokens in tokenized]
        average_length = sum(lengths) / len(lengths) or 1.0
        document_frequency: dict[str, int] = {}
        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total_documents = len(tokenized)
        ranked: list[dict] = []
        for index, (vector_id, text, metadata) in enumerate(
            zip(result["ids"], documents, result.get("metadatas") or [], strict=True)
        ):
            tokens = tokenized[index]
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            score = 0.0
            for term in set(terms):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequency[term]
                inverse_document_frequency = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * lengths[index] / average_length)
                score += inverse_document_frequency * frequency * 2.5 / denominator

            if name_intent and _PERSON_NAME_PATTERN.search(text or ""):
                # Identity questions commonly use no words that appear beside a name in a resume.
                # Prefer name-like text near the start, where resumes usually place the person's name.
                match = _PERSON_NAME_PATTERN.search(text or "")
                assert match is not None
                score += 1.0 / (1.0 + match.start() / 200)

            if score > 0:
                ranked.append({"id": vector_id, "text": text, "metadata": metadata or {},
                               "lexical_score": score})

        ranked.sort(key=lambda row: (-row["lexical_score"],
                                     int(row["metadata"].get("chunk_index", 0)), row["id"]))
        return ranked[:top_k]

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def close(self) -> None:
        self._client.close()

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
