import json
from uuid import uuid4

from app.core.config import Settings
from app.services.rag.pipeline import RAGPipeline


class Embedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class Store:
    def query(self, embedding, top_k, document_ids=None):
        if document_ids and self.doc_id not in document_ids:
            return []
        return [{"id": self.chunk, "text": "The report says revenue grew 15%.", "distance": 0.1,
                 "metadata": {"document_id": self.doc_id, "chunk_id": self.chunk,
                              "chunk_index": 0, "source_filename": "report.txt", "page_number": None}}]


class Provider:
    name, model = "test", "deterministic"

    def __init__(self, chunk):
        self.chunk = chunk

    def generate(self, system_prompt, user_prompt, max_tokens):
        return json.dumps({"answer": "Revenue grew 15%.", "citation_chunk_ids": [self.chunk]})


def test_query_conversation_round_trip_and_delete(api_client):
    client, _, session_factory = api_client
    store = Store()
    store.doc_id, store.chunk = str(uuid4()), str(uuid4())
    pipeline = RAGPipeline(Embedder(), store, Settings(), Provider(store.chunk))
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/api/v1/chat/query", json={"question": "Revenue?", "document_ids": [store.doc_id]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generation_status"] == "generated"
    assert body["citations"][0]["chunk_id"] == store.chunk
    conversation_id = body["conversation_id"]
    listed = client.get("/api/v1/chat/conversations")
    assert listed.status_code == 200 and listed.json()[0]["id"] == conversation_id
    detail = client.get(f"/api/v1/chat/conversations/{conversation_id}")
    assert len(detail.json()["messages"]) == 2
    assert detail.json()["messages"][1]["citations"][0]["quoted_text"] == "The report says revenue grew 15%."
    assert client.delete(f"/api/v1/chat/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/v1/chat/conversations/{conversation_id}").status_code == 404


def test_validation_and_retrieve_without_persistence(api_client):
    client, _, _ = api_client
    class EmptyStore:
        def query(self, embedding, top_k, document_ids=None):
            return []

    pipeline = RAGPipeline(Embedder(), EmptyStore(), Settings())
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    response = client.post("/api/v1/chat/query", json={"question": "   "})
    assert response.status_code == 422
    response = client.post("/api/v1/chat/query", json={"question": "x", "top_k": 0})
    assert response.status_code == 422
    response = client.post("/api/v1/chat/retrieve", json={"question": "search only"})
    assert response.status_code == 200
    assert response.json()["conversation_id"] is None
    assert response.json()["generation_status"] == "no_evidence"
    assert client.get("/api/v1/chat/conversations").json() == []


def test_retrieve_endpoint_returns_hybrid_evidence_with_verifiable_citation(api_client):
    client, _, _ = api_client
    store = Store()
    store.doc_id, store.chunk = str(uuid4()), str(uuid4())

    class LexicalOnlyStore(Store):
        def query(self, embedding, top_k, document_ids=None):
            row = super().query(embedding, top_k, document_ids)[0]
            row["distance"] = 0.95
            return [row]

        def lexical_query(self, query, top_k, document_ids=None):
            return super().query([0.0], top_k, document_ids)

    hybrid_store = LexicalOnlyStore()
    hybrid_store.doc_id, hybrid_store.chunk = store.doc_id, store.chunk
    pipeline = RAGPipeline(Embedder(), hybrid_store, Settings())
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/api/v1/chat/retrieve", json={
        "question": "What is the person's name?", "document_ids": [store.doc_id]
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["generation_status"] == "retrieval_only"
    assert body["retrieved_chunks"][0]["document_id"] == store.doc_id
    assert body["citations"][0]["document_id"] == store.doc_id
    assert body["citations"][0]["chunk_id"] == body["retrieved_chunks"][0]["chunk_id"]


def test_missing_conversation_on_followup_returns_404(api_client):
    client, _, _ = api_client
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(Embedder(), type("Empty", (), {"query": lambda self, e, k, d=None: []})(), Settings())
    response = client.post("/api/v1/chat/query", json={"question": "x", "conversation_id": str(uuid4())})
    assert response.status_code == 404
