import json
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.services.rag.graph import RAGGraphStageError
from app.services.rag.pipeline import ProviderError, RAGPipeline


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


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_query_conversation_round_trip_and_delete(api_client, execution_mode, monkeypatch):
    client, _, session_factory = api_client
    store = Store()
    store.doc_id, store.chunk = str(uuid4()), str(uuid4())
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(Embedder(), store, settings, Provider(store.chunk))
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    if execution_mode == "legacy":
        from app.api.routes import chat as chat_route
        monkeypatch.setattr(chat_route, "get_rag_graph", lambda _: pytest.fail("legacy mode built a graph"))
    else:
        from app.api.routes import chat as chat_route
        original_get_graph = chat_route.get_rag_graph
        graph_inputs = {}

        class CapturingGraph:
            def __init__(self, graph):
                self.graph = graph

            def invoke(self, state):
                graph_inputs.update(state)
                return self.graph.invoke(state)

        monkeypatch.setattr(
            chat_route, "get_rag_graph",
            lambda graph_pipeline: CapturingGraph(original_get_graph(graph_pipeline)),
        )
        monkeypatch.setattr(pipeline, "query", lambda *args, **kwargs: pytest.fail("graph mode used legacy query"))

    response = client.post("/api/v1/chat/query", json={
        "question": "Revenue?", "document_ids": [store.doc_id], "top_k": 1,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "conversation_id", "answer", "citations", "retrieved_chunks", "provider", "model",
        "generation_status", "retrieval_metadata",
    }
    assert body["generation_status"] == "generated"
    assert body["citations"][0]["chunk_id"] == store.chunk
    if execution_mode == "langgraph":
        assert graph_inputs == {
            "question": "Revenue?", "document_ids": [store.doc_id], "top_k": 1,
            "retrieval_only": False,
        }
    conversation_id = body["conversation_id"]
    listed = client.get("/api/v1/chat/conversations")
    assert listed.status_code == 200 and len(listed.json()) == 1
    assert listed.json()[0]["id"] == conversation_id
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


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_retrieve_endpoint_returns_hybrid_evidence_with_verifiable_citation(api_client, execution_mode):
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
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(Embedder(), hybrid_store, settings)
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
    assert client.get("/api/v1/chat/conversations").json() == []


def test_langgraph_stage_failure_returns_safe_503_and_logs_only_stage_metadata(api_client, caplog, monkeypatch):
    client, _, _ = api_client
    settings = Settings()
    settings.rag_execution_mode = "langgraph"
    pipeline = RAGPipeline(Embedder(), Store(), settings)
    from app.api.routes import chat as chat_route
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app

    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    class BrokenGraph:
        def invoke(self, state):
            raise RAGGraphStageError("retrieve")

    monkeypatch.setattr(chat_route, "get_rag_graph", lambda _: BrokenGraph())

    response = client.post("/api/v1/chat/query", json={"question": "private prompt text"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Query processing is temporarily unavailable"}
    assert "RAG graph execution failed stage=retrieve category=stage_failure" in caplog.text
    assert "private prompt text" not in caplog.text
    assert client.get("/api/v1/chat/conversations").json() == []


def test_legacy_provider_failure_remains_a_persisted_provider_error_response(api_client):
    client, _, _ = api_client
    store = Store()
    store.doc_id, store.chunk = str(uuid4()), str(uuid4())

    class FailedProvider:
        name, model = "test", "deterministic"

        def generate(self, system_prompt, user_prompt, max_tokens):
            raise ProviderError("private provider detail", category="timeout")

    pipeline = RAGPipeline(Embedder(), store, Settings(), FailedProvider())
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/api/v1/chat/query", json={"question": "Revenue?"})

    assert response.status_code == 200
    assert response.json()["generation_status"] == "provider_error"
    assert response.json()["answer"] is None
    assert "private provider detail" not in response.text
    conversations = client.get("/api/v1/chat/conversations").json()
    assert len(conversations) == 1
    detail = client.get(f"/api/v1/chat/conversations/{conversations[0]['id']}").json()
    assert len(detail["messages"]) == 2


def test_missing_conversation_on_followup_returns_404(api_client):
    client, _, _ = api_client
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(Embedder(), type("Empty", (), {"query": lambda self, e, k, d=None: []})(), Settings())
    response = client.post("/api/v1/chat/query", json={"question": "x", "conversation_id": str(uuid4())})
    assert response.status_code == 404
