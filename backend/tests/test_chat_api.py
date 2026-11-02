import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from app.core.config import Settings
from app.services.rag.graph import RAGGraphStageError
from app.services.rag.pipeline import ProviderError, RAGPipeline


class Embedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class TrackingEmbedder(Embedder):
    def __init__(self):
        self.queries = []

    def embed(self, texts):
        self.queries.extend(texts)
        return super().embed(texts)


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
            "question": "Revenue?",
            "resolved_question": "Revenue?", "resolution_status": "standalone",
            "document_ids": [store.doc_id], "top_k": 1,
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


def test_missing_conversation_on_followup_returns_404_before_rag(api_client, monkeypatch):
    client, _, _ = api_client
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pytest.fail("pipeline must not be initialized")

    from app.api.routes import chat as chat_route
    monkeypatch.setattr(chat_route, "_execute_query", lambda *args, **kwargs: pytest.fail("RAG must not run"))

    response = client.post("/api/v1/chat/query", json={"question": "x", "conversation_id": str(uuid4())})

    assert response.status_code == 404


def test_persistence_keeps_original_user_question(api_client):
    client, _, session_factory = api_client
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(
        Embedder(), type("Empty", (), {"query": lambda self, e, k, d=None: []})(), Settings(),
    )
    original_question = "What technologies does that project use?"
    response = client.post("/api/v1/chat/query", json={"question": original_question})

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    with session_factory() as db:
        from app.models.conversation import ChatMessage
        user_message = db.query(ChatMessage).filter_by(
            conversation_id=UUID(conversation_id), role="user",
        ).one()

    assert user_message.content == original_question


def test_followup_reuses_existing_conversation_and_persists_original_question(api_client):
    client, _, session_factory = api_client
    from app.models.conversation import ChatMessage, Conversation

    conversation_id = uuid4()
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="Project discussion"))
        db.add_all([
            ChatMessage(conversation_id=conversation_id, role="user", content="What projects are mentioned?"),
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content="The resume mentions two projects: AI-Powered Quiz Generator and KnowSphere AI.",
                citations=[],
                generation_status="generated",
            ),
        ])
        db.commit()

    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(
        Embedder(), type("Empty", (), {"query": lambda self, e, k, d=None: []})(), Settings(),
    )
    original_question = "What technologies does that project use?"
    response = client.post("/api/v1/chat/query", json={
        "question": original_question,
        "conversation_id": str(conversation_id),
    })

    assert response.status_code == 200, response.text
    assert response.json()["conversation_id"] == str(conversation_id)
    with session_factory() as db:
        persisted_conversation = db.get(Conversation, conversation_id)
        messages = db.query(ChatMessage).filter_by(conversation_id=conversation_id).all()
        conversation_count = db.query(Conversation).count()

    assert persisted_conversation is not None
    assert conversation_count == 1
    assert len(messages) == 4
    assert any(message.role == "user" and message.content == original_question for message in messages)


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_standalone_question_uses_original_query_in_both_modes(api_client, execution_mode, monkeypatch):
    client, _, _ = api_client
    embedder = TrackingEmbedder()
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(embedder, type("Empty", (), {"query": lambda self, e, k, d=None: []})(), settings)
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    from app.api.routes import chat as chat_route
    monkeypatch.setattr(
        chat_route, "load_recent_conversation_history",
        lambda *args, **kwargs: pytest.fail("new conversations must not load history"),
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    question = "Which technical skills appear in the document?"

    response = client.post("/api/v1/chat/query", json={"question": question})

    assert response.status_code == 200, response.text
    assert embedder.queries == [question]


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_resolved_followup_query_reaches_retrieval_in_both_modes(api_client, execution_mode):
    client, _, session_factory = api_client
    from app.models.conversation import ChatMessage, Conversation

    conversation_id = uuid4()
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="Project discussion"))
        db.add(ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content="The project is KnowSphere AI.",
            citations=[],
            generation_status="generated",
        ))
        db.commit()

    embedder = TrackingEmbedder()
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(embedder, type("Empty", (), {"query": lambda self, e, k, d=None: []})(), settings)
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    original_question = "What technologies does that project use?"

    response = client.post("/api/v1/chat/query", json={
        "question": original_question,
        "conversation_id": str(conversation_id),
    })

    assert response.status_code == 200, response.text
    assert embedder.queries == ["What technologies does KnowSphere AI use?"]
    detail = client.get(f"/api/v1/chat/conversations/{conversation_id}").json()
    assert any(message["role"] == "user" and message["content"] == original_question
               for message in detail["messages"])


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_ambiguous_followup_abstains_without_retrieval_or_generation(api_client, execution_mode):
    client, _, session_factory = api_client
    from app.models.conversation import ChatMessage, Conversation

    conversation_id = uuid4()
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="Skills discussion"))
        db.add(ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content="Skills: Angular Material and React Hooks.",
            citations=[],
            generation_status="generated",
        ))
        db.commit()

    embedder = TrackingEmbedder()
    settings = Settings()
    settings.rag_execution_mode = execution_mode

    class NoRetrievalStore:
        def query(self, embedding, top_k, document_ids=None):
            pytest.fail("ambiguous reference must not retrieve")

    class NoGenerationProvider:
        name, model = "test", "unused"

        def generate(self, system_prompt, user_prompt, max_tokens):
            pytest.fail("ambiguous reference must not generate")

    pipeline = RAGPipeline(embedder, NoRetrievalStore(), settings, NoGenerationProvider())
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    question = "What does the second one use?"

    response = client.post("/api/v1/chat/query", json={
        "question": question,
        "conversation_id": str(conversation_id),
    })

    assert response.status_code == 200, response.text
    assert response.json()["generation_status"] == "no_evidence"
    assert response.json()["citations"] == []
    assert response.json()["answer"] == "I couldn't determine what you're referring to. Could you clarify?"
    assert embedder.queries == []
    detail = client.get(f"/api/v1/chat/conversations/{conversation_id}").json()
    assert any(message["role"] == "user" and message["content"] == question for message in detail["messages"])


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_clarification_continuity_rewrites_reply_using_fresh_retrieval(api_client, execution_mode):
    client, _, session_factory = api_client
    from app.models.conversation import ChatMessage, Conversation

    conversation_id = uuid4()
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="Project discussion"))
        db.add(ChatMessage(
            conversation_id=conversation_id,
            role="assistant",
            content="The resume includes AI-Powered Quiz Generator and KnowSphere AI.",
            citations=[],
            generation_status="generated",
        ))
        db.commit()

    first_question = "Which of those projects uses RAG?"
    first = client.post("/api/v1/chat/query", json={
        "question": first_question,
        "conversation_id": str(conversation_id),
    })
    assert first.status_code == 200, first.text
    assert first.json()["generation_status"] == "no_evidence"
    assert first.json()["answer"] == "I couldn't determine what you're referring to. Could you clarify?"

    embedder = TrackingEmbedder()
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(embedder, type("Empty", (), {"query": lambda self, e, k, d=None: []})(), settings)
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    clarification_reply = "I mean KnowSphere AI."
    second = client.post("/api/v1/chat/query", json={
        "question": clarification_reply,
        "conversation_id": str(conversation_id),
    })

    assert second.status_code == 200, second.text
    assert embedder.queries == ["Does KnowSphere AI use RAG?"]
    detail = client.get(f"/api/v1/chat/conversations/{conversation_id}").json()
    assert len(detail["messages"]) == 5
    assert detail["messages"][1]["role"] == "user"
    assert detail["messages"][1]["content"] == first_question
    assert detail["messages"][2]["generation_status"] == "clarification"
    assert detail["messages"][2]["provider"] is None
    assert detail["messages"][3]["role"] == "user"
    assert detail["messages"][3]["content"] == clarification_reply
    assert len(client.get("/api/v1/chat/conversations").json()) == 1


@pytest.mark.parametrize("execution_mode", ["legacy", "langgraph"])
def test_followup_history_is_not_evidence_and_old_citations_are_not_reused(api_client, execution_mode):
    client, _, session_factory = api_client
    from app.models.conversation import ChatMessage, Conversation

    conversation_id = uuid4()
    old_chunk_id, current_chunk_id = str(uuid4()), str(uuid4())
    document_id = str(uuid4())
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="Project discussion"))
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        db.add_all([
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content="The project is KnowSphere AI. Private previous claim must not be evidence.",
                citations=[{"document_id": str(uuid4()), "chunk_id": old_chunk_id, "quoted_text": "old source"}],
                generation_status="generated",
                created_at=base_time,
            ),
            ChatMessage(
                conversation_id=conversation_id,
                role="user",
                content="Tell me more about that project.",
                created_at=base_time + timedelta(seconds=1),
            ),
            ChatMessage(
                conversation_id=conversation_id,
                role="assistant",
                content="I couldn't determine what you're referring to. Could you clarify?",
                citations=[],
                generation_status="clarification",
                created_at=base_time + timedelta(seconds=2),
            ),
        ])
        db.commit()

    row = {
        "id": current_chunk_id,
        "text": "KEY PROJECT EXPERIENCE\nProject: KnowSphere AI\nKnowSphere AI uses Python and FastAPI for its backend.",
        "distance": 0.1,
        "metadata": {
            "document_id": document_id,
            "chunk_id": current_chunk_id,
            "chunk_index": 0,
            "source_filename": "architecture.txt",
            "page_number": 1,
        },
    }

    class CurrentEvidenceStore:
        def query(self, embedding, top_k, document_ids=None):
            return [row]

        def lexical_query(self, query, top_k, document_ids=None):
            return [row]

        def get_document_chunks(self, requested_document_id):
            return [row] if requested_document_id == document_id else []

    class EvidenceProvider:
        name, model = "test", "deterministic"

        def __init__(self):
            self.prompt = None

        def generate(self, system_prompt, user_prompt, max_tokens):
            self.prompt = user_prompt
            return json.dumps({
                "answer": "KnowSphere AI uses Python and FastAPI for its backend.",
                "citation_chunk_ids": [current_chunk_id],
            })

    provider = EvidenceProvider()
    settings = Settings()
    settings.rag_execution_mode = execution_mode
    pipeline = RAGPipeline(Embedder(), CurrentEvidenceStore(), settings, provider)
    from app.api.routes.chat import get_rag_pipeline
    from app.main import app
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    response = client.post("/api/v1/chat/query", json={
        "question": "I mean KnowSphere AI.",
        "conversation_id": str(conversation_id),
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert [citation["chunk_id"] for citation in body["citations"]] == [current_chunk_id]
    assert old_chunk_id not in {citation["chunk_id"] for citation in body["citations"]}
    assert "Private previous claim must not be evidence" not in provider.prompt
    assert "old source" not in provider.prompt
    assert all(chunk["chunk_id"] == current_chunk_id for chunk in body["retrieved_chunks"])
