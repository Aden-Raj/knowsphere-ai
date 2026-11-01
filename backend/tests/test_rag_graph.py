import json

import pytest

from app.core.config import Settings
from app.services.rag.factory import get_rag_graph
from app.services.rag.graph import RAGGraphStageError, build_rag_graph
from app.services.rag.pipeline import ProviderError, RAGPipeline
from app.services.rag.state import RAGGraphState


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self, records):
        self.records = records

    def query(self, embedding, top_k, document_ids=None):
        records = self.records
        if document_ids:
            records = [row for row in records if row["metadata"]["document_id"] in document_ids]
        return records[:top_k]

    def lexical_query(self, query, top_k, document_ids=None):
        return self.query([], top_k, document_ids)


class FakeProvider:
    name = "fake"
    model = "test-model"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, system_prompt, user_prompt, max_tokens):
        return self.answer


def make_pipeline(provider=None, records=None):
    records = records or []
    settings = Settings(_env_file=None, llm_provider="none")
    return RAGPipeline(FakeEmbedder(), FakeStore(records), settings, provider)


def record():
    return {
        "id": "doc_0",
        "text": "Revenue grew by 15%.",
        "metadata": {
            "document_id": "doc",
            "source_filename": "report.txt",
            "chunk_id": "doc_0",
            "chunk_index": 0,
            "page_number": 1,
        },
        "distance": 0.1,
    }


def test_graph_state_has_only_per_request_rag_fields():
    expected_fields = {
        "question", "document_ids", "top_k", "retrieval_only", "query_analysis", "retrieved_chunks",
        "selected_chunks", "assembled_context", "retrieval_metadata", "provider", "model",
        "generated_answer", "citation_chunk_ids", "validated_citations", "generation_status",
        "error_category", "final_result",
    }

    assert set(RAGGraphState.__annotations__) == expected_fields
    assert not expected_fields.intersection({"db", "session", "provider_instance", "api_key", "credentials"})


def test_graph_factory_reuses_compiled_graph_for_pipeline_instance():
    pipeline = make_pipeline()
    get_rag_graph.cache_clear()
    try:
        first = get_rag_graph(pipeline)
        second = get_rag_graph(pipeline)
    finally:
        get_rag_graph.cache_clear()

    assert first is second


def test_graph_compiles_expected_stage_nodes_and_skips_generation_when_provider_missing(monkeypatch):
    p = make_pipeline(records=[record()])
    monkeypatch.setattr(
        p, "generate_response",
        lambda *args, **kwargs: pytest.fail("generation must not run when provider is missing"),
    )
    graph = build_rag_graph(p)

    assert {"retrieve", "analyze", "generate", "finalize"}.issubset(graph.get_graph().nodes)

    graph_state = graph.invoke({"question": "What happened to revenue?"})

    assert graph_state["generation_status"] == "provider_unavailable"
    assert graph_state["error_category"] is None
    assert graph_state["final_result"]["generation_status"] == "provider_unavailable"
    assert graph_state["final_result"]["citations"][0]["chunk_id"] == "doc_0"
    assert graph_state["citation_chunk_ids"] == ["doc_0"]


def test_graph_generation_branch_matches_query_facade_result():
    row = record()
    output = json.dumps({"answer": "Revenue increased by 15 percent.", "citation_chunk_ids": ["doc_0"]})
    p = make_pipeline(provider=FakeProvider(output), records=[row])
    question = "What happened to revenue?"

    expected = p.query(question)
    graph_state = build_rag_graph(p).invoke({"question": question})

    assert graph_state["final_result"] == expected
    assert graph_state["generated_answer"] == expected["answer"]
    assert graph_state["citation_chunk_ids"] == ["doc_0"]
    assert graph_state["validated_citations"] == expected["citations"]
    assert graph_state["provider"] == "fake"
    assert graph_state["model"] == "test-model"


def test_graph_provider_failure_keeps_only_safe_status_metadata():
    sensitive_detail = "credential=private-token prompt=private-question document=private-text"

    class BrokenProvider:
        name = "fake"
        model = "test-model"

        def generate(self, system_prompt, user_prompt, max_tokens):
            raise ProviderError(sensitive_detail, category="http_error")

    graph_state = build_rag_graph(make_pipeline(BrokenProvider(), [record()])).invoke(
        {"question": "What happened to revenue?"},
    )

    assert graph_state["generation_status"] == "provider_error"
    assert graph_state["error_category"] == "provider_error"
    assert sensitive_detail not in repr(graph_state["final_result"])


def test_graph_executes_successful_stages_in_order(monkeypatch):
    row = record()
    output = json.dumps({"answer": "Revenue increased by 15 percent.", "citation_chunk_ids": ["doc_0"]})
    p = make_pipeline(provider=FakeProvider(output), records=[row])
    events = []

    for method_name, label in (
        ("retrieve_context", "retrieve"),
        ("analyze_query", "analyze"),
        ("generate_response", "generate"),
        ("parse_provider_response", "parse"),
        ("validate_response", "validate"),
        ("build_result", "finalize"),
    ):
        original = getattr(p, method_name)

        def tracked(*args, _original=original, _label=label, **kwargs):
            events.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(p, method_name, tracked)

    original_provider_generate = p.provider.generate

    def tracked_provider_generate(*args, **kwargs):
        events.append("provider")
        return original_provider_generate(*args, **kwargs)

    monkeypatch.setattr(p.provider, "generate", tracked_provider_generate)

    graph_state = build_rag_graph(p).invoke({"question": "What happened to revenue?"})

    assert events == ["retrieve", "analyze", "generate", "provider", "parse", "validate", "finalize"]
    assert graph_state["final_result"]["answer"] == "Revenue increased by 15 percent."
    assert graph_state["final_result"]["citations"][0]["chunk_id"] == "doc_0"


@pytest.mark.parametrize(
    ("question", "records", "graph_input", "expected_status"),
    [
        ("What does the document say?", [], {}, "no_evidence"),
        ("What happened to revenue?", [record()], {"retrieval_only": True}, "retrieval_only"),
        ("What happened to revenue?", [record()], {}, "provider_unavailable"),
        ("Have I worked with Kubernetes?", [record()], {}, "generated"),
    ],
)
def test_graph_early_routes_skip_generation(question, records, graph_input, expected_status, monkeypatch):
    p = make_pipeline(records=records)
    monkeypatch.setattr(
        p, "generate_response",
        lambda *args, **kwargs: pytest.fail("early route must not invoke generation"),
    )

    graph_state = build_rag_graph(p).invoke({"question": question, **graph_input})
    legacy_result = p.query(question, retrieval_only=graph_input.get("retrieval_only", False))

    assert graph_state["final_result"] == legacy_result
    assert graph_state["final_result"]["generation_status"] == expected_status
    if expected_status == "no_evidence":
        assert graph_state["final_result"]["answer"] is None
        assert graph_state["final_result"]["citations"] == []
    elif expected_status == "retrieval_only":
        assert graph_state["final_result"]["answer"] is None
        assert graph_state["final_result"]["citations"][0]["chunk_id"] == "doc_0"
    elif question.startswith("Have I"):
        assert graph_state["final_result"]["answer"]
        assert graph_state["final_result"]["citations"] == []


@pytest.mark.parametrize(
    ("stage_method", "route_with_provider"),
    [
        ("retrieve_context", False),
        ("analyze_query", False),
        ("generate_response", True),
        ("build_result", False),
    ],
)
def test_graph_stage_errors_are_sanitized(stage_method, route_with_provider, monkeypatch):
    secret_markers = "credential=private-token prompt=private-question document=private-text"
    provider = FakeProvider(json.dumps({"answer": "Answer.", "citation_chunk_ids": ["doc_0"]})) if route_with_provider else None
    p = make_pipeline(provider=provider, records=[record()])

    def fail_with_sensitive_detail(*args, **kwargs):
        raise RuntimeError(secret_markers)

    monkeypatch.setattr(p, stage_method, fail_with_sensitive_detail)
    graph = build_rag_graph(p)

    with pytest.raises(RAGGraphStageError) as error:
        graph.invoke({"question": "What happened to revenue?"})

    assert error.value.category == "stage_failure"
    assert error.value.stage in {"retrieve", "analyze", "generate", "finalize"}
    assert "private-token" not in str(error.value)
    assert "private-question" not in str(error.value)
    assert "private-text" not in str(error.value)
    assert error.value.__cause__ is None
