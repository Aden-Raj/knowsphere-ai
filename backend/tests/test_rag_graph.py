import json

import pytest

from app.core.config import Settings
import app.services.rag.graph as rag_graph
from app.services.rag.factory import get_rag_graph
from app.services.rag.graph import (
    RAGGraphStageError,
    build_adjustment_queries,
    build_rag_graph,
    classify_task,
    evaluate_evidence_coverage,
    plan_queries,
)
from app.services.rag.pipeline import ProviderError, RAGPipeline, RetrievedChunk
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
        "question", "resolved_question", "resolution_status", "task_type",
        "subqueries", "adjustment_queries", "task_evidence", "retrieval_round", "evidence_coverage",
        "document_ids", "top_k", "retrieval_only", "query_analysis", "retrieved_chunks",
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

    assert {
        "classify", "plan", "retrieve", "retrieve_tasks", "aggregate", "evaluate",
        "retrieve_adjustments", "evaluate_after_adjustment", "analyze", "generate", "finalize",
    }.issubset(graph.get_graph().nodes)

    graph_state = graph.invoke({"question": "What happened to revenue?"})

    assert graph_state["generation_status"] == "provider_unavailable"
    assert graph_state["task_type"] == "direct"
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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the reported revenue?", "direct"),
        ("What does semantic search mean?", "direct"),
        ("Explain the API response format.", "direct"),
        ("Investigate the causes and findings in the report.", "research"),
        ("Compare Angular and React.", "comparison"),
        ("What are the differences between these two projects?", "comparison"),
        ("Summarize my experience, projects, and technical strengths.", "synthesis"),
        ("Combine findings across multiple documents into an overview.", "synthesis"),
        ("Compare the research findings for the two projects.", "comparison"),
        ("Research and summarize my experience, projects, and skills.", "synthesis"),
        ("Can KnowSphere AI use Angular and React?", "direct"),
        ("Tell me more about the project.", "direct"),
    ],
)
def test_deterministic_task_classifier(question, expected):
    assert classify_task(question) == expected


def test_research_planner_splits_aspects_and_caps_queries_at_three():
    plan = plan_queries(
        "Research the architecture, retrieval methods, deployment technologies, and security controls used in KnowSphere AI.",
        "research",
    )

    assert len(plan) == 3
    assert plan[0]["query"] == "architecture used in KnowSphere AI"
    assert plan[1]["query"] == "retrieval methods used in KnowSphere AI"
    assert "deployment technologies and security controls" in plan[2]["query"]


def test_synthesis_planner_splits_requested_topics():
    plan = plan_queries(
        "Summarize my professional experience, projects, and technical strengths.", "synthesis",
    )

    assert [item["query"] for item in plan] == [
        "my professional experience", "projects", "technical strengths",
    ]


def test_comparison_planner_preserves_subjects_and_adds_difference_query():
    plan = plan_queries("Compare Angular and React.", "comparison")

    assert [item["subject"] for item in plan] == ["Angular", "React", None]
    assert [item["aspect"] for item in plan] == ["Angular", "React", "comparison"]
    assert len(plan) == 3
    assert "Angular" in plan[2]["query"] and "React" in plan[2]["query"]


@pytest.mark.parametrize("question", [
    "What are the differences between these two projects?",
    "Compare them.",
    "Research the topic.",
])
def test_uncertain_or_unhelpful_plan_falls_back_to_original_question(question):
    task_type = classify_task(question)
    expected_aspect = "comparison subjects" if task_type == "comparison" else question
    assert plan_queries(question, task_type) == [
        {"query": question, "subject": None, "aspect": expected_aspect},
    ]


def test_evidence_coverage_marks_all_research_aspects_and_deduplicates_chunks():
    plan = [
        {"query": "architecture", "subject": None, "aspect": "architecture"},
        {"query": "deployment", "subject": None, "aspect": "deployment"},
    ]
    evidence = [
        {"query": "architecture", "subject": None, "aspect": "architecture", "chunk_ids": ["a", "a"]},
        {"query": "deployment", "subject": None, "aspect": "deployment", "chunk_ids": ["b"]},
    ]

    coverage = evaluate_evidence_coverage("research", plan, evidence, 1)

    assert coverage == {
        "sufficient": True,
        "covered_aspects": ["architecture", "deployment"],
        "missing_aspects": [],
        "evidence_counts": {"architecture": 1, "deployment": 1},
        "adjustment_recommended": False,
    }


def test_evidence_coverage_identifies_one_missing_research_aspect_and_adjusts_only_it():
    plan = [
        {"query": "architecture", "subject": None, "aspect": "architecture"},
        {"query": "deployment", "subject": None, "aspect": "deployment"},
    ]
    evidence = [{
        "query": "architecture", "subject": None, "aspect": "architecture", "chunk_ids": ["a"],
    }]

    coverage = evaluate_evidence_coverage("research", plan, evidence, 1)
    adjustments = build_adjustment_queries(plan, coverage["missing_aspects"])

    assert coverage["covered_aspects"] == ["architecture"]
    assert coverage["missing_aspects"] == ["deployment"]
    assert coverage["adjustment_recommended"] is True
    assert len(adjustments) == 1
    assert adjustments[0]["aspect"] == "deployment"
    assert adjustments[0]["query"] != "deployment"


def test_comparison_coverage_requires_each_subject_and_ignores_duplicate_chunk_ids():
    plan = [
        {"query": "Angular features", "subject": "Angular", "aspect": "Angular"},
        {"query": "React features", "subject": "React", "aspect": "React"},
        {"query": "Angular and React differences", "subject": None, "aspect": "comparison"},
    ]
    evidence = [
        {"query": "Angular features", "subject": "Angular", "aspect": "Angular", "chunk_ids": ["a", "a"]},
        {"query": "React features", "subject": "React", "aspect": "React", "chunk_ids": ["b"]},
    ]

    complete = evaluate_evidence_coverage("comparison", plan, evidence, 1)
    partial = evaluate_evidence_coverage("comparison", plan, evidence[:1], 1)

    assert complete["sufficient"] is True
    assert complete["evidence_counts"] == {"Angular": 1, "React": 1}
    assert partial["sufficient"] is False
    assert partial["missing_aspects"] == ["React"]
    adjustments = build_adjustment_queries(plan, partial["missing_aspects"])
    assert len(adjustments) == 1
    assert adjustments[0]["subject"] == "React"


def test_empty_or_unidentified_comparison_evidence_is_not_called_sufficient():
    plan = [{"query": "Compare these two projects", "subject": None, "aspect": "comparison subjects"}]
    evidence = [{
        "query": "Compare these two projects", "subject": None,
        "aspect": "comparison subjects", "chunk_ids": ["a"],
    }]

    coverage = evaluate_evidence_coverage("comparison", plan, evidence, 1)
    empty = evaluate_evidence_coverage("research", [
        {"query": "architecture", "subject": None, "aspect": "architecture"},
    ], [], 1)

    assert coverage["sufficient"] is False
    assert coverage["missing_aspects"] == ["comparison subjects"]
    assert empty["sufficient"] is False
    assert empty["evidence_counts"] == {"architecture": 0}


def test_chunks_excluded_from_budgeted_context_do_not_count_as_usable_coverage():
    plan = [{"query": "architecture", "subject": None, "aspect": "architecture"}]
    evidence = [{
        "query": "architecture", "subject": None, "aspect": "architecture", "chunk_ids": ["retrieved"]
    }]

    coverage = evaluate_evidence_coverage(
        "research", plan, evidence, retrieval_round=1, usable_chunk_text={},
    )

    assert coverage["sufficient"] is False
    assert coverage["evidence_counts"] == {"architecture": 0}
    assert coverage["missing_aspects"] == ["architecture"]


def test_comparison_evidence_for_one_subject_does_not_cover_another_subject():
    plan = [
        {"query": "Angular details", "subject": "Angular", "aspect": "Angular"},
        {"query": "React details", "subject": "React", "aspect": "React"},
    ]
    evidence = [
        {"query": "Angular details", "subject": "Angular", "aspect": "Angular", "chunk_ids": ["angular"]},
        {"query": "React details", "subject": "React", "aspect": "React", "chunk_ids": ["angular"]},
    ]

    coverage = evaluate_evidence_coverage(
        "comparison", plan, evidence, 1, {"angular": "Angular is used for component development."},
    )

    assert coverage["covered_aspects"] == ["Angular"]
    assert coverage["missing_aspects"] == ["React"]
    assert coverage["evidence_counts"] == {"Angular": 1, "React": 0}


def test_second_evaluation_never_recommends_a_third_retrieval_round():
    plan = [{"query": "deployment", "subject": None, "aspect": "deployment"}]
    coverage = evaluate_evidence_coverage("research", plan, [], retrieval_round=2)

    assert coverage["sufficient"] is False
    assert coverage["adjustment_recommended"] is False


def test_graph_makes_one_adjustment_only_for_missing_aspects_and_rechecks_coverage(monkeypatch):
    chunks = {
        "architecture": RetrievedChunk("doc", "source.txt", "arch", 0, 1, "Architecture is documented.", 0.9),
        "retrieval methods": RetrievedChunk("doc", "source.txt", "retrieval", 1, 2, "Retrieval methods are documented.", 0.8),
        "deployment technologies": RetrievedChunk("doc", "source.txt", "deployment", 2, 3, "Deployment technologies are documented.", 0.7),
    }
    p = make_pipeline(records=[])
    p.settings.rag_context_token_budget = 1280
    calls = []

    def retrieve(question, document_ids, top_k):
        calls.append((question, document_ids, top_k))
        if "architecture" in question:
            return [chunks["architecture"]], {"candidate_count": 1}
        if "specific supporting details" in question and "retrieval methods" in question:
            return [chunks["retrieval methods"]], {"candidate_count": 1}
        if "specific supporting details" in question and "deployment technologies" in question:
            return [chunks["deployment technologies"]], {"candidate_count": 1}
        return [], {"candidate_count": 0}

    monkeypatch.setattr(p, "retrieve", retrieve)
    state = build_rag_graph(p).invoke({
        "question": "Research the architecture, retrieval methods, and deployment technologies used in KnowSphere AI.",
        "document_ids": ["only-this-document"],
        "top_k": 7,
    })

    assert len(calls) == 5
    assert all(document_ids == ["only-this-document"] and top_k == 7
               for _, document_ids, top_k in calls)
    assert len({query.casefold() for query, _, _ in calls}) == len(calls)
    assert state["retrieval_round"] == 2
    assert state["retrieval_metadata"]["retrieval_round"] == 2
    assert state["evidence_coverage"]["sufficient"] is True
    assert state["evidence_coverage"]["missing_aspects"] == []
    assert {chunk.chunk_id for chunk in state["retrieved_chunks"]} == {"arch", "retrieval", "deployment"}


def test_insufficient_evidence_after_adjustment_generates_one_grounded_partial_response(monkeypatch):
    source = RetrievedChunk("doc", "source.txt", "arch", 0, 1, "Architecture is documented.", 0.9)

    class CapturingProvider(FakeProvider):
        calls = 0
        prompt = ""

        def generate(self, system_prompt, user_prompt, max_tokens):
            self.calls += 1
            self.prompt = user_prompt
            return self.answer

    answer = "The document describes the architecture."
    output = json.dumps({"answer": answer, "citation_chunk_ids": ["arch"]})
    provider = CapturingProvider(output)
    p = make_pipeline(provider=provider, records=[])
    calls = []

    def retrieve(question, document_ids, top_k):
        calls.append(question)
        return ([source], {"candidate_count": 1}) if "architecture" in question else ([], {"candidate_count": 0})

    monkeypatch.setattr(p, "retrieve", retrieve)
    state = build_rag_graph(p).invoke({
        "question": "Research the architecture, retrieval methods, and deployment technologies used in KnowSphere AI.",
    })

    assert len(calls) == 5
    assert state["retrieval_round"] == 2
    assert state["evidence_coverage"]["missing_aspects"] == [
        "retrieval methods", "deployment technologies",
    ]
    assert state["final_result"]["answer"].startswith(answer)
    assert "The retrieved evidence did not cover: retrieval methods, deployment technologies." in state["final_result"]["answer"]
    assert [citation["chunk_id"] for citation in state["final_result"]["citations"]] == ["arch"]
    assert provider.calls == 1
    assert '"missing_aspects"' in provider.prompt
    assert "retrieval methods" in provider.prompt and "deployment technologies" in provider.prompt


def test_no_evidence_after_adjustment_uses_existing_no_evidence_route(monkeypatch):
    p = make_pipeline(records=[])
    calls = []

    def retrieve(question, document_ids, top_k):
        calls.append(question)
        return [], {"candidate_count": 0}

    monkeypatch.setattr(p, "retrieve", retrieve)
    monkeypatch.setattr(
        p, "generate_response", lambda *args, **kwargs: pytest.fail("empty evidence must not generate"),
    )

    state = build_rag_graph(p).invoke({
        "question": "Research the architecture, retrieval methods, and deployment technologies used in KnowSphere AI.",
    })

    assert len(calls) == 6
    assert state["retrieval_round"] == 2
    assert state["evidence_coverage"]["sufficient"] is False
    assert state["final_result"]["generation_status"] == "no_evidence"
    assert state["final_result"]["answer"] is None
    assert state["final_result"]["citations"] == []


def test_direct_task_has_no_planning_work(monkeypatch):
    p = make_pipeline(provider=None, records=[record()])
    monkeypatch.setattr(
        rag_graph, "plan_queries", lambda *args: pytest.fail("direct task must bypass planning"),
    )

    state = build_rag_graph(p).invoke({"question": "What happened to revenue?"})

    assert state["task_type"] == "direct"
    assert "subqueries" not in state
    assert "task_evidence" not in state


def test_research_retrieval_applies_document_filters_deduplicates_and_keeps_provenance(monkeypatch):
    document_id = "filtered-doc"
    source_chunk = RetrievedChunk(
        document_id, "source.txt", "chunk-a", 0, 1, "Architecture and retrieval facts.", 0.8,
    )
    p = make_pipeline(records=[])
    calls = []

    def retrieve(question, document_ids, top_k):
        calls.append((question, document_ids, top_k))
        return [source_chunk], {"candidate_count": 1}

    monkeypatch.setattr(p, "retrieve", retrieve)
    state = build_rag_graph(p).invoke({
        "question": "Research the architecture, retrieval methods, and deployment technologies used in KnowSphere AI.",
        "document_ids": [document_id],
        "top_k": 4,
    })

    assert len(calls) == 3
    assert all(ids == [document_id] and top_k == 4 for _, ids, top_k in calls)
    assert len(state["retrieved_chunks"]) == 1
    assert state["retrieved_chunks"][0].chunk_id == "chunk-a"
    assert all(item["chunk_ids"] == ["chunk-a"] for item in state["task_evidence"])
    assert state["retrieval_metadata"]["planned_query_count"] == 3


def test_comparison_evidence_remains_associated_with_subject_in_context(monkeypatch):
    angular = RetrievedChunk("doc", "frameworks.txt", "angular", 0, 1, "Angular uses components.", 0.9)
    react = RetrievedChunk("doc", "frameworks.txt", "react", 1, 2, "React uses components.", 0.85)
    p = make_pipeline(records=[])

    def retrieve(question, document_ids, top_k):
        if question.startswith("Angular "):
            chunks = [angular]
        elif question.startswith("React "):
            chunks = [react]
        else:
            chunks = [angular, react]
        return chunks, {"candidate_count": len(chunks)}

    monkeypatch.setattr(p, "retrieve", retrieve)
    state = build_rag_graph(p).invoke({"question": "Compare Angular and React."})

    by_subject = {item["subject"]: item["chunk_ids"] for item in state["task_evidence"]}
    assert by_subject["Angular"] == ["angular"]
    assert by_subject["React"] == ["react"]
    context_entries = {
        entry["chunk_id"]: entry
        for entry in (json.loads(line) for line in state["assembled_context"].splitlines())
    }
    assert "Angular" in context_entries["angular"]["retrieval_scopes"]
    assert "React" in context_entries["react"]["retrieval_scopes"]
    assert {chunk.chunk_id for chunk in state["selected_chunks"]} == {"angular", "react"}


def test_complex_retrieval_uses_one_global_context_budget(monkeypatch):
    chunks = [
        RetrievedChunk("doc", "large.txt", f"chunk-{index}", index, None, "evidence " * 900, 0.9 - index / 100)
        for index in range(3)
    ]
    p = make_pipeline(records=[])

    def retrieve(question, document_ids, top_k):
        index = {"architecture": 0, "retrieval methods": 1, "deployment technologies": 2}
        key = next(key for key in index if key in question)
        chunk = chunks[index[key]]
        return [chunk], {"candidate_count": 1}

    monkeypatch.setattr(p, "retrieve", retrieve)
    state = build_rag_graph(p).invoke({
        "question": "Research the architecture, retrieval methods, and deployment technologies used in KnowSphere AI.",
        "retrieval_only": True,
    })

    assert len(state["retrieved_chunks"]) == 3
    assert len(state["selected_chunks"]) < 3
    assert len({chunk.chunk_id for chunk in state["selected_chunks"]}) == len(state["selected_chunks"])
    assert state["final_result"]["retrieval_metadata"]["context_token_estimate"] <= p.settings.rag_context_token_budget
    assert state["final_result"]["generation_status"] == "retrieval_only"


@pytest.mark.parametrize(
    ("question", "expected_task"),
    [
        ("What happened to revenue?", "direct"),
        ("Investigate the revenue trend and supporting findings.", "research"),
        ("Compare revenue and operating costs.", "comparison"),
        ("Summarize experience, projects, and technical strengths.", "synthesis"),
    ],
)
def test_all_classified_tasks_use_the_existing_rag_path_without_extra_generation(
    question, expected_task,
):
    class CountingProvider(FakeProvider):
        calls = 0

        def generate(self, system_prompt, user_prompt, max_tokens):
            self.calls += 1
            return self.answer

    output = json.dumps({"answer": "Revenue increased by 15 percent.", "citation_chunk_ids": ["doc_0"]})
    provider = CountingProvider(output)
    p = make_pipeline(provider=provider, records=[record()])
    graph_state = build_rag_graph(p).invoke({"question": question})

    assert graph_state["task_type"] == expected_task
    assert graph_state["final_result"]["answer"].startswith("Revenue increased by 15 percent.")
    assert set(graph_state["final_result"]) == {
        "answer", "citations", "retrieved_chunks", "provider", "model",
        "generation_status", "retrieval_metadata",
    }
    assert provider.calls == 1
    if expected_task == "direct":
        assert "evidence_coverage" not in graph_state
    else:
        assert provider.calls == 1
        if graph_state["evidence_coverage"]["sufficient"]:
            assert graph_state["retrieval_round"] == 1
            assert graph_state["retrieval_metadata"]["adjustment_query_count"] == 0
        else:
            assert graph_state["retrieval_round"] == 2
            assert "The retrieved evidence did not cover:" in graph_state["final_result"]["answer"]


def test_graph_uses_resolved_question_for_retrieval_analysis_and_generation(monkeypatch):
    row = record()
    output = json.dumps({"answer": "Revenue increased by 15 percent.", "citation_chunk_ids": ["doc_0"]})
    p = make_pipeline(provider=FakeProvider(output), records=[row])
    questions = {}
    for method_name in ("retrieve_context", "analyze_query", "generate_response"):
        original = getattr(p, method_name)

        def tracked(*args, _original=original, _name=method_name, **kwargs):
            questions[_name] = args[0]
            return _original(*args, **kwargs)

        monkeypatch.setattr(p, method_name, tracked)

    original_question = "What happened in the previous turn?"
    resolved_question = "What happened to revenue in the report?"
    result = build_rag_graph(p).invoke({
        "question": original_question,
        "resolved_question": resolved_question,
        "resolution_status": "resolved_singular_reference",
    })

    assert questions == {
        "retrieve_context": resolved_question,
        "analyze_query": resolved_question,
        "generate_response": resolved_question,
    }
    assert result["final_result"]["answer"] == "Revenue increased by 15 percent."


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
    original_classifier = rag_graph.classify_task

    def tracked_classifier(question):
        events.append("classify")
        return original_classifier(question)

    monkeypatch.setattr(rag_graph, "classify_task", tracked_classifier)

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

    assert events == ["classify", "retrieve", "analyze", "generate", "provider", "parse", "validate", "finalize"]
    assert graph_state["task_type"] == "direct"
    assert graph_state["final_result"]["answer"] == "Revenue increased by 15 percent."
    assert graph_state["final_result"]["retrieval_metadata"]["context_chunk_count"] == 1
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
