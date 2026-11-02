"""LangGraph composition for the existing deterministic and generated RAG stages."""

from collections.abc import Callable, Hashable
from dataclasses import replace
import re
from typing import TypeVar, cast

from langgraph.graph import END, START, StateGraph

from app.services.rag.pipeline import RAGPipeline
from app.services.rag.state import (
    PlannedQuery,
    QueryAnalysisResult,
    RAGGraphState,
    EvidenceCoverage,
    TaskEvidence,
    TaskType,
)

_T = TypeVar("_T")

_COMPARISON_REQUEST = re.compile(
    r"\b(?:compare|contrast|differences?\s+between|differentiate|versus|vs\.?)\b",
    re.IGNORECASE,
)
_SYNTHESIS_REQUEST = re.compile(
    r"\b(?:synthesi[sz]e|combine|consolidate|summari[sz]e|overview)\b", re.IGNORECASE,
)
_SYNTHESIS_SCOPE = re.compile(
    r"\b(?:across\s+(?:multiple|different|all)|from\s+(?:multiple|different|all)|"
    r"multiple\s+(?:documents?|sources?|sections?|topics?|areas?)|"
    r"(?:experience|projects?|skills?|strengths?|areas?|topics?).{0,100}\band\b.{0,100}"
    r"(?:experience|projects?|skills?|strengths?|areas?|topics?))\b",
    re.IGNORECASE,
)
_RESEARCH_REQUEST = re.compile(
    r"\b(?:research|investigat\w*|explor\w*|comprehensive\s+(?:findings|analysis|review)|"
    r"in[- ]depth\s+(?:research|analysis|review)|thorough\s+(?:research|analysis|review))\b",
    re.IGNORECASE,
)
_COMPARISON_GENERIC = re.compile(
    r"^(?:these|those|the)\s+(?:two|three|multiple)\s+(?:projects?|documents?|"
    r"technologies|entities|alternatives|options)$|^(?:them|each other|one another)$",
    re.IGNORECASE,
)
_COMPARISON_FOCUS = re.compile(
    r"\s+(?:in terms of|with respect to|regarding|focusing on|based on)\b.*$", re.I,
)
_PLANNING_ACTION = re.compile(
    r"\b(?:research|investigate|explore|analy[sz]e|review|summari[sz]e|"
    r"synthesi[sz]e|combine|consolidate)\b", re.I,
)
_PLAN_SCOPE = re.compile(
    r"\s+\b(?:used in|used by|across|from|within|for|of|in)\s+(?P<scope>.+)$", re.I,
)


def classify_task(question: str) -> TaskType:
    """Classify explicit task intent conservatively without calling an LLM."""
    if _COMPARISON_REQUEST.search(question):
        return "comparison"
    if _SYNTHESIS_REQUEST.search(question) and (
        _SYNTHESIS_SCOPE.search(question) or re.search(r"\b(?:multiple|various|different)\b", question, re.I)
    ):
        return "synthesis"
    if _RESEARCH_REQUEST.search(question):
        return "research"
    return "direct"


def _split_topic_list(value: str) -> list[str]:
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", value.strip(), flags=re.I)
    return [part.strip(" .?!;:") for part in parts if part.strip(" .?!;:")]


def _comparison_subjects(question: str) -> list[str]:
    match = re.search(
        r"\b(?:compare|contrast)\s+(?P<subjects>.+)|"
        r"\bdifferences?\s+between\s+(?P<subjects2>.+)|"
        r"(?P<subjects3>.+?\s+(?:versus|vs\.?)\s+.+)",
        question.strip().rstrip("?.!"), re.I,
    )
    if not match:
        return []
    phrase = match.group("subjects") or match.group("subjects2") or match.group("subjects3") or ""
    phrase = _COMPARISON_FOCUS.sub("", phrase)
    phrase = re.split(r"\s+(?:in comparison|compared with)\b", phrase, maxsplit=1, flags=re.I)[0]
    subjects = [
        subject.strip(" \t,.;:()[]{}")
        for subject in re.split(r"\s+(?:and|versus|vs\.?)\s+|\s*,\s*", phrase, flags=re.I)
    ]
    subjects = [subject for subject in subjects if subject]
    if len(subjects) < 2 or len(subjects) > 3:
        return []
    if any(_COMPARISON_GENERIC.fullmatch(subject) for subject in subjects):
        return []
    if any(len(subject) > 80 or len(subject.split()) > 8 for subject in subjects):
        return []
    return list(dict.fromkeys(subjects)) if len({s.casefold() for s in subjects}) == len(subjects) else []


def _topic_queries(question: str) -> list[PlannedQuery]:
    action = _PLANNING_ACTION.search(question)
    if action is None:
        return []
    tail = question[action.end():].strip().rstrip("?.! ")
    tail = re.sub(r"^(?:the|a|an)\s+", "", tail, flags=re.I)
    scope_match = _PLAN_SCOPE.search(tail)
    scope = f"{scope_match.group(0).strip()}" if scope_match else ""
    topics_text = tail[:scope_match.start()].strip(" ,") if scope_match else tail
    topics = _split_topic_list(topics_text)
    if len(topics) < 2:
        return []
    if len(topics) > 3:
        topics = [*topics[:2], " and ".join(topics[2:])]
    return [
        {
            "query": " ".join(part for part in (topic, scope) if part).strip(),
            "subject": None,
            "aspect": topic,
        }
        for topic in topics
        if topic
    ]


def plan_queries(question: str, task_type: TaskType) -> list[PlannedQuery]:
    """Build at most three focused retrieval queries, falling back to the original safely."""
    plan: list[PlannedQuery] = []
    if task_type == "comparison":
        subjects = _comparison_subjects(question)
        if len(subjects) == 2:
            plan = [
                {
                    "query": f"{subject} architecture, features, and strengths",
                    "subject": subject,
                    "aspect": subject,
                }
                for subject in subjects
            ]
            plan.append({
                "query": f"{subjects[0]} and {subjects[1]} key differences",
                "subject": None,
                "aspect": "comparison",
            })
        elif subjects:
            plan = [
                {
                    "query": f"{subject} architecture, features, and strengths",
                    "subject": subject,
                    "aspect": subject,
                }
                for subject in subjects
            ]
    elif task_type in {"research", "synthesis"}:
        plan = _topic_queries(question)

    plan = [
        item for item in plan[:3]
        if isinstance(item.get("query"), str) and item["query"].strip()
    ]
    if len(plan) < 2:
        aspect = "comparison subjects" if task_type == "comparison" else question
        return [{"query": question, "subject": None, "aspect": aspect}]
    return plan


def evaluate_evidence_coverage(
    task_type: TaskType,
    subqueries: list[PlannedQuery],
    task_evidence: list[TaskEvidence],
    retrieval_round: int,
    usable_chunk_text: dict[str, str] | None = None,
) -> EvidenceCoverage:
    """Measure retrieval coverage by unique chunk association, not factual entailment."""
    if task_type == "comparison":
        subjects = [item["subject"] for item in subqueries if item.get("subject")]
        aspects = list(dict.fromkeys(subjects)) or ["comparison subjects"]
    else:
        aspects = list(dict.fromkeys(item["aspect"] for item in subqueries))
    counts_by_aspect: dict[str, set[str]] = {aspect: set() for aspect in aspects}
    for item in task_evidence:
        aspect = (
            item.get("subject") or item.get("aspect")
            if task_type == "comparison"
            else item.get("aspect")
        )
        if aspect in counts_by_aspect:
            for chunk_id in item["chunk_ids"]:
                if usable_chunk_text is not None and chunk_id not in usable_chunk_text:
                    continue
                if task_type == "comparison" and item.get("subject") and usable_chunk_text is not None:
                    subject_pattern = rf"(?<!\w){re.escape(item['subject'])}(?!\w)"
                    if not re.search(subject_pattern, usable_chunk_text.get(chunk_id, ""), re.I):
                        continue
                counts_by_aspect[aspect].add(chunk_id)

    evidence_counts = {aspect: len(chunk_ids) for aspect, chunk_ids in counts_by_aspect.items()}
    unknown_comparison_subjects = task_type == "comparison" and not any(
        item.get("subject") for item in subqueries
    )
    covered = [
        aspect for aspect, count in evidence_counts.items()
        if count > 0 and not (unknown_comparison_subjects and aspect == "comparison subjects")
    ]
    missing = [
        aspect for aspect, count in evidence_counts.items()
        if count == 0 or (unknown_comparison_subjects and aspect == "comparison subjects")
    ]
    sufficient = bool(evidence_counts) and not missing
    return {
        "sufficient": sufficient,
        "covered_aspects": covered,
        "missing_aspects": missing,
        "evidence_counts": evidence_counts,
        "adjustment_recommended": bool(missing) and retrieval_round < 2,
    }


def build_adjustment_queries(
    subqueries: list[PlannedQuery], missing_aspects: list[str],
) -> list[PlannedQuery]:
    """Create at most three focused queries for missing aspects without repeating originals."""
    missing = {aspect.casefold() for aspect in missing_aspects}
    initial_queries = {item["query"].strip().casefold() for item in subqueries}
    adjustments: list[PlannedQuery] = []
    seen: set[str] = set()
    for item in subqueries:
        aspect = item.get("subject") or item["aspect"]
        if aspect.casefold() not in missing:
            continue
        if item.get("subject"):
            query = f"{item['subject']} implementation details and supporting evidence"
        else:
            query = f"{item['query']} specific supporting details"
        normalized = query.casefold()
        if normalized in initial_queries or normalized in seen:
            continue
        adjustments.append({"query": query, "subject": item.get("subject"), "aspect": aspect})
        seen.add(normalized)
        if len(adjustments) == 3:
            break
    return adjustments


def _route_after_initial_evaluation(state: RAGGraphState) -> str:
    coverage = state.get("evidence_coverage")
    if coverage and coverage["sufficient"]:
        return "continue"
    if (
        coverage
        and coverage["adjustment_recommended"]
        and state.get("adjustment_queries")
        and state.get("retrieval_round", 1) == 1
    ):
        return "adjust"
    return "partial"


def _route_after_aggregation(state: RAGGraphState) -> str:
    return "evaluate" if state.get("retrieval_round", 1) == 1 else "evaluate_after_adjustment"


class RAGGraphStageError(RuntimeError):
    """Safe graph failure that reports stage metadata without leaking exception details."""

    category = "stage_failure"

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"RAG graph stage failed: {stage}.")


def _run_stage(stage: str, operation: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    try:
        return operation(*args, **kwargs)
    except Exception:
        raise RAGGraphStageError(stage) from None


def _route_after_analysis(state: RAGGraphState) -> str:
    """Route existing early decisions directly to finalization; otherwise generate."""
    return "finalize" if state.get("query_analysis") is not None else "generate"


def _route_after_classification(state: RAGGraphState) -> str:
    """Send direct requests to the baseline and complex tasks through bounded planning."""
    return "retrieve" if state.get("task_type", "direct") == "direct" else "plan"


def build_rag_graph(pipeline: RAGPipeline):
    """Build a compiled graph that delegates all RAG behavior to ``RAGPipeline``."""

    def classify_node(state: RAGGraphState) -> dict[str, object]:
        question = state.get("resolved_question") or state["question"]
        return {"task_type": _run_stage("classify", classify_task, question)}

    def plan_node(state: RAGGraphState) -> dict[str, object]:
        question = state.get("resolved_question") or state["question"]
        task_type = state.get("task_type", "direct")
        return {"subqueries": _run_stage("plan", plan_queries, question, task_type)}

    def retrieve_node(state: RAGGraphState) -> dict[str, object]:
        retrieval_question = state.get("resolved_question") or state["question"]
        chunks, selected, context, metadata = _run_stage(
            "retrieve", pipeline.retrieve_context,
            retrieval_question, state.get("document_ids"), state.get("top_k"),
        )
        return {
            "retrieved_chunks": chunks,
            "selected_chunks": selected,
            "assembled_context": context,
            "retrieval_metadata": metadata,
        }

    def retrieve_tasks_node(state: RAGGraphState) -> dict[str, object]:
        question = state.get("resolved_question") or state["question"]
        document_ids, top_k = state.get("document_ids"), state.get("top_k")
        subqueries = state.get("subqueries") or [{"query": question, "subject": None}]
        merged = {}
        order: dict[str, int] = {}
        evidence: list[TaskEvidence] = []
        metadata_rows: list[dict[str, object]] = []
        for query_index, task in enumerate(subqueries[:3]):
            query = task["query"]
            chunks, metadata = _run_stage(
                "retrieve", pipeline.retrieve, query, document_ids, top_k,
            )
            metadata_rows.append(metadata)
            chunk_ids: list[str] = []
            for chunk in chunks:
                if chunk.chunk_id not in merged:
                    merged[chunk.chunk_id] = chunk
                    order[chunk.chunk_id] = query_index
                elif chunk.relevance_score > merged[chunk.chunk_id].relevance_score:
                    merged[chunk.chunk_id] = replace(
                        merged[chunk.chunk_id], relevance_score=chunk.relevance_score,
                    )
                if chunk.chunk_id not in chunk_ids:
                    chunk_ids.append(chunk.chunk_id)
            evidence.append({
                "query": query,
                "subject": task.get("subject"),
                "aspect": task["aspect"],
                "chunk_ids": chunk_ids,
            })

        ranked = sorted(
            merged.values(),
            key=lambda chunk: (-chunk.relevance_score, order[chunk.chunk_id], chunk.chunk_index, chunk.chunk_id),
        )
        aggregate_metadata: dict[str, object] = {
            "candidate_count": sum(int(row.get("candidate_count", 0)) for row in metadata_rows),
            "filtered_count": len(ranked),
            "semantic_candidate_count": sum(int(row.get("semantic_candidate_count", 0)) for row in metadata_rows),
            "lexical_candidate_count": sum(int(row.get("lexical_candidate_count", 0)) for row in metadata_rows),
            "top_k": top_k or pipeline.settings.rag_top_k,
            "minimum_relevance": pipeline.settings.rag_min_relevance,
            "ranking_method": "reciprocal_rank_fusion",
            "planned_query_count": len(subqueries[:3]),
            "agentic_retrieval": True,
        }
        return {
            "retrieved_chunks": ranked,
            "task_evidence": evidence,
            "retrieval_metadata": aggregate_metadata,
            "retrieval_round": 1,
        }

    def evaluate_node(state: RAGGraphState) -> dict[str, object]:
        task_type = state.get("task_type", "direct")
        subqueries = state.get("subqueries", [])
        retrieval_round = state.get("retrieval_round", 1)
        coverage = _run_stage(
            "evaluate", evaluate_evidence_coverage,
            task_type, subqueries, state.get("task_evidence", []), retrieval_round,
            {chunk.chunk_id: chunk.text for chunk in state.get("selected_chunks", [])},
        )
        adjustments = (
            build_adjustment_queries(subqueries, coverage["missing_aspects"])
            if coverage["adjustment_recommended"] else []
        )
        if not adjustments:
            coverage = {**coverage, "adjustment_recommended": False}
        metadata = dict(state.get("retrieval_metadata", {}))
        metadata["evidence_coverage"] = coverage
        metadata["retrieval_round"] = retrieval_round
        metadata["adjustment_query_count"] = (
            len(adjustments)
            if retrieval_round == 1
            else int(metadata.get("adjustment_query_count", 0))
        )
        return {
            "evidence_coverage": coverage,
            "adjustment_queries": adjustments,
            "retrieval_metadata": metadata,
        }

    def retrieve_adjustments_node(state: RAGGraphState) -> dict[str, object]:
        document_ids, top_k = state.get("document_ids"), state.get("top_k")
        existing = {chunk.chunk_id: chunk for chunk in state.get("retrieved_chunks", [])}
        order = {chunk_id: index for index, chunk_id in enumerate(existing)}
        task_evidence = list(state.get("task_evidence", []))
        metadata = dict(state.get("retrieval_metadata", {}))
        candidate_count = semantic_count = lexical_count = 0
        for task in state.get("adjustment_queries", [])[:3]:
            chunks, query_metadata = _run_stage(
                "adjustment_retrieve", pipeline.retrieve, task["query"], document_ids, top_k,
            )
            candidate_count += int(query_metadata.get("candidate_count", 0))
            semantic_count += int(query_metadata.get("semantic_candidate_count", 0))
            lexical_count += int(query_metadata.get("lexical_candidate_count", 0))
            chunk_ids: list[str] = []
            next_order = len(order)
            for chunk in chunks:
                if chunk.chunk_id not in existing:
                    existing[chunk.chunk_id] = chunk
                    order[chunk.chunk_id] = next_order
                    next_order += 1
                elif chunk.relevance_score > existing[chunk.chunk_id].relevance_score:
                    existing[chunk.chunk_id] = replace(
                        existing[chunk.chunk_id], relevance_score=chunk.relevance_score,
                    )
                if chunk.chunk_id not in chunk_ids:
                    chunk_ids.append(chunk.chunk_id)
            task_evidence.append({
                "query": task["query"],
                "subject": task.get("subject"),
                "aspect": task["aspect"],
                "chunk_ids": chunk_ids,
            })
        ranked = sorted(
            existing.values(),
            key=lambda chunk: (-chunk.relevance_score, order[chunk.chunk_id], chunk.chunk_index, chunk.chunk_id),
        )
        metadata["candidate_count"] = int(metadata.get("candidate_count", 0)) + candidate_count
        metadata["semantic_candidate_count"] = int(metadata.get("semantic_candidate_count", 0)) + semantic_count
        metadata["lexical_candidate_count"] = int(metadata.get("lexical_candidate_count", 0)) + lexical_count
        metadata["filtered_count"] = len(ranked)
        metadata["retrieval_round"] = 2
        metadata["adjustment_query_count"] = len(state.get("adjustment_queries", [])[:3])
        return {
            "retrieved_chunks": ranked,
            "task_evidence": task_evidence,
            "retrieval_metadata": metadata,
            "retrieval_round": 2,
        }

    def aggregate_node(state: RAGGraphState) -> dict[str, object]:
        chunks = state.get("retrieved_chunks", [])
        scopes: dict[str, list[str]] = {}
        for item in state.get("task_evidence", []):
            label = item.get("subject") or item["query"]
            for chunk_id in item["chunk_ids"]:
                scopes.setdefault(chunk_id, [])
                if label not in scopes[chunk_id]:
                    scopes[chunk_id].append(label)
        selected, context = _run_stage("aggregate", pipeline.assemble_context, chunks, scopes)
        return {"selected_chunks": selected, "assembled_context": context}

    def analyze_node(state: RAGGraphState) -> dict[str, object]:
        retrieval_question = state.get("resolved_question") or state["question"]
        decision = _run_stage(
            "analyze", pipeline.analyze_query,
            retrieval_question, state.get("selected_chunks", []), state.get("retrieval_only", False),
        )
        if (
            decision is not None
            and state.get("task_type", "direct") != "direct"
            and not state.get("retrieval_only", False)
            and decision[2] == "generated"
        ):
            # Deterministic single-purpose answers must not short-circuit a planned multi-aspect task.
            decision = None
        if decision is None:
            return {"query_analysis": None}
        answer, citations, status = decision
        analysis: QueryAnalysisResult = {
            "answer": answer,
            "citations": citations,
            "generation_status": status,
        }
        return {"query_analysis": analysis}

    def generate_node(state: RAGGraphState) -> dict[str, object]:
        retrieval_question = state.get("resolved_question") or state["question"]
        coverage = state.get("evidence_coverage")
        generation_options = (
            {"evidence_coverage": coverage}
            if coverage and coverage["missing_aspects"] else {}
        )
        answer, citations, status = _run_stage(
            "generate", pipeline.generate_response,
            retrieval_question, state.get("selected_chunks", []), state.get("assembled_context", ""),
            **generation_options,
        )
        if answer and status == "generated" and coverage and coverage["missing_aspects"]:
            missing = [
                re.sub(r"\s+", " ", aspect).strip(" .;:")[:120]
                for aspect in coverage["missing_aspects"][:3]
                if isinstance(aspect, str) and aspect.strip()
            ]
            if missing:
                answer = f"{answer.rstrip()}\n\nThe retrieved evidence did not cover: {', '.join(missing)}."
        citation_ids = [
            citation["chunk_id"]
            for citation in citations
            if isinstance(citation.get("chunk_id"), str)
        ]
        return {
            "generated_answer": answer,
            "citation_chunk_ids": citation_ids,
            "validated_citations": citations,
            "generation_status": status,
            "error_category": status if status in {"provider_error", "invalid_response"} else None,
        }

    def finalize_node(state: RAGGraphState) -> dict[str, object]:
        analysis = state.get("query_analysis")
        if analysis is not None:
            answer = analysis["answer"]
            citations = analysis["citations"]
            status = analysis["generation_status"]
        else:
            answer = state.get("generated_answer")
            citations = state.get("validated_citations", [])
            status = state.get("generation_status", "invalid_response")

        result = _run_stage(
            "finalize", pipeline.build_result,
            state.get("retrieved_chunks", []),
            state.get("selected_chunks", []),
            state.get("assembled_context", ""),
            state.get("retrieval_metadata", {}),
            answer,
            citations,
            status,
        )
        provider = getattr(pipeline.provider, "name", None)
        model = getattr(pipeline.provider, "model", None)
        return {
            "provider": provider,
            "model": model,
            "citation_chunk_ids": [
                citation["chunk_id"]
                for citation in citations
                if isinstance(citation.get("chunk_id"), str)
            ],
            "validated_citations": citations,
            "generation_status": status,
            "error_category": status if status in {"provider_error", "invalid_response"} else None,
            "final_result": result,
        }

    builder = StateGraph(RAGGraphState)
    builder.add_node("classify", classify_node)
    builder.add_node("plan", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("retrieve_tasks", retrieve_tasks_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("retrieve_adjustments", retrieve_adjustments_node)
    builder.add_node("evaluate_after_adjustment", evaluate_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classification,
        cast(dict[Hashable, str], {"retrieve": "retrieve", "plan": "plan"}),
    )
    builder.add_edge("plan", "retrieve_tasks")
    builder.add_edge("retrieve_tasks", "aggregate")
    builder.add_conditional_edges(
        "aggregate",
        _route_after_aggregation,
        cast(dict[Hashable, str], {
            "evaluate": "evaluate",
            "evaluate_after_adjustment": "evaluate_after_adjustment",
        }),
    )
    builder.add_conditional_edges(
        "evaluate",
        _route_after_initial_evaluation,
        cast(dict[Hashable, str], {
            "continue": "analyze",
            "adjust": "retrieve_adjustments",
            "partial": "analyze",
        }),
    )
    builder.add_edge("retrieve_adjustments", "aggregate")
    builder.add_edge("evaluate_after_adjustment", "analyze")
    builder.add_edge("retrieve", "analyze")
    builder.add_conditional_edges(
        "analyze",
        _route_after_analysis,
        cast(dict[Hashable, str], {"generate": "generate", "finalize": "finalize"}),
    )
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
