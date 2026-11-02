"""Deterministic, evidence-neutral resolution of clear conversational references."""

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from app.services.conversation_memory import ConversationHistoryMessage

ResolutionStatus = Literal[
    "standalone", "resolved_plural_reference", "resolved_singular_reference",
    "resolved_ordinal_reference", "resolved_clarification", "no_history",
    "ambiguous_reference", "unresolved_reference",
]


@dataclass(frozen=True)
class FollowUpResolution:
    original_question: str
    retrieval_question: str
    resolved: bool
    status: ResolutionStatus


_PLURAL_REFERENCE = re.compile(r"\b(?:those|these)\s+(?:projects?|ones?)\b", re.I)
_ORDINAL_REFERENCE = re.compile(r"\bthe\s+(first|second|third|last)\s+(?:one|project)\b", re.I)
_SINGULAR_REFERENCE = re.compile(r"\b(?:that|this)\s+project\b|\b(?:it|its)\b", re.I)
_ITS_COMPONENT = re.compile(r"\bits\s+(backend|frontend|server|client|database|API|interface)\b", re.I)
_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9&.+#-]*(?:\s+[A-Z][A-Za-z0-9&.+#-]*)+\b"
)
_ENTITY_LEADING_STOPWORDS = {
    "the", "a", "an", "at", "for", "by", "from", "with", "my", "project", "projects",
    "resume", "technologies", "candidate",
}
_PROJECT_LIST = re.compile(
    r"\bprojects?(?:\s+(?:mentioned|listed|named|identified))?\s*"
    r"(?::|\b(?:include|includes|are|were|is|was|such as|listed as|mentioned as)\b)"
    r"\s*(?P<items>[^.!?\n]+)",
    re.I,
)
_PROJECTS_HEADING = re.compile(r"^\s*projects?\s*:\s*$", re.I)
_STRUCTURED_SECTION_HEADING = re.compile(
    r"^\s*(?:technologies|technology|skills|tools|stack|features|responsibilities)\s*:", re.I,
)
_PROJECTS_REVERSE = re.compile(
    r"(?P<items>[^.!?\n]+?)\s+(?:is|are|was|were)\s+(?:the\s+)?projects?\b", re.I,
)
_PROJECT_SINGLE_REVERSE = re.compile(
    r"(?P<items>[^.!?\n]+?)\s+is\s+(?:a|the)\s+project\b", re.I,
)
_ORDINAL_INDEX = {"first": 0, "second": 1, "third": 2}
_CLARIFICATION_ENTITY_REPLY = re.compile(
    r"^\s*(?:I\s+mean|I\s+meant|the\s+project\s+is)\s+"
    r"(?P<entity>[^.!?\n]+?)\s*[.!?]?\s*$",
    re.I,
)
_CLARIFICATION_ORDINAL_REPLY = re.compile(
    r"^\s*(?:I\s+mean\s+)?(?:the\s+)?(?P<ordinal>first|second|third|last)\s+"
    r"(?:one|project)\s*[.!?]?\s*$",
    re.I,
)
_WHICH_PROJECTS_USE = re.compile(
    r"^\s*Which\s+of\s+(?:those|these)\s+projects?\s+uses\s+(?P<subject>.+?)\s*\?\s*$",
    re.I,
)


def _project_list_items(content: str) -> list[str]:
    """Extract names only from text explicitly presented as project context."""
    items: list[str] = []
    for match in _PROJECT_LIST.finditer(content):
        value = match.group("items")
        # Stop at an inline transition to another labeled section.
        value = re.split(
            r";\s*(?=(?:technologies|technology|skills|tools|stack|features|responsibilities)\s*:)",
            value, maxsplit=1, flags=re.I,
        )[0]
        items.append(value)
    for match in (*_PROJECTS_REVERSE.finditer(content), *_PROJECT_SINGLE_REVERSE.finditer(content)):
        items.append(match.group("items"))

    lines = content.splitlines()
    for index, line in enumerate(lines):
        if not _PROJECTS_HEADING.fullmatch(line):
            continue
        for following in lines[index + 1:]:
            if _STRUCTURED_SECTION_HEADING.match(following):
                break
            bullet = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", following)
            if bullet:
                items.append(bullet.group(1))
            elif following.strip():
                break
    return items


def _entities_in_explicit_project_context(content: str) -> list[str]:
    entities: list[str] = []
    for item in _project_list_items(content):
        for match in _ENTITY_PATTERN.finditer(item):
            entity = match.group(0).strip()
            while entity.split() and entity.split()[0].casefold() in _ENTITY_LEADING_STOPWORDS:
                entity = " ".join(entity.split()[1:])
            if entity and entity.casefold() not in {value.casefold() for value in entities}:
                entities.append(entity)
    return entities


def _latest_project_entities(history: Sequence[ConversationHistoryMessage]) -> list[str]:
    """Extract antecedents only when the latest assistant turn explicitly identifies projects."""
    for message in reversed(history):
        if message.role != "assistant":
            continue
        entities = _entities_in_explicit_project_context(message.content)
        if entities:
            return entities
        return []
    return []


def _entities_before_pending_clarification(
    history: Sequence[ConversationHistoryMessage],
) -> tuple[str, list[str]] | None:
    """Return the immediately preceding user question and its explicit project list."""
    if not history or history[-1].role != "assistant" or not history[-1].is_clarification:
        return None
    clarification_index = len(history) - 1
    prior_user_index = next(
        (index for index in range(clarification_index - 1, -1, -1)
         if history[index].role == "user"),
        None,
    )
    if prior_user_index is None:
        return None
    prior_assistant = next(
        (message for message in reversed(history[:prior_user_index]) if message.role == "assistant"),
        None,
    )
    if prior_assistant is None:
        return None
    entities = _entities_in_explicit_project_context(prior_assistant.content)
    prior_question = history[prior_user_index].content
    if not (_PLURAL_REFERENCE.search(prior_question)
            or _ORDINAL_REFERENCE.search(prior_question)
            or _SINGULAR_REFERENCE.search(prior_question)):
        return None
    return prior_question, entities


def _rewrite_pending_question(question: str, original_question: str, entity: str) -> str | None:
    """Bind a clarification answer to the prior question without using it as evidence."""
    plural_question = _WHICH_PROJECTS_USE.fullmatch(original_question)
    if plural_question:
        return f"Does {entity} use {plural_question.group('subject')}?"
    plural = _PLURAL_REFERENCE.search(original_question)
    if plural:
        return None
    ordinal = _ORDINAL_REFERENCE.search(original_question)
    if ordinal:
        return original_question[:ordinal.start()] + entity + original_question[ordinal.end():]
    singular = _SINGULAR_REFERENCE.search(original_question)
    if singular:
        resolved = _ITS_COMPONENT.sub(lambda match: f"the {match.group(1)} of {entity}", original_question)
        return _SINGULAR_REFERENCE.sub(entity, resolved)
    return None


def _resolve_clarification_reply(
    question: str, history: Sequence[ConversationHistoryMessage],
) -> FollowUpResolution | None:
    pending = _entities_before_pending_clarification(history)
    if pending is None:
        return None
    original_question, entities = pending
    entity_reply = _CLARIFICATION_ENTITY_REPLY.fullmatch(question)
    ordinal_reply = _CLARIFICATION_ORDINAL_REPLY.fullmatch(question)
    if entity_reply:
        reply = entity_reply.group("entity").strip().strip("'\"“”‘’")
        matches = [entity for entity in entities if entity.casefold() == reply.casefold()]
        if len(matches) > 1 or (entities and not matches):
            return FollowUpResolution(question, question, False, "unresolved_reference")
        entity = matches[0] if matches else reply
    elif ordinal_reply:
        ordinal = ordinal_reply.group("ordinal").casefold()
        index = len(entities) - 1 if ordinal == "last" else _ORDINAL_INDEX[ordinal]
        if index >= len(entities):
            return FollowUpResolution(question, question, False, "ambiguous_reference")
        entity = entities[index]
    else:
        return None

    resolved = _rewrite_pending_question(question, original_question, entity)
    if resolved is None:
        return FollowUpResolution(question, question, False, "unresolved_reference")
    return FollowUpResolution(question, resolved, True, "resolved_clarification")


def resolve_follow_up(
    question: str, history: Sequence[ConversationHistoryMessage],
) -> FollowUpResolution:
    """Resolve only clear references; returned text is a retrieval query, never factual evidence."""
    clarification_resolution = _resolve_clarification_reply(question, history)
    if clarification_resolution is not None:
        return clarification_resolution
    plural = _PLURAL_REFERENCE.search(question)
    ordinal = _ORDINAL_REFERENCE.search(question)
    singular = _SINGULAR_REFERENCE.search(question)
    reference = plural or ordinal or singular
    if reference is None:
        return FollowUpResolution(question, question, False, "standalone")
    if not history:
        return FollowUpResolution(question, question, False, "no_history")

    entities = _latest_project_entities(history)
    if not entities:
        return FollowUpResolution(question, question, False, "unresolved_reference")

    if plural:
        if len(entities) < 2:
            return FollowUpResolution(question, question, False, "ambiguous_reference")
        replacement = "these projects (" + " and ".join(entities) + ")"
        resolved = question[:plural.start()] + replacement + question[plural.end():]
        return FollowUpResolution(question, resolved, True, "resolved_plural_reference")

    if ordinal:
        ordinal_name = ordinal.group(1).casefold()
        index = len(entities) - 1 if ordinal_name == "last" else _ORDINAL_INDEX.get(ordinal_name)
        if index is None or index >= len(entities):
            return FollowUpResolution(question, question, False, "ambiguous_reference")
        entity = entities[index]
        resolved = question[:ordinal.start()] + entity + question[ordinal.end():]
        return FollowUpResolution(question, resolved, True, "resolved_ordinal_reference")

    if len(entities) != 1:
        return FollowUpResolution(question, question, False, "ambiguous_reference")
    entity = entities[0]
    resolved = _ITS_COMPONENT.sub(lambda match: f"the {match.group(1)} of {entity}", question)
    resolved = _SINGULAR_REFERENCE.sub(entity, resolved)
    return FollowUpResolution(question, resolved, True, "resolved_singular_reference")
