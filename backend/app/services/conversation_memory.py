"""Bounded, conversation-scoped history loading for reference resolution."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.conversation import ChatMessage

_PLACEHOLDER_PREFIXES = (
    "no answer was generated.",
    "answer generation is unavailable because no response could be generated.",
)
CLARIFICATION_RESPONSE = "I couldn't determine what you're referring to. Could you clarify?"


@dataclass(frozen=True)
class ConversationHistoryMessage:
    role: str
    content: str
    is_clarification: bool = False


def _usable_message(message: ChatMessage) -> bool:
    content = (message.content or "").strip()
    if message.role == "user":
        return bool(content)
    if message.role != "assistant" or not content:
        return False
    if message.generation_status == "clarification":
        return content == CLARIFICATION_RESPONSE
    if message.generation_status != "generated":
        return False
    return not any(content.casefold().startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


def load_recent_conversation_history(
    db: Session, conversation_id: UUID, settings: Settings,
) -> list[ConversationHistoryMessage]:
    """Load a bounded chronological suffix of messages belonging to exactly one conversation."""
    recent = list(db.scalars(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(settings.conversation_history_max_messages)
    ).all())

    bounded_newest_first: list[ConversationHistoryMessage] = []
    characters_used = 0
    for message in recent:
        if not _usable_message(message):
            continue
        content = message.content.strip()
        message_cost = len(content) + len(message.role) + 4
        if characters_used + message_cost > settings.conversation_history_max_chars:
            # Keep a contiguous recent window; do not fall back to older, stale antecedents.
            break
        bounded_newest_first.append(ConversationHistoryMessage(
            message.role,
            content,
            message.generation_status == "clarification",
        ))
        characters_used += message_cost

    return list(reversed(bounded_newest_first))
