from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.core.config import Settings
from app.models.conversation import ChatMessage, Conversation
from app.services.conversation_memory import load_recent_conversation_history


def _store_messages(session_factory, conversation_id, messages):
    with session_factory() as db:
        db.add(Conversation(id=conversation_id, title="History test"))
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index, message_fields in enumerate(messages):
            role, content, status = message_fields[:3]
            provider = message_fields[3] if len(message_fields) > 3 else None
            db.add(ChatMessage(
                id=uuid4(), conversation_id=conversation_id, role=role, content=content,
                citations=[], generation_status=status, provider=provider,
                created_at=base + timedelta(seconds=index),
            ))
        db.commit()


def test_history_is_conversation_scoped_and_returned_chronologically(api_client):
    _, _, session_factory = api_client
    target_id, other_id = uuid4(), uuid4()
    _store_messages(session_factory, target_id, [
        ("user", "First question", None),
        ("assistant", "First answer.", "generated"),
    ])
    _store_messages(session_factory, other_id, [
        ("assistant", "Secret from another conversation.", "generated"),
    ])

    with session_factory() as db:
        history = load_recent_conversation_history(db, target_id, Settings(_env_file=None))

    assert [(message.role, message.content) for message in history] == [
        ("user", "First question"), ("assistant", "First answer."),
    ]
    assert all("another conversation" not in message.content for message in history)


def test_history_uses_configured_latest_message_limit(api_client):
    _, _, session_factory = api_client
    conversation_id = uuid4()
    _store_messages(session_factory, conversation_id, [
        ("user", f"Question {index}", None) for index in range(12)
    ])
    settings = Settings(_env_file=None, conversation_history_max_messages=4)

    with session_factory() as db:
        history = load_recent_conversation_history(db, conversation_id, settings)

    assert [message.content for message in history] == [
        "Question 8", "Question 9", "Question 10", "Question 11",
    ]


def test_history_excludes_failed_placeholder_and_empty_assistant_messages(api_client):
    _, _, session_factory = api_client
    conversation_id = uuid4()
    _store_messages(session_factory, conversation_id, [
        ("user", "Question", None),
        ("assistant", "A valid answer mentioning KnowSphere AI.", "generated"),
        ("user", "Follow-up", None),
        ("assistant", "No answer was generated. See generation_status and retrieved evidence.", "invalid_response"),
        ("assistant", "Answer generation is unavailable because no response could be generated.", "provider_unavailable"),
        ("assistant", "   ", "generated"),
    ])

    with session_factory() as db:
        history = load_recent_conversation_history(db, conversation_id, Settings(_env_file=None))

    assert [(message.role, message.content) for message in history] == [
        ("user", "Question"),
        ("assistant", "A valid answer mentioning KnowSphere AI."),
        ("user", "Follow-up"),
    ]


def test_history_includes_only_explicitly_marked_clarification_responses(api_client):
    _, _, session_factory = api_client
    conversation_id = uuid4()
    _store_messages(session_factory, conversation_id, [
        ("user", "Which project?", None),
        ("assistant", "Could you clarify?", "no_evidence"),
        ("assistant", "I couldn't determine what you're referring to. Could you clarify?", "clarification"),
        ("assistant", "Different text with a clarification status.", "clarification"),
        ("assistant", "The project is KnowSphere AI.", "generated"),
    ])

    with session_factory() as db:
        history = load_recent_conversation_history(db, conversation_id, Settings(_env_file=None))

    assert [(message.role, message.content, message.is_clarification) for message in history] == [
        ("user", "Which project?", False),
        ("assistant", "I couldn't determine what you're referring to. Could you clarify?", True),
        ("assistant", "The project is KnowSphere AI.", False),
    ]


def test_history_respects_character_budget_without_using_older_noncontiguous_context(api_client):
    _, _, session_factory = api_client
    conversation_id = uuid4()
    _store_messages(session_factory, conversation_id, [
        ("user", "Old context that must not be selected." * 12, None),
        ("assistant", "Recent project context: KnowSphere AI.", "generated"),
        ("user", "Latest follow-up", None),
    ])
    settings = Settings(_env_file=None, conversation_history_max_chars=256)

    with session_factory() as db:
        history = load_recent_conversation_history(db, conversation_id, settings)

    assert [message.content for message in history] == [
        "Recent project context: KnowSphere AI.", "Latest follow-up",
    ]
    assert sum(len(message.content) + len(message.role) + 4 for message in history) <= 256
