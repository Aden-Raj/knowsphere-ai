import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_llm_timeout_defaults_to_180_seconds_without_local_env(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_timeout_seconds == 180
    assert settings.llm_max_tokens == 800


def test_llm_timeout_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "240")

    settings = Settings(_env_file=None)

    assert settings.llm_timeout_seconds == 240


def test_rag_execution_mode_defaults_to_legacy_without_local_configuration(monkeypatch):
    monkeypatch.delenv("RAG_EXECUTION_MODE", raising=False)

    settings = Settings(_env_file=None)

    assert settings.rag_execution_mode == "legacy"


def test_rag_execution_mode_accepts_langgraph_environment_override(monkeypatch):
    monkeypatch.setenv("RAG_EXECUTION_MODE", "langgraph")

    settings = Settings(_env_file=None)

    assert settings.rag_execution_mode == "langgraph"


def test_rag_execution_mode_rejects_unknown_values_safely(monkeypatch):
    monkeypatch.setenv("RAG_EXECUTION_MODE", "invalid-mode")

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert "rag_execution_mode" in str(error.value)


def test_conversation_history_limits_have_bounded_defaults():
    settings = Settings(_env_file=None)

    assert settings.conversation_history_max_messages == 10
    assert settings.conversation_history_max_chars == 8000


def test_conversation_history_limits_can_be_configured(monkeypatch):
    monkeypatch.setenv("CONVERSATION_HISTORY_MAX_MESSAGES", "6")
    monkeypatch.setenv("CONVERSATION_HISTORY_MAX_CHARS", "4200")

    settings = Settings(_env_file=None)

    assert settings.conversation_history_max_messages == 6
    assert settings.conversation_history_max_chars == 4200
