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
