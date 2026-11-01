from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "KnowSphere AI API"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://localhost:5432/knowsphere"
    cors_origins: list[str] = ["http://localhost:4200"]
    upload_dir: Path = Path("data/uploads")
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "knowsphere_documents"
    max_upload_size_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_upload_files: int = Field(default=20, ge=1, le=100)
    max_uncompressed_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_extracted_chars: int = Field(default=5_000_000, gt=0)
    max_document_pages: int = Field(default=2000, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    huggingface_cache_dir: Path = Path("data/huggingface/hub")
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_min_relevance: float = Field(default=0.20, ge=0, le=1)
    rag_context_token_budget: int = Field(default=4000, ge=128, le=100000)
    rag_execution_mode: Literal["legacy", "langgraph"] = "legacy"
    llm_provider: Literal["none", "ollama", "openai_compatible"] = "none"
    llm_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=180, gt=0, le=600)
    llm_max_tokens: int = Field(default=800, ge=1, le=32000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
