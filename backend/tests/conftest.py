from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.factory import get_document_processor, get_file_storage
from app.services.file_storage import FileStorage
from tests.support import FakeProcessor


@pytest.fixture(autouse=True)
def isolate_local_settings(monkeypatch):
    test_config = {**Settings.model_config, "env_file": None}
    monkeypatch.setattr(Settings, "model_config", test_config)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


@pytest.fixture
def api_client(tmp_path) -> Generator[tuple[TestClient, FakeProcessor, sessionmaker], None, None]:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    processor = FakeProcessor()

    def override_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_file_storage] = lambda: FileStorage(
        tmp_path / "uploads", max_upload_size_bytes=1024, max_uncompressed_bytes=8192
    )
    app.dependency_overrides[get_document_processor] = lambda: processor
    try:
        yield TestClient(app), processor, session_factory
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
