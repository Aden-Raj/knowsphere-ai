from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import get_db

client = TestClient(app)


def test_health_reports_liveness() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_database_connection() -> None:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    test_session = sessionmaker(bind=engine)

    def override_get_db():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = client.get("/api/v1/ready")
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "connected"}


def test_readiness_returns_service_unavailable_when_database_fails() -> None:
    class BrokenSession:
        def execute(self, _statement):
            raise RuntimeError("Database is unavailable")

    def unavailable_db():
        yield BrokenSession()

    app.dependency_overrides[get_db] = unavailable_db
    try:
        response = client.get("/api/v1/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
