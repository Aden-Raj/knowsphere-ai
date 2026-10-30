# KnowSphere AI

KnowSphere AI is an agentic knowledge intelligence platform. Day 1 establishes its API, database, and frontend foundations; RAG ingestion and agent workflows are intentionally out of scope.

## Architecture

- `backend/`: FastAPI HTTP layer, settings, SQLAlchemy models/session, and tests.
- `frontend/`: Angular single-page application.
- PostgreSQL is the system of record. ChromaDB, LlamaIndex, LangGraph, embeddings, and LLM providers will be introduced in later milestones.
- Configuration is supplied through environment variables. Do not commit `.env` files.

The API is organized by HTTP routes (`app/api`), shared settings (`app/core`), persistence (`app/db`), and domain models (`app/models`). Liveness at `/api/v1/health` does not require database access; readiness at `/api/v1/ready` checks PostgreSQL connectivity.

## Local development

1. Copy `.env.example` to `.env` and `backend/.env.example` to `backend/.env`. Set the same local PostgreSQL password in both files.
2. Start PostgreSQL with `docker compose up -d postgres`.
3. In `backend/`, create a virtual environment, install `pip install -e ".[dev]"`, run `alembic upgrade head`, then start `uvicorn app.main:app --reload`.
4. In `frontend/`, run `npm install` and `npm start`.

Backend tests: from `backend/`, run `pytest`.

On the current Windows machine, port 5432 is used by a local PostgreSQL service and port 5433 by another container. The ignored local env files map this project to port 15432; keep `POSTGRES_PORT` and the backend `DATABASE_URL` port aligned. The example files use the standard PostgreSQL port 5432 for other environments.

The initial schema migration creates the knowledge document table. Application CRUD endpoints are planned for later work; this milestone exposes health/readiness only.
