# KnowSphere AI Demo Guide

This guide is a suggested walkthrough, not a report of a live demo or guaranteed model output.

## Prerequisites

- Docker Desktop with Docker Compose.
- Python 3.12 or newer supported by the backend project, and Node.js/npm compatible with the Angular project.
- PostgreSQL configured through the root `.env` and `backend/.env` files.
- For generated answers, a configured LLM provider. Local Ollama is optional; model inference may be slow on CPU. Retrieval-only behavior is available without a provider.
- A small text-based PDF, DOCX, TXT, or Markdown document that is safe to use in a demo. For the follow-up step, use content that explicitly names at least two projects and contains evidence that can answer the follow-up.

## Suggested sequence

1. **Start PostgreSQL.** From the repository root run `docker compose up -d postgres`, then check `docker compose ps` and wait for the health check.
2. **Start the API.** In `backend/`, activate or use the existing virtual environment, run `python -m alembic upgrade head`, then `python -m uvicorn app.main:app --reload`.
3. **Start the frontend.** In `frontend/`, run `npm.cmd start` and open `http://localhost:4200`.
4. **Select the workflow.** Set `RAG_EXECUTION_MODE=langgraph` in `backend/.env` to demonstrate classification and complex workflows; restart the API after changing it. Legacy mode remains the default.
5. **Upload a document.** Open Knowledge Base, upload a supported file, and wait until its status is `completed`. Show the filename, status, chunk count, and document details.
6. **Ask a direct question.** Example: “Which retrieval techniques does the document describe?” Choose a question whose answer is explicitly present in the uploaded material.
7. **Ask a research question.** Example: “Research the architecture, retrieval methods, and deployment approach described in the document.” This is suitable when those distinct aspects are present in the source.
8. **Ask a comparison question.** Example: “Compare the two projects described in the document.” Use a document that explicitly describes both subjects; otherwise the response may identify missing evidence.
9. **Demonstrate follow-up resolution.** Ask: “What projects are mentioned in the document?” Then, if the answer explicitly lists multiple projects, ask: “Which of those projects uses RAG?” Follow-up resolution uses bounded conversation history to interpret the reference, while the answer must still come from fresh document retrieval.
10. **Inspect provenance.** Expand retrieved evidence and open the citation/source cards. Confirm the displayed passages correspond to the answer’s citations.
11. **Show abstention.** Ask about a detail you have checked is absent from the uploaded source. The application should not treat related concepts as proof; inspect the returned status and evidence rather than expecting a particular model wording.

## What to point out

- Document processing status is stored in the application and extracted chunks are indexed for retrieval.
- Retrieval combines ChromaDB semantic candidates with BM25 lexical candidates using Reciprocal Rank Fusion.
- The LangGraph direct route avoids complex planning. Research, comparison, and synthesis use bounded deterministic query planning and at most one retrieval-adjustment round.
- Citations link to current retrieved chunks. Coverage evaluation measures retrieval coverage; it is not a general semantic fact verifier.
- Conversation turns help resolve clear references but are not used as factual evidence.

Avoid showing real personal or confidential documents in a public recording. Use a sample document you are authorized to share.
