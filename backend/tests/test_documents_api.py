from uuid import UUID

from app.models.knowledge_document import DocumentStatus, KnowledgeDocument
from app.services.document_tasks import release_document_task, reserve_document_task


def markdown_file(filename: str = "notes.md", text: bytes = b"# Knowledge\n\nUseful content."):
    return {"file": (filename, text, "text/markdown")}


def test_upload_list_retrieve_and_delete_document(api_client, tmp_path) -> None:
    client, processor, _ = api_client
    response = client.post("/api/v1/documents/upload", files=markdown_file("../notes.md"))

    assert response.status_code == 202
    uploaded = response.json()
    document_id = uploaded["id"]
    assert UUID(document_id)
    assert uploaded["original_filename"] == "notes.md"
    assert uploaded["status"] == "uploaded"
    assert uploaded["mime_type"] == "text/markdown"
    assert "storage_path" not in uploaded
    completed = client.get(f"/api/v1/documents/{document_id}").json()
    assert completed["status"] == "completed"
    assert completed["chunk_count"] > 0
    assert document_id in processor.vectors

    listing = client.get("/api/v1/documents")
    retrieved = client.get(f"/api/v1/documents/{document_id}")
    assert listing.status_code == 200
    assert len(listing.json()) == 1
    assert retrieved.json()["id"] == document_id

    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 404
    assert document_id not in processor.vectors
    assert list((tmp_path / "uploads").iterdir()) == []


def test_reindex_retries_after_a_recorded_processing_failure(api_client) -> None:
    client, processor, _ = api_client
    upload_response = client.post("/api/v1/documents/upload", files=markdown_file())
    document_id = upload_response.json()["id"]
    assert client.get(f"/api/v1/documents/{document_id}").json()["status"] == "completed"

    processor.fail_next = True
    failed_reindex = client.post(f"/api/v1/documents/{document_id}/reindex")
    assert failed_reindex.status_code == 202
    failed_document = client.get(f"/api/v1/documents/{document_id}").json()
    assert failed_document["status"] == "failed"
    assert failed_document["error_message"]
    assert document_id not in processor.vectors

    retried = client.post(f"/api/v1/documents/{document_id}/reindex")
    assert retried.status_code == 202
    recovered = client.get(f"/api/v1/documents/{document_id}").json()
    assert recovered["status"] == "completed"
    assert recovered["error_message"] is None
    assert processor.calls[document_id] == 3


def test_upload_rejects_empty_invalid_and_oversized_files(api_client) -> None:
    client, _, _ = api_client
    assert client.post("/api/v1/documents/upload", files=markdown_file("empty.md", b"")).status_code == 422
    assert client.post(
        "/api/v1/documents/upload", files={"file": ("fake.pdf", b"not pdf", "application/pdf")}
    ).status_code == 422
    assert client.post(
        "/api/v1/documents/upload", files={"file": ("notes.txt", b"hello", "application/pdf")}
    ).status_code == 422
    assert client.post(
        "/api/v1/documents/upload", files=markdown_file("large.md", b"x" * 1025)
    ).status_code == 413


def test_multiple_upload_returns_per_file_results(api_client) -> None:
    client, _, _ = api_client
    response = client.post(
        "/api/v1/documents/upload/multiple",
        files=[
            ("files", ("valid.md", b"valid text", "text/markdown")),
            ("files", ("bad.exe", b"bad", "application/octet-stream")),
        ],
    )

    assert response.status_code == 202
    body = response.json()
    assert len(body["accepted"]) == 1
    assert body["accepted"][0]["status"] == "uploaded"
    document_id = body["accepted"][0]["id"]
    assert client.get(f"/api/v1/documents/{document_id}").json()["status"] == "completed"
    assert len(body["rejected"]) == 1


def test_missing_document_and_missing_source_are_reported(api_client, tmp_path) -> None:
    client, _, _ = api_client
    assert client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000").status_code == 404
    uploaded = client.post("/api/v1/documents/upload", files=markdown_file()).json()
    source = next((tmp_path / "uploads").iterdir())
    source.unlink()

    assert client.post(f"/api/v1/documents/{uploaded['id']}/reindex").status_code == 409


def test_reindex_recovers_processing_state_left_by_a_previous_process(api_client, tmp_path) -> None:
    client, processor, session_factory = api_client
    source = tmp_path / "uploads" / "stale.md"
    source.parent.mkdir(parents=True)
    source.write_text("Recoverable processing source", encoding="utf-8")
    document = KnowledgeDocument(
        title="stale.md",
        original_filename="stale.md",
        storage_path=str(source),
        file_extension=".md",
        content_type="text/markdown",
        file_size_bytes=30,
        status=DocumentStatus.PROCESSING,
    )
    with session_factory() as db:
        db.add(document)
        db.commit()
        document_id = str(document.id)

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 202
    assert client.get(f"/api/v1/documents/{document_id}").json()["status"] == "completed"
    assert processor.calls[document_id] == 1


def test_active_document_task_cannot_be_reindexed_or_deleted(api_client) -> None:
    client, _, _ = api_client
    uploaded = client.post("/api/v1/documents/upload", files=markdown_file()).json()
    document_id = UUID(uploaded["id"])
    assert reserve_document_task(document_id)
    try:
        assert client.post(f"/api/v1/documents/{document_id}/reindex").status_code == 409
        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 409
    finally:
        release_document_task(document_id)
