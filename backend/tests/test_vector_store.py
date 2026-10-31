from pathlib import Path
from uuid import uuid4
import gc

from app.services.chunking import DocumentChunk
from app.services.vector_store import ChromaVectorStore


def test_chroma_persists_document_chunks_and_supports_deletion(tmp_path: Path) -> None:
    document_id = str(uuid4())
    chunk = DocumentChunk(
        id=str(uuid4()),
        document_id=document_id,
        index=0,
        text="Persistent vector content",
        source_filename="source.pdf",
        page_number=2,
        source_location="page 2",
    )
    collection_name = f"test_{uuid4().hex}"
    persist_directory = tmp_path / "chroma"
    store = ChromaVectorStore(persist_directory, collection_name)
    store.replace_document_chunks(document_id, [chunk], [[0.0, 1.0, 0.0]])
    assert len(store.get_document_chunks(document_id)) == 1
    del store
    gc.collect()

    reopened_store = ChromaVectorStore(persist_directory, collection_name)
    results = reopened_store.get_document_chunks(document_id)
    assert results[0]["text"] == "Persistent vector content"
    assert results[0]["metadata"]["page_number"] == 2

    reopened_store.delete_document(document_id)
    assert reopened_store.get_document_chunks(document_id) == []


def test_chroma_replacement_is_idempotent(tmp_path: Path) -> None:
    document_id = str(uuid4())
    chunk = DocumentChunk(str(uuid4()), document_id, 0, "A chunk", "a.txt", None, "lines 1-1")
    store = ChromaVectorStore(tmp_path / "chroma", f"test_{uuid4().hex}")

    store.replace_document_chunks(document_id, [chunk], [[1.0, 0.0]])
    store.replace_document_chunks(document_id, [chunk], [[1.0, 0.0]])

    assert len(store.get_document_chunks(document_id)) == 1


def test_chroma_semantic_query_filters_document_and_keeps_metadata(tmp_path: Path) -> None:
    wanted, other = str(uuid4()), str(uuid4())
    chunks = [
        DocumentChunk("wanted_chunk", wanted, 2, "Revenue was 15%", "annual.pdf", 8, "page 8"),
        DocumentChunk("other_chunk", other, 0, "Expenses were 5%", "other.pdf", None, "lines 1-2"),
    ]
    store = ChromaVectorStore(tmp_path / "persistent", f"test_{uuid4().hex}")
    store.replace_document_chunks(wanted, [chunks[0]], [[1.0, 0.0]])
    store.replace_document_chunks(other, [chunks[1]], [[0.0, 1.0]])
    result = store.query([1.0, 0.0], 5, [wanted])
    assert [item["id"] for item in result] == ["wanted_chunk"]
    assert result[0]["metadata"]["source_filename"] == "annual.pdf"
    assert result[0]["metadata"]["page_number"] == 8
    assert result[0]["distance"] < 0.01


def test_chroma_lexical_query_finds_names_and_respects_document_filter(tmp_path: Path) -> None:
    wanted, other = str(uuid4()), str(uuid4())
    chunks = [
        DocumentChunk("resume_name", wanted, 0, "ADEN ABISHEK RAJ\nWork Experience",
                      "resume.pdf", 1, "page 1"),
        DocumentChunk("other_name", other, 0, "SARA JOHNSON\nEducation", "other.pdf", 1, "page 1"),
        DocumentChunk("retention", wanted, 1, "Employee retention was 91 percent.",
                      "resume.pdf", 2, "page 2"),
    ]
    store = ChromaVectorStore(tmp_path / "persistent", f"test_{uuid4().hex}")
    store.replace_document_chunks(wanted, [chunks[0], chunks[2]], [[1.0, 0.0], [0.0, 1.0]])
    store.replace_document_chunks(other, [chunks[1]], [[0.5, 0.5]])

    name_results = store.lexical_query(
        "What is the name of the person mentioned in the uploaded document?", 5, [wanted]
    )
    exact_results = store.lexical_query("ADEN ABISHEK RAJ", 5, [wanted])
    keyword_results = store.lexical_query("employee retention", 5, [wanted])
    unrelated_results = store.lexical_query("What is the name of the Martian president?", 5, [wanted])

    assert name_results[0]["id"] == "resume_name"
    assert all(row["metadata"]["document_id"] == wanted for row in name_results)
    assert exact_results[0]["id"] == "resume_name"
    assert keyword_results[0]["id"] == "retention"
    assert unrelated_results == []
