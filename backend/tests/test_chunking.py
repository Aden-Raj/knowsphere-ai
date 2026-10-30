from uuid import uuid4

from app.services.chunking import chunk_sections
from app.services.document_extraction import ExtractedSection
from app.services.text_processing import clean_text


def test_chunks_respect_size_overlap_and_have_stable_ids() -> None:
    document_id = uuid4()
    text = " ".join(f"token-{index:03d}" for index in range(50))
    section = ExtractedSection(text, page_number=3, source_location="page 3")

    first_pass = chunk_sections([section], document_id, "source.pdf", 80, 20)
    second_pass = chunk_sections([section], document_id, "source.pdf", 80, 20)

    assert len(first_pass) > 1
    assert all(len(chunk.text) <= 80 for chunk in first_pass)
    assert [chunk.id for chunk in first_pass] == [chunk.id for chunk in second_pass]
    assert all(chunk.page_number == 3 for chunk in first_pass)
    assert any(
        set(first_pass[index].text.split()[-4:]) & set(first_pass[index + 1].text.split()[:4])
        for index in range(len(first_pass) - 1)
    )


def test_chunk_metadata_records_source_lines() -> None:
    chunks = chunk_sections(
        [ExtractedSection("first line\nsecond line\nthird line")],
        uuid4(),
        "notes.md",
        20,
        2,
    )

    assert chunks
    assert all("lines " in chunk.source_location for chunk in chunks)
    assert chunks[0].source_filename == "notes.md"


def test_chunk_settings_must_be_valid() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunk_sections([ExtractedSection("text")], uuid4(), "a.txt", 10, 10)


def test_text_cleaning_preserves_meaningful_unicode_and_markdown_spacing() -> None:
    cleaned = clean_text("# Title\r\n\r\nA\u00a0B 👩\u200d💻\u0000")

    assert cleaned == "# Title\n\nA B 👩\u200d💻"
