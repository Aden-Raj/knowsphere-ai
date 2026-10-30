import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.services.file_storage import FileStorage, FileTooLargeError, UploadValidationError


def upload(filename: str, content: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md", "notes.markdown"])
def test_saves_text_upload_under_generated_filename(tmp_path: Path, filename: str) -> None:
    storage = FileStorage(tmp_path / "uploads", 100, 1000)

    stored = asyncio.run(storage.save_upload(upload(f"..\\private\\{filename}", b"hello", "text/plain")))

    assert stored.original_filename == filename
    assert stored.path.parent == (tmp_path / "uploads").resolve()
    assert stored.path.name != filename
    assert stored.path.read_bytes() == b"hello"


def test_rejects_empty_unsupported_mismatched_and_oversized_uploads(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path / "uploads", 4, 1000)

    with pytest.raises(UploadValidationError, match="Empty files"):
        asyncio.run(storage.save_upload(upload("empty.txt", b"", "text/plain")))
    with pytest.raises(UploadValidationError, match="Unsupported file extension"):
        asyncio.run(storage.save_upload(upload("archive.exe", b"data", "application/octet-stream")))
    with pytest.raises(UploadValidationError, match="Content type"):
        asyncio.run(storage.save_upload(upload("notes.txt", b"data", "application/pdf")))
    with pytest.raises(FileTooLargeError):
        asyncio.run(storage.save_upload(upload("large.txt", b"12345", "text/plain")))


def test_rejects_fake_pdf_and_path_deletion_outside_storage(tmp_path: Path) -> None:
    storage = FileStorage(tmp_path / "uploads", 100, 1000)

    with pytest.raises(UploadValidationError, match="not a valid PDF"):
        asyncio.run(storage.save_upload(upload("fake.pdf", b"not a pdf", "application/pdf")))
    outside = tmp_path / "valuable.txt"
    outside.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        storage.delete_file(str(outside))
    assert outside.exists()
