import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from fastapi import UploadFile


class UploadValidationError(ValueError):
    """An uploaded file does not meet the accepted document constraints."""


class FileTooLargeError(UploadValidationError):
    """The streamed upload exceeded the configured byte limit."""


@dataclass(frozen=True)
class StoredFile:
    original_filename: str
    path: Path
    extension: str
    mime_type: str
    size_bytes: int


MIME_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/x-markdown", "text/plain"},
}


def safe_original_filename(filename: str | None) -> str:
    if not filename:
        raise UploadValidationError("A filename is required")
    base_name = PureWindowsPath(filename.replace("/", "\\")).name
    base_name = PurePosixPath(base_name).name
    base_name = re.sub(r"[\x00-\x1f\x7f]", "", base_name).strip()
    if not base_name or base_name in {".", ".."}:
        raise UploadValidationError("A valid filename is required")
    if len(base_name) > 512:
        raise UploadValidationError("Filename exceeds the 512 character limit")
    return base_name


def validate_upload_metadata(filename: str | None, content_type: str | None) -> tuple[str, str]:
    original_filename = safe_original_filename(filename)
    extension = Path(original_filename).suffix.lower()
    allowed_mime_types = MIME_TYPES.get(extension)
    if allowed_mime_types is None:
        raise UploadValidationError("Unsupported file extension; allowed formats are PDF, DOCX, TXT, and Markdown")
    mime_type = (content_type or "").split(";", 1)[0].strip().lower()
    if mime_type not in allowed_mime_types:
        raise UploadValidationError("Content type does not match a supported format")
    return original_filename, extension


class FileStorage:
    def __init__(self, root: Path, max_upload_size_bytes: int, max_uncompressed_bytes: int) -> None:
        self.root = root.expanduser().resolve()
        self.max_upload_size_bytes = max_upload_size_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes

    async def save_upload(self, upload: UploadFile) -> StoredFile:
        original_filename, extension = validate_upload_metadata(upload.filename, upload.content_type)
        self.root.mkdir(parents=True, exist_ok=True)
        generated_path = self.root / f"{uuid4().hex}{extension}"
        resolved_path = generated_path.resolve()
        if resolved_path.parent != self.root:
            raise UploadValidationError("Invalid upload destination")

        total_size = 0
        try:
            with resolved_path.open("xb") as destination:
                while block := await upload.read(64 * 1024):
                    total_size += len(block)
                    if total_size > self.max_upload_size_bytes:
                        raise FileTooLargeError(
                            f"File exceeds the {self.max_upload_size_bytes} byte upload limit"
                        )
                    destination.write(block)
            if total_size == 0:
                raise UploadValidationError("Empty files are not accepted")
            self._validate_file_signature(resolved_path, extension)
        except Exception:
            resolved_path.unlink(missing_ok=True)
            raise

        return StoredFile(
            original_filename=original_filename,
            path=resolved_path,
            extension=extension,
            mime_type=(upload.content_type or "").split(";", 1)[0].strip().lower(),
            size_bytes=total_size,
        )

    def delete_file(self, storage_path: str | None) -> None:
        if not storage_path:
            return
        path = Path(storage_path).resolve()
        if path.parent != self.root:
            raise ValueError("Stored document path is outside the configured storage directory")
        path.unlink(missing_ok=True)

    def _validate_file_signature(self, path: Path, extension: str) -> None:
        with path.open("rb") as source:
            header = source.read(8)
        if extension == ".pdf" and not header.lstrip().startswith(b"%PDF-"):
            raise UploadValidationError("File content is not a valid PDF")
        if extension == ".docx":
            try:
                with zipfile.ZipFile(path) as archive:
                    entries = archive.infolist()
                    uncompressed_size = sum(entry.file_size for entry in entries)
                    names = {entry.filename for entry in entries}
            except (OSError, zipfile.BadZipFile) as exc:
                raise UploadValidationError("File content is not a valid DOCX document") from exc
            if uncompressed_size > self.max_uncompressed_bytes or len(entries) > 2000:
                raise UploadValidationError("DOCX expands beyond the supported document limits")
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise UploadValidationError("File content is not a valid DOCX document")
        if extension in {".txt", ".md", ".markdown"} and b"\x00" in header:
            raise UploadValidationError("Text documents cannot contain binary data")
