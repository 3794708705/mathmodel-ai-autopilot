"""MathModel AI — File service.

Handles file ingestion, storage, type detection, hashing,
and dispatches to appropriate parsers.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional

from mathmodel.domain.data import FileRecord, ParserStatus

logger = logging.getLogger(__name__)

UNSAFE_EXTENSIONS = {".py", ".sh", ".exe", ".bat", ".cmd", ".ps1", ".dll", ".so"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


class FileService:
    """Service for file ingestion, storage, and parsing."""

    def __init__(self, storage_dir: str | Path = "./data/files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        file_path: str | Path,
        project_id: Optional[str] = None,
        problem_id: Optional[str] = None,
    ) -> FileRecord:
        """Ingest a file: validate, hash, copy to storage, create record."""
        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not src.is_file():
            raise ValueError(f"Not a file: {file_path}")

        self._validate_file(src)

        sha256 = self._hash_file(src)
        safe_name = self._safe_filename(src.name)
        extension = src.suffix.lower()
        media_type = self._detect_media_type(src)

        dest = self.storage_dir / safe_name
        if dest.exists():
            base = dest.stem
            counter = 1
            while dest.exists():
                dest = self.storage_dir / f"{base}_{counter}{extension}"
                counter += 1

        content = src.read_bytes()
        dest.write_bytes(content)

        parser_name = self._dispatch_parser(extension, media_type)

        return FileRecord(
            project_id=project_id,
            problem_id=problem_id,
            original_name=src.name,
            safe_name=safe_name,
            media_type=media_type,
            extension=extension,
            size_bytes=src.stat().st_size,
            sha256=sha256,
            storage_path=str(dest),
            parser_name=parser_name,
            parser_status=ParserStatus.PENDING,
            is_original=True,
        )

    def _validate_file(self, path: Path) -> None:
        name = path.name.lower()
        ext = path.suffix.lower()
        if ext in UNSAFE_EXTENSIONS:
            raise ValueError(f"Unsafe file extension: {ext}")
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE_BYTES})")
        if size == 0:
            raise ValueError("Empty file")
        if ".." in name or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe filename: {name}")

    @staticmethod
    def _safe_filename(original: str) -> str:
        safe = re.sub(r'[<>:"/\\|?*\';\s]', "_", original)
        safe = re.sub(r"_+", "_", safe)
        safe = safe.strip("._")
        return safe if safe else "unnamed_file"

    @staticmethod
    def _hash_file(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _detect_media_type(path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            return mime
        ext = path.suffix.lower()
        ext_map = {
            ".csv": "text/csv", ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp",
            ".zip": "application/zip",
        }
        return ext_map.get(ext, "application/octet-stream")

    @staticmethod
    def _dispatch_parser(extension: str, media_type: str) -> Optional[str]:
        ext = extension.lower()
        if ext == ".csv": return "csv_parser"
        if ext == ".txt": return "txt_parser"
        if ext == ".pdf": return "pdf_parser"
        if ext in (".xlsx", ".xls"): return "excel_parser"
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"): return "image_registrar"
        return None