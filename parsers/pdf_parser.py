"""Bounded PDF text extraction with citation-friendly metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from utils.logger import get_logger

from .common import (
    DocumentParseError,
    chunk_text,
    file_within_limit,
    safe_path,
    source_name,
    validate_chunking,
    validate_limit,
)

logger = get_logger(__name__)


class PDFParser:
    """Extract text from PDFs while enforcing inexpensive resource limits."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        *,
        allowed_root: str | Path | None = None,
        max_file_bytes: int = 5_000_000,
        max_pages: int = 100,
        max_text_chars: int = 1_000_000,
    ) -> None:
        validate_chunking(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self.max_file_bytes = validate_limit("max_file_bytes", max_file_bytes)
        self.max_pages = validate_limit("max_pages", max_pages)
        self.max_text_chars = validate_limit("max_text_chars", max_text_chars)
        logger.info(
            "PDF parser initialized",
            extra={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )

    def parse(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """Parse one PDF into bounded chunks with page and source metadata."""

        path = safe_path(pdf_path, self.allowed_root, (".pdf",))
        file_within_limit(path, self.max_file_bytes)
        try:
            reader = PdfReader(str(path), strict=True)
            page_count = len(reader.pages)
        except Exception as exc:
            raise DocumentParseError(f"invalid PDF: {path.name}") from exc
        if page_count > self.max_pages:
            raise DocumentParseError(
                f"PDF has too many pages ({page_count} > {self.max_pages})"
            )

        source = source_name(path, self.allowed_root)
        document = source
        chunks: list[dict[str, Any]] = []
        extracted_chars = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                raise DocumentParseError(
                    f"could not extract text from {document}, page {page_number}"
                ) from exc
            extracted_chars += len(text)
            if extracted_chars > self.max_text_chars:
                raise DocumentParseError(
                    f"PDF exceeds max_text_chars ({extracted_chars} > {self.max_text_chars})"
                )
            page_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
            for chunk_number, chunk in enumerate(page_chunks, start=1):
                chunk_id = f"{document}#page-{page_number:04d}-chunk-{chunk_number:04d}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": {
                            "document": document,
                            "source": source,
                            "source_type": "pdf",
                            "pdf_file": path.name,
                            "page": page_number,
                            "page_count": page_count,
                            "chunk": chunk_id,
                        },
                    }
                )
        if not chunks:
            raise DocumentParseError(f"PDF contains no extractable text: {document}")
        return chunks

    def parse_directory(self, dir_path: str | Path) -> list[dict[str, Any]]:
        """Parse all regular PDFs below ``dir_path`` in stable path order."""

        root = Path(dir_path).expanduser().resolve()
        if not root.is_dir():
            raise DocumentParseError(f"source directory does not exist: {dir_path}")
        previous_root = self.allowed_root
        self.allowed_root = root
        try:
            paths = sorted(
                (candidate for candidate in root.rglob("*.pdf") if candidate.is_file()),
                key=lambda candidate: candidate.relative_to(root).as_posix(),
            )
            chunks: list[dict[str, Any]] = []
            for path in paths:
                chunks.extend(self.parse(path))
            return chunks
        finally:
            self.allowed_root = previous_root
