"""Persistent index build, integrity validation, and hybrid retrieval wiring."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parsers.common import DocumentParseError
from parsers.html_parser import HTMLParser
from parsers.pdf_parser import PDFParser

from .bm25 import BM25Retrieval
from .errors import IndexCorruptionError
from .faiss_store import FaissRetrieval
from .hybrid import HybridRetrieval

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexCorruptionError(f"invalid index JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IndexCorruptionError(f"index payload must be an object: {path}")
    return value


@dataclass(frozen=True)
class BuildResult:
    """Summary returned by an idempotent rebuild."""

    document_count: int
    chunk_count: int
    index_path: str


class IndexBuilder:
    """Parse configured sources and atomically publish local retrieval indexes."""

    def __init__(
        self,
        source_path: str | Path,
        index_path: str | Path,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        max_file_bytes: int = 5_000_000,
        max_pages: int = 100,
        max_text_chars: int = 1_000_000,
        embedding_model: str = "local-hash-384",
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_bytes = max_file_bytes
        self.max_pages = max_pages
        self.max_text_chars = max_text_chars
        self.embedding_model = embedding_model

    def _parse_sources(self) -> list[dict[str, Any]]:
        if not self.source_path.is_dir():
            raise DocumentParseError(f"source directory does not exist: {self.source_path}")
        pdf_parser = PDFParser(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            allowed_root=self.source_path,
            max_file_bytes=self.max_file_bytes,
            max_pages=self.max_pages,
            max_text_chars=self.max_text_chars,
        )
        html_parser = HTMLParser(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            allowed_root=self.source_path,
            max_file_bytes=self.max_file_bytes,
            max_text_chars=self.max_text_chars,
        )
        chunks: list[dict[str, Any]] = []
        for path in sorted(
            (
                candidate
                for candidate in self.source_path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".pdf", ".html", ".htm"}
            ),
            key=lambda candidate: candidate.relative_to(self.source_path).as_posix(),
        ):
            if path.suffix.lower() == ".pdf":
                chunks.extend(pdf_parser.parse(path))
            else:
                chunks.extend(html_parser.parse(path))
        if not chunks:
            raise DocumentParseError(f"no PDF or HTML sources found in {self.source_path}")
        # Parser order and chunk ids are already stable; sorting protects against a
        # future parser that returns pages in a different order.
        return sorted(chunks, key=lambda chunk: str(chunk["id"]))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def build(self) -> BuildResult:
        records = self._parse_sources()
        # A sibling temporary directory means no partially-written index can be
        # mistaken for the final directory by a reader.
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.index_path.name}-", dir=self.index_path.parent)
        )
        try:
            documents_path = temporary / "documents.json"
            bm25_path = temporary / "bm25" / "index.json"
            vector_path = temporary / "faiss" / "index.json"
            self._write_json(
                documents_path,
                {"schema_version": SCHEMA_VERSION, "kind": "documents", "documents": records},
            )
            bm25 = BM25Retrieval(bm25_path)
            bm25.build_index(records, bm25_path)
            vector = FaissRetrieval(vector_path, embedding_model=self.embedding_model)
            vector.build_from_documents(records, vector_path)

            files = {
                "documents.json": _sha256(documents_path),
                "bm25/index.json": _sha256(bm25_path),
                "faiss/index.json": _sha256(vector_path),
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "kind": "winter-garden-index",
                "document_count": len({chunk["metadata"].get("document") for chunk in records}),
                "chunk_count": len(records),
                "files": files,
            }
            self._write_json(temporary / "manifest.json", manifest)

            backup: Path | None = None
            if self.index_path.exists():
                backup = self.index_path.with_name(f".{self.index_path.name}-old")
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(self.index_path, backup)
            try:
                os.replace(temporary, self.index_path)
            except Exception:
                if backup and backup.exists() and not self.index_path.exists():
                    os.replace(backup, self.index_path)
                raise
            temporary = Path(".")  # ownership transferred to index_path
            if backup and backup.exists():
                shutil.rmtree(backup)
        finally:
            if temporary != Path(".") and temporary.exists():
                shutil.rmtree(temporary)
        return BuildResult(
            document_count=len({chunk["metadata"].get("document") for chunk in records}),
            chunk_count=len(records),
            index_path=str(self.index_path),
        )


class IndexManager:
    """Validate an index manifest before exposing hybrid retrieval."""

    def __init__(self, index_path: str | Path, *, embedding_model: str = "local-hash-384") -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.embedding_model = embedding_model
        self.manifest = self._validate_manifest()
        self.hybrid = HybridRetrieval(
            self.index_path / "bm25",
            self.index_path / "faiss",
        )

    def _validate_manifest(self) -> dict[str, Any]:
        manifest_path = self.index_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"index manifest not found: {manifest_path}")
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("kind") != "winter-garden-index"
        ):
            raise IndexCorruptionError("unsupported index manifest schema")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise IndexCorruptionError("index manifest has no file checksums")
        for relative, expected in files.items():
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise IndexCorruptionError("index manifest contains an unsafe path")
            path = self.index_path / relative
            if not path.is_file() or _sha256(path) != expected:
                raise IndexCorruptionError(f"index integrity check failed: {relative}")
        return manifest

    @property
    def chunk_count(self) -> int:
        return int(self.manifest["chunk_count"])

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.hybrid.retrieve(query, top_k)
