"""Small, dependency-light HTML parser with source safety limits."""

from __future__ import annotations

from html.parser import HTMLParser as StdlibHTMLParser
from pathlib import Path
from typing import Any

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


class _TextExtractor(StdlibHTMLParser):
    """Extract visible text while ignoring executable and presentation elements."""

    _ignored = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in self._ignored:
            self._ignored_depth += 1
        elif lowered == "title":
            self._in_title = True
        elif not self._ignored_depth and lowered in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._ignored and self._ignored_depth:
            self._ignored_depth -= 1
        elif lowered == "title":
            self._in_title = False
        elif not self._ignored_depth and lowered in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)


class HTMLParser:
    """Parse HTML files into chunks with stable citation metadata."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        *,
        allowed_root: str | Path | None = None,
        max_file_bytes: int = 5_000_000,
        max_text_chars: int = 1_000_000,
    ) -> None:
        validate_chunking(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.allowed_root = Path(allowed_root).expanduser().resolve() if allowed_root else None
        self.max_file_bytes = validate_limit("max_file_bytes", max_file_bytes)
        self.max_text_chars = validate_limit("max_text_chars", max_text_chars)

    def parse(self, html_path: str | Path) -> list[dict[str, Any]]:
        """Parse one UTF-8 HTML document and return bounded chunks."""

        path = safe_path(html_path, self.allowed_root, (".html", ".htm"))
        file_within_limit(path, self.max_file_bytes)
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                raise DocumentParseError("HTML contains a NUL byte")
            text = raw.decode("utf-8")
            extractor = _TextExtractor()
            extractor.feed(text)
            extractor.close()
        except DocumentParseError:
            raise
        except (UnicodeDecodeError, ValueError) as exc:
            raise DocumentParseError(f"invalid UTF-8 HTML: {path.name}") from exc
        except Exception as exc:
            raise DocumentParseError(f"invalid HTML: {path.name}") from exc

        visible_text = " ".join(extractor.parts)
        if len(visible_text) > self.max_text_chars:
            raise DocumentParseError(
                f"HTML exceeds max_text_chars ({len(visible_text)} > {self.max_text_chars})"
            )
        chunks = chunk_text(visible_text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise DocumentParseError(f"HTML contains no visible text: {path.name}")

        source = source_name(path, self.allowed_root)
        document = source
        title = " ".join(extractor.title_parts).strip() or document
        return [
            {
                "id": f"{document}#chunk-{number:04d}",
                "text": chunk,
                "metadata": {
                    "document": document,
                    "source": source,
                    "source_type": "html",
                    "html_file": path.name,
                    "page": None,
                    "title": title,
                    "chunk": f"{document}#chunk-{number:04d}",
                },
            }
            for number, chunk in enumerate(chunks, start=1)
        ]

    def parse_directory(self, dir_path: str | Path) -> list[dict[str, Any]]:
        """Parse all regular HTML files below ``dir_path`` in stable path order."""

        root = Path(dir_path).expanduser().resolve()
        if not root.is_dir():
            raise DocumentParseError(f"source directory does not exist: {dir_path}")
        previous_root = self.allowed_root
        self.allowed_root = root
        try:
            paths = sorted(
                (
                    candidate
                    for candidate in root.rglob("*")
                    if candidate.is_file() and candidate.suffix.lower() in {".html", ".htm"}
                ),
                key=lambda candidate: candidate.relative_to(root).as_posix(),
            )
            chunks: list[dict[str, Any]] = []
            for path in paths:
                chunks.extend(self.parse(path))
            return chunks
        finally:
            self.allowed_root = previous_root
