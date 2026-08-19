"""Shared safety and chunking helpers for document parsers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


class DocumentParseError(ValueError):
    """Raised when a source cannot be safely or reliably parsed."""


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk limits before any source is opened."""

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("chunk_size must be an integer")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("chunk_overlap must be an integer")
    if chunk_size < 64:
        raise ValueError("chunk_size must be at least 64 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")


def validate_limit(name: str, value: int, minimum: int = 1) -> int:
    """Return an integer limit or raise a useful configuration error."""

    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def safe_path(
    path: str | Path,
    allowed_root: str | Path | None = None,
    extensions: Iterable[str] = (),
) -> Path:
    """Resolve a source path and reject traversal, symlinks outside root, and files.

    A parser can be used against an explicitly supplied file without a root.  When a
    root is configured (the normal directory-ingestion path), the resolved target
    must remain below that root.  Resolving symlinks before the containment check is
    intentional: a symlink inside the source directory must not escape it.
    """

    raw = Path(path).expanduser()
    root = Path(allowed_root).expanduser().resolve() if allowed_root else None
    candidate = raw if raw.is_absolute() or root is None else root / raw
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DocumentParseError(f"source does not exist: {path}") from exc

    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise DocumentParseError("source path escapes the configured source root") from exc

    if not resolved.is_file():
        raise DocumentParseError(f"source is not a regular file: {path}")

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    if allowed and resolved.suffix.lower() not in allowed:
        raise DocumentParseError(
            f"unsupported source extension {resolved.suffix!r}; expected {sorted(allowed)}"
        )
    return resolved


def source_name(path: Path, allowed_root: str | Path | None) -> str:
    """Return a stable, non-machine-specific source name for citation metadata."""

    if allowed_root:
        root = Path(allowed_root).expanduser().resolve()
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            # ``safe_path`` normally makes this unreachable.  The fallback avoids
            # leaking an absolute local path if a caller supplies a custom parser.
            return path.name
    return path.name


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into deterministic, bounded character chunks.

    Chunks end at whitespace where possible and overlap by characters.  Character
    limits are preferable for this small service because they are easy to enforce
    before indexing and do not require a tokenizer dependency.
    """

    validate_chunking(chunk_size, chunk_overlap)
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        next_start = max(start + 1, end - chunk_overlap)
        start = next_start
    return chunks


def file_within_limit(path: Path, max_file_bytes: int) -> None:
    """Reject oversized files before handing them to a parser library."""

    size = path.stat().st_size
    if size > max_file_bytes:
        raise DocumentParseError(
            f"source exceeds max_file_bytes ({size} > {max_file_bytes})"
        )
