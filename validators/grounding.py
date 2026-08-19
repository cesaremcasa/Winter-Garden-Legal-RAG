from __future__ import annotations

import re
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


ABSTENTION_MARKER = "I don't have enough evidence"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def validate_answer(
    answer: str,
    context: str,
    chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check answer word coverage and citation references against retrieved text."""

    if ABSTENTION_MARKER in answer:
        return {"is_valid": not chunks, "confidence": 0.0, "issues": []}
    answer_words = _words(answer)
    context_words = _words(context)
    coverage = len(answer_words & context_words) / max(len(answer_words), 1)
    valid_citations = validate_citations(citations or [], chunks) if citations is not None else True
    issues: list[str] = []
    if coverage < 0.75:
        issues.append("answer contains terms absent from retrieved context")
    if citations is not None and not valid_citations:
        issues.append("one or more citations do not match retrieved chunks")
    return {
        "is_valid": coverage >= 0.75 and valid_citations,
        "confidence": round(coverage, 4),
        "issues": issues,
    }


def validate_citations(
    citations: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> bool:
    """
    Validate that all citations reference actual chunks.
    
    Args:
        citations: List of citations
        chunks: Retrieved chunks
    
    Returns:
        True if all citations are valid
    
    """
    logger.info("validating citations", extra={"citation_count": len(citations)})
    by_id = {str(chunk.get("id")): chunk for chunk in chunks}
    for citation in citations:
        if not isinstance(citation, dict):
            return False
        identifier = citation.get("chunk")
        chunk = by_id.get(str(identifier))
        if chunk is None:
            return False
        metadata = chunk.get("metadata") or {}
        if citation.get("document") != metadata.get("document"):
            return False
        if citation.get("source") != metadata.get("source"):
            return False
        excerpt = citation.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt or excerpt not in str(chunk.get("text", "")):
            return False
    return True
