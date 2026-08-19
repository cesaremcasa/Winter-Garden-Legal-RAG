"""Deterministic tokenization shared by sparse and local dense retrieval."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "you",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text without locale-dependent or model-dependent behavior."""

    return [
        token
        for match in TOKEN_RE.finditer(text)
        if (token := match.group(0).lower()) not in STOPWORDS
    ]
