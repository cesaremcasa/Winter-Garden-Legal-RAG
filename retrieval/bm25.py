"""A small, persisted BM25 implementation with no service dependency."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from utils.logger import get_logger

from .errors import IndexCorruptionError
from .tokenize import tokenize

logger = get_logger(__name__)

SCHEMA_VERSION = 1


def _index_file(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.suffix == ".json" else candidate / "index.json"


def _normalise_document(document: str | dict[str, Any], number: int) -> dict[str, Any]:
    text: Any
    if isinstance(document, str):
        text = document
        metadata: dict[str, Any] = {}
        identifier = str(number)
    elif isinstance(document, dict):
        text = document.get("text")
        metadata = document.get("metadata") or {}
        identifier = str(document.get("id") or metadata.get("chunk") or number)
    else:
        raise ValueError("documents must be strings or mappings")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("each indexed document must contain non-empty text")
    if not isinstance(metadata, dict):
        raise ValueError("document metadata must be a mapping")
    return {"id": identifier, "text": text, "metadata": metadata}


class BM25Retrieval:
    """BM25Okapi-style sparse retrieval persisted as JSON."""

    def __init__(self, index_path: str | Path):
        self.index_path = _index_file(index_path)
        self.documents: list[dict[str, Any]] = []
        self._token_counts: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        if self.index_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("index payload must be an object")
            if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != "bm25":
                raise ValueError("unsupported BM25 index schema")
            raw_documents = payload["documents"]
            if not isinstance(raw_documents, list):
                raise ValueError("documents must be a list")
            self.documents = [
                _normalise_document(document, i) for i, document in enumerate(raw_documents)
            ]
            identifiers = [document["id"] for document in self.documents]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("duplicate document ids")
            self._token_counts = [
                Counter(tokenize(document["text"])) for document in self.documents
            ]
            self._avgdl = float(payload["avg_doc_length"])
            raw_idf = payload["idf"]
            if not isinstance(raw_idf, dict):
                raise ValueError("idf must be a mapping")
            self._idf = {str(token): float(value) for token, value in raw_idf.items()}
            if self.documents and self._avgdl <= 0:
                raise ValueError("avg_doc_length must be positive for a non-empty index")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, IndexCorruptionError):
                raise
            raise IndexCorruptionError(f"invalid BM25 index: {self.index_path}") from exc

    @property
    def count(self) -> int:
        return len(self.documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return non-zero BM25 matches in deterministic score/id order."""

        if top_k < 1:
            return []
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []
        query_terms = set(query_tokens)
        scored: list[tuple[float, str, int]] = []
        k1 = 1.5
        b = 0.75
        for number, counts in enumerate(self._token_counts):
            document_length = sum(counts.values())
            score = 0.0
            matched_terms = 0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                matched_terms += 1
                idf = self._idf.get(term, 0.0)
                denominator = frequency + k1 * (1 - b + b * document_length / self._avgdl)
                score += idf * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((score, self.documents[number]["id"], matched_terms))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, identifier, matched_terms) in enumerate(scored[:top_k], start=1):
            number = next(
                i for i, document in enumerate(self.documents) if document["id"] == identifier
            )
            result = dict(self.documents[number])
            result.update(
                {
                    "score": float(score),
                    "rank": rank,
                    "retriever": "bm25",
                    "matched_terms": matched_terms,
                }
            )
            results.append(result)
        return results

    def build_index(
        self,
        documents: Sequence[str | dict[str, Any]],
        save_path: str | Path,
    ) -> Path:
        """Build and save a deterministic BM25 JSON index."""

        records = [_normalise_document(document, i) for i, document in enumerate(documents)]
        if not records:
            raise ValueError("cannot build a BM25 index without documents")
        token_counts = [Counter(tokenize(document["text"])) for document in records]
        document_frequency: Counter[str] = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())
        count = len(records)
        idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in sorted(document_frequency.items())
        }
        average_length = sum(sum(counts.values()) for counts in token_counts) / count
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "bm25",
            "documents": records,
            "avg_doc_length": average_length,
            "idf": idf,
        }
        destination = _index_file(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self.index_path = destination
        self._load()
        return destination
