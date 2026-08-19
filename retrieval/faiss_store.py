"""Deterministic local vector retrieval.

The original scaffold called this module ``faiss_store``.  v0.1.0 keeps that
public class name but uses a dependency-light hashed TF-IDF cosine index.  It is
an equivalent local vector store for the small fixture and can be replaced by
FAISS later without changing the API or persisted document metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from utils.logger import get_logger

from .bm25 import _normalise_document
from .errors import IndexCorruptionError
from .tokenize import tokenize

logger = get_logger(__name__)

SCHEMA_VERSION = 1
DEFAULT_DIMENSION = 384


def _index_file(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.suffix == ".json" else candidate / "index.json"


def _bucket(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dimension


class FaissRetrieval:
    """Persisted local hashed TF-IDF cosine retrieval (FAISS-compatible role)."""

    def __init__(self, index_path: str | Path, embedding_model: str = "local-hash-384"):
        self.index_path = _index_file(index_path)
        self.embedding_model = embedding_model
        self.documents: list[dict[str, Any]] = []
        self.dimension = DEFAULT_DIMENSION
        self.idf: dict[str, float] = {}
        self.vectors: list[list[float]] = []
        if self.index_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"vector index not found: {self.index_path}")
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("index payload must be an object")
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("kind") != "local-vector"
            ):
                raise ValueError("unsupported vector index schema")
            self.dimension = int(payload["dimension"])
            if self.dimension < 1 or self.dimension > 16_384:
                raise ValueError("invalid vector dimension")
            self.documents = [
                _normalise_document(document, i)
                for i, document in enumerate(payload["documents"])
            ]
            identifiers = [document["id"] for document in self.documents]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("duplicate document ids")
            raw_idf = payload["idf"]
            if not isinstance(raw_idf, dict):
                raise ValueError("idf must be a mapping")
            self.idf = {str(token): float(value) for token, value in raw_idf.items()}
            vectors = payload["vectors"]
            if not isinstance(vectors, list) or len(vectors) != len(self.documents):
                raise ValueError("vector matrix shape does not match documents")
            for vector in vectors:
                if (
                    not isinstance(vector, list)
                    or len(vector) != self.dimension
                    or not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in vector
                    )
                ):
                    raise ValueError("vector matrix shape or values are invalid")
            self.vectors = [[float(value) for value in vector] for vector in vectors]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IndexCorruptionError(f"invalid vector index: {self.index_path}") from exc

    @property
    def count(self) -> int:
        return len(self.documents)

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(tokenize(text))
        if not counts:
            return vector
        for token, frequency in counts.items():
            vector[_bucket(token, self.dimension)] += float(frequency) * self.idf.get(token, 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return cosine matches above a small evidence floor."""

        if top_k < 1 or not self.documents:
            return []
        query_vector = self._vectorize(query)
        if not any(query_vector):
            return []

        def cosine(vector: list[float]) -> float:
            return sum(value * query for value, query in zip(vector, query_vector, strict=True))

        ranked = sorted(
            (
                (cosine(vector), document["id"], number)
                for number, (vector, document) in enumerate(
                    zip(self.vectors, self.documents, strict=True)
                )
                if cosine(vector) >= 0.05
            ),
            key=lambda item: (-item[0], item[1]),
        )
        results: list[dict[str, Any]] = []
        for rank, (score, _identifier, number) in enumerate(ranked[:top_k], start=1):
            result = dict(self.documents[number])
            result.update({"score": score, "rank": rank, "retriever": "vector"})
            results.append(result)
        return results

    def build_from_documents(
        self,
        documents: Sequence[str | dict[str, Any]],
        save_path: str | Path,
        *,
        dimension: int = DEFAULT_DIMENSION,
    ) -> Path:
        """Build a hashed TF-IDF index directly from document records."""

        if dimension < 1 or dimension > 16_384:
            raise ValueError("dimension must be between 1 and 16384")
        records = [_normalise_document(document, i) for i, document in enumerate(documents)]
        if not records:
            raise ValueError("cannot build a vector index without documents")
        document_frequency: Counter[str] = Counter()
        token_counts: list[Counter[str]] = []
        for record in records:
            counts = Counter(tokenize(record["text"]))
            token_counts.append(counts)
            document_frequency.update(counts.keys())
        count = len(records)
        idf = {
            token: math.log(1 + (count + 1) / (frequency + 1))
            for token, frequency in sorted(document_frequency.items())
        }
        vectors = [[0.0] * dimension for _ in range(count)]
        for row, counts in enumerate(token_counts):
            for token, frequency in counts.items():
                vectors[row][_bucket(token, dimension)] += frequency * idf.get(token, 1.0)
            norm = math.sqrt(sum(value * value for value in vectors[row]))
            if norm:
                vectors[row] = [value / norm for value in vectors[row]]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "local-vector",
            "embedding_model": self.embedding_model,
            "dimension": dimension,
            "documents": records,
            "idf": idf,
            "vectors": vectors,
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

    def build_index(
        self,
        embeddings: Any,
        documents: list[dict[str, Any]],
        save_path: str | Path,
    ) -> Path:
        """Compatibility builder for callers of the original FAISS scaffold.

        Passing ``None`` builds the local hashed TF-IDF representation.  An
        explicit 2-D embedding matrix is also accepted for migration scripts;
        query vectors remain hash-based in that mode, so normal application builds
        should use :meth:`build_from_documents`.
        """

        if embeddings is None:
            return self.build_from_documents(documents, save_path)
        records = [_normalise_document(document, i) for i, document in enumerate(documents)]
        matrix = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        if (
            not isinstance(matrix, list)
            or len(matrix) != len(records)
            or not matrix
            or not isinstance(matrix[0], list)
        ):
            raise ValueError("embeddings must be a 2-D matrix matching documents")
        self.dimension = len(matrix[0])
        if self.dimension < 1 or any(len(row) != self.dimension for row in matrix):
            raise ValueError("embeddings must be a rectangular 2-D matrix")
        normalised: list[list[float]] = []
        for row in matrix:
            values = [float(value) for value in row]
            norm = math.sqrt(sum(value * value for value in values))
            normalised.append([value / norm for value in values] if norm else values)
        self.idf = {}
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "local-vector",
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "documents": records,
            "idf": self.idf,
            "vectors": normalised,
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
