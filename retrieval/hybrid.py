from pathlib import Path
from typing import Any

from utils.logger import get_logger

from .bm25 import BM25Retrieval
from .faiss_store import FaissRetrieval

logger = get_logger(__name__)


class HybridRetrieval:
    """Hybrid BM25 + local-vector retrieval using deterministic RRF fusion."""
    
    def __init__(
        self,
        bm25_index_path: str | Path,
        faiss_index_path: str | Path,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25_index_path: Path to BM25 index
            faiss_index_path: Path to FAISS index
            bm25_weight: Weight for BM25 scores
            faiss_weight: Weight for FAISS scores
        """
        self.bm25 = BM25Retrieval(bm25_index_path)
        self.faiss = FaissRetrieval(faiss_index_path)
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        logger.info("Hybrid retrieval initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top results after reciprocal-rank fusion.

        RRF depends only on rank, not floating point score calibration.  Every tie
        is resolved by the stable chunk id so repeated rebuilds and queries have
        reproducible ordering.
        """

        if top_k < 1:
            return []
        candidate_k = max(top_k * 2, top_k)
        bm25_results = self.bm25.retrieve(query, candidate_k)
        vector_results = self.faiss.retrieve(query, candidate_k)
        merged: dict[str, dict[str, Any]] = {}
        rrf_k = 60.0
        for weight, results, name in (
            (self.bm25_weight, bm25_results, "bm25"),
            (self.faiss_weight, vector_results, "vector"),
        ):
            for result in results:
                identifier = str(result["id"])
                entry = merged.setdefault(
                    identifier,
                    {
                        "id": identifier,
                        "text": result["text"],
                        "metadata": result.get("metadata", {}),
                        "rrf_score": 0.0,
                        "sources": [],
                    },
                )
                rank = int(result["rank"])
                entry["rrf_score"] += float(weight) / (rrf_k + rank)
                entry["sources"].append(name)
                entry[f"{name}_score"] = float(result["score"])
                entry[f"{name}_rank"] = rank

        ordered = sorted(
            merged.values(), key=lambda result: (-result["rrf_score"], result["id"])
        )
        for rank, result in enumerate(ordered[:top_k], start=1):
            result["score"] = result["rrf_score"]
            result["rank"] = rank
            result["retriever"] = "hybrid"
            result["sources"] = sorted(set(result["sources"]))
        return ordered[:top_k]
