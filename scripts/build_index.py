#!/usr/bin/env python3
"""
Build index script for Winter Garden Legal RAG.

This script:
1. Loads configuration
2. Parses documents from data_path
3. Generates chunks
4. Creates FAISS and BM25 indexes
5. Saves indexes to configured paths
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import load_config, resolve_runtime_path, source_path_for_config
from retrieval.index_manager import IndexBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


def build_index(config_path: str | None = None):
    """Build both local retrieval indexes from configured PDF/HTML sources."""
    try:
        config = load_config(config_path or "config/config.yaml")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
    index_path = resolve_runtime_path(
        str(config.get("index_path", "./data/index/")), env_var="WGLR_INDEX_PATH"
    )
    with source_path_for_config(str(config.get("data_path", "./data/fixtures/"))) as source_path:
        builder = IndexBuilder(
            source_path,
            index_path,
            chunk_size=int(config.get("chunk_size", 800)),
            chunk_overlap=int(config.get("chunk_overlap", 120)),
            max_file_bytes=int(config.get("max_file_bytes", 5_000_000)),
            max_pages=int(config.get("max_pages", 100)),
            max_text_chars=int(config.get("max_text_chars", 1_000_000)),
            embedding_model=str(config.get("embedding_model_name", "local-hash-384")),
        )
        result = builder.build()
    logger.info(
        "index build completed",
        extra={"documents": result.document_count, "chunks": result.chunk_count},
    )
    return result


def main() -> None:
    """CLI entrypoint."""

    try:
        result = build_index()
    except Exception as exc:
        logger.error("index build failed")
        raise SystemExit(1) from exc
    print(
        f"Built {result.chunk_count} chunks from {result.document_count} documents "
        f"at {result.index_path}"
    )


if __name__ == "__main__":
    main()
