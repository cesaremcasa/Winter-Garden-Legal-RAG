#!/usr/bin/env python3
"""
Reprocess data script for Winter Garden Legal RAG.

This script:
1. Deletes old processed outputs inside the project
2. Re-runs the bounded PDF and HTML parsers
3. Regenerates a deterministic JSONL chunk file
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import load_config
from parsers.html_parser import HTMLParser
from parsers.pdf_parser import PDFParser
from utils.logger import get_logger

logger = get_logger(__name__)


def clean_outputs(processed_path: str):
    """
    Delete old processed data.
    
    Args:
        processed_path: Path to processed data directory
    """
    logger.info(f"Cleaning old outputs from: {processed_path}")
    
    path = Path(processed_path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("processed_path must remain inside the project") from exc
    if path.exists():
        shutil.rmtree(path)
        logger.info("Old outputs deleted")
    
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory recreated")


def reprocess_sources(data_path: str, output_path: str, config: dict[str, Any]):
    """
    Reprocess PDF documents.
    
    Args:
        data_path: Path to raw PDFs
        output_path: Path to save processed data
        config: Configuration dict
    """
    logger.info(f"Reprocessing PDF and HTML sources from: {data_path}")
    
    parser = PDFParser(
        chunk_size=config.get("chunk_size", 500),
        chunk_overlap=config.get("chunk_overlap", 120),
        allowed_root=data_path,
        max_file_bytes=config.get("max_file_bytes", 5_000_000),
        max_pages=config.get("max_pages", 100),
        max_text_chars=config.get("max_text_chars", 1_000_000),
    )
    
    html_parser = HTMLParser(
        chunk_size=config.get("chunk_size", 800),
        chunk_overlap=config.get("chunk_overlap", 120),
        allowed_root=data_path,
        max_file_bytes=config.get("max_file_bytes", 5_000_000),
        max_text_chars=config.get("max_text_chars", 1_000_000),
    )
    chunks = parser.parse_directory(data_path) + html_parser.parse_directory(data_path)
    chunks.sort(key=lambda chunk: str(chunk["id"]))
    output = Path(output_path).resolve() / "chunks.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("data reprocessing completed", extra={"chunks": len(chunks)})


# Compatibility name used by the original script.
reprocess_pdfs = reprocess_sources


def main():
    """Main reprocessing function."""
    logger.info("Starting data reprocessing")
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(config.get("data_path", "./data/fixtures/"))
    processed_path = Path(config.get("processed_path", "./data/processed/"))
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not processed_path.is_absolute():
        processed_path = project_root / processed_path
    
    # Clean old outputs
    clean_outputs(str(processed_path))
    
    # Reprocess documents
    reprocess_sources(str(data_path), str(processed_path), config)
    
    logger.info("Data reprocessing completed successfully")


if __name__ == "__main__":
    main()
