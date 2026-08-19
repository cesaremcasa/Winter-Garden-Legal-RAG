from __future__ import annotations

import os
import secrets
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config.loader import load_config, resolve_runtime_path, source_path_for_config
from llm.client import ABSTENTION, LLMClient, LLMProvider
from retrieval.errors import IndexCorruptionError
from retrieval.index_manager import IndexBuilder, IndexManager
from utils.logger import generate_request_id, get_logger
from validators.grounding import validate_answer

logger = get_logger(__name__)
app = FastAPI(title="Winter Garden Legal RAG API", version="0.1.0")

# Load config
try:
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    config = {}


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(min_length=1, max_length=1_000)


class QueryResponse(BaseModel):
    """Stable response contract with additive grounding fields."""

    answer: str
    citations: list[dict[str, Any]]
    request_id: str
    latency_ms: float
    grounded: bool = False
    abstained: bool = False
    provider: str = "extractive"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
):
    """Retrieve local evidence and answer only from grounded excerpts."""
    start_time = time.time()
    request_id = generate_request_id(x_request_id)
    
    logger.info(
        "Query received",
        extra={
            "request_id": request_id,
            "query": request.query
        }
    )
    
    try:
        manager = _load_index_manager()
        chunks = manager.retrieve(request.query, top_k=int(config.get("top_k", 5)))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="index is not built; run `python scripts/build_index.py`",
        ) from exc
    except IndexCorruptionError as exc:
        logger.error("index integrity validation failed", extra={"request_id": request_id})
        raise HTTPException(status_code=503, detail="index is corrupt; rebuild it") from exc

    llm = LLMClient(
        provider=config.get("provider", LLMProvider.EXTRACTIVE.value),
        model_name=config.get("llm_model_name", "local-extractive"),
        temperature=float(config.get("temperature", 0.0)),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    generated = llm.generate_with_citations(chunks, request.query)
    context = "\n".join(str(chunk.get("text", "")) for chunk in chunks)
    validation = validate_answer(
        generated["answer"],
        context,
        chunks,
        generated.get("citations", []),
    )
    grounded = bool(validation["is_valid"] and generated.get("citations"))
    if not grounded:
        generated = {
            **generated,
            "answer": ABSTENTION,
            "citations": [],
            "abstained": True,
        }
    
    latency_ms = (time.time() - start_time) * 1000
    
    logger.info(
        "Query completed",
        extra={
            "request_id": request_id,
            "latency_ms": latency_ms
        }
    )
    
    return QueryResponse(
        answer=generated["answer"],
        citations=generated["citations"],
        request_id=request_id,
        latency_ms=latency_ms,
        grounded=grounded,
        abstained=bool(generated.get("abstained", False)),
        provider=str(generated.get("provider", "extractive")),
    )


@app.post("/rebuild-index")
@app.post("/index", include_in_schema=False)
async def rebuild_index(
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
):
    """Rebuild local indexes; requires the server-side admin token."""
    request_id = generate_request_id(x_request_id)
    expected = os.getenv(str(config.get("admin_token_env", "WGLR_ADMIN_TOKEN")))
    if not expected:
        raise HTTPException(status_code=503, detail="index rebuild is not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="valid X-Admin-Token is required")
    logger.info("index rebuild requested", extra={"request_id": request_id})
    try:
        result = _build_index()
    except Exception as exc:
        logger.error("index rebuild failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="index rebuild failed") from exc
    return {
        "status": "success",
        "message": "Index rebuilt",
        "request_id": request_id,
        "documents": result.document_count,
        "chunks": result.chunk_count,
    }


def _load_index_manager() -> IndexManager:
    return IndexManager(
        resolve_runtime_path(
            str(config.get("index_path", "./data/index/")), env_var="WGLR_INDEX_PATH"
        ),
        embedding_model=str(config.get("embedding_model_name", "local-hash-384")),
    )


def _build_index():
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
        return builder.build()
