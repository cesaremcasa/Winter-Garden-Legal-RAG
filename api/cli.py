"""Command-line entrypoint for serving the local API."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    """Run Uvicorn without changing files in the installed package."""

    parser = argparse.ArgumentParser(description="Serve the Winter Garden Legal RAG API")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run("api.routes:app", host=args.host, port=args.port)
