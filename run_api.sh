#!/bin/bash

# Winter Garden Legal RAG - API Startup Script
# Usage: ./run_api.sh [port]

set -e

# Get port from argument or environment variable or default to 8000
PORT=${1:-${PORT:-8000}}

echo "Starting Winter Garden Legal RAG API on port $PORT..."

# Check if config exists
if [ ! -f "config/config.yaml" ]; then
    echo "Warning: config/config.yaml not found"
fi

# Start API (uv keeps the lockfile and environment reproducible).
if command -v uv >/dev/null 2>&1; then
    exec uv run uvicorn api.routes:app --host "${HOST:-127.0.0.1}" --port "$PORT"
fi
exec uvicorn api.routes:app --host "${HOST:-127.0.0.1}" --port "$PORT"
