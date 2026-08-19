# Winter Garden Legal RAG

Small, local-first legal retrieval service for a reproducible v0.1.0 demo. The
core path parses authorized PDF/HTML sources, builds a persistent BM25 + local
hashed-vector index, fuses results with reciprocal rank fusion (RRF), and
returns extractive answers with verifiable citations. It abstains when the
retrieved evidence is missing or cannot be grounded.

This is retrieval software, not legal advice. The sample source is synthetic
and must not be presented as an enacted ordinance.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run python scripts/build_index.py
uv run uvicorn api.routes:app --host 127.0.0.1 --port 8000
```

Then query the local service:

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"query":"What are the public counter hours for a permit application?"}'
```

The committed fixture is under `data/fixtures/`. To index documents you are
authorized to use, place PDF or HTML files in a local source directory, update
`data_path` in `config/config.yaml`, and run the build command again. Basic
limits are configured for file bytes, PDF pages, extracted characters, chunk
size, and overlap.

## API contract

- `GET /health` keeps the original `{"status":"ok"}` response.
- `POST /query` accepts `{"query":"..."}` and preserves `answer`,
  `citations`, `request_id`, and `latency_ms`. Citations contain `document`,
  `source`, `chunk`, and an excerpt that must occur in the indexed chunk.
  Additive fields report `grounded`, `abstained`, and the provider.
- `POST /rebuild-index` is the canonical rebuild endpoint. The legacy
  `POST /index` alias remains available but is not advertised in OpenAPI.
  Both require `X-Admin-Token`, checked against the environment variable named
  by `admin_token_env` (default `WGLR_ADMIN_TOKEN`). There is no default token;
  the token is never logged or stored in the index.

Example rebuild:

```bash
export WGLR_ADMIN_TOKEN='set-this-only-in-your-server-environment'
curl -s -X POST http://127.0.0.1:8000/rebuild-index \
  -H "X-Admin-Token: $WGLR_ADMIN_TOKEN"
```

The local vector store is a deterministic hashed TF-IDF cosine index. It keeps
the original `FaissRetrieval` module/class boundary without requiring a remote
embedding service or heavyweight model download. The provider abstraction
defaults to the key-free extractive provider; no real OpenAI, Anthropic, or
Ollama provider is configured in v0.1.0.

## Development and checks

```bash
uv sync --group dev
uv run ruff check .
uv run mypy api config llm parsers retrieval scripts validators
uv run pytest -q
uv run pip-audit
uv run detect-secrets scan --all-files
```

CI runs the same lint, type-check, tests, dependency audit, and secret scan.
The index is intentionally rebuilt from the fixture in tests rather than
depending on machine-specific paths.

## Provenance and license

The application code is MIT-licensed. `data/fixtures/sample_ordinance.html` is
original synthetic training text dedicated to the public domain under CC0 1.0;
see [`data/fixtures/PROVENANCE.md`](data/fixtures/PROVENANCE.md). No municipal
PDF or municipal-derived chunk is redistributed. Operators are responsible for
the license and authorization of every source they index.
