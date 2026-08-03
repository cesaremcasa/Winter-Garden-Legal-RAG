# Winter Garden Legal RAG Backend (MVP Architecture)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)
![Status](https://img.shields.io/badge/Status-MVP-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A modular, architecture-first backend for a **Legal Retrieval-Augmented Generation (RAG)** system built on the *City of Winter Garden, Florida* legal code.

This MVP focuses on **correct engineering structure**, **clean subsystem boundaries**, and **production-style organization**, serving as a blueprint for how a real legal RAG backend is built.  
It is intentionally implementation-light so the architecture can be evaluated safely and clearly.

---

## 1. Overview

The system defines all essential layers required in a modern RAG backend:

- FastAPI service (`/query`, `/index`, `/health`)
- Modular retrieval subsystem (BM25, FAISS, Hybrid)
- Parsing layer for PDFs and HTML
- Configurable LLM client wrapper
- Grounding & citation validation stubs
- Structured JSON logging with request tracing
- Centralized YAML configuration
- Operational scripts for indexing & data processing
- API contract tests (using FastAPI TestClient)

This project demonstrates **engineering discipline**, not raw capability:
clean separation of concerns, explicit interfaces, reproducible scripts, and an extendable design.

---

## 2. Current Capabilities (MVP Scope)

### API
- `/health` returns service status  
- `/query` accepts natural language queries, returns structured placeholder answers  
- `/index` is scaffolding for index rebuild workflows  
- Automatic request ID propagation  
- Per-request latency tracking  

### Logging
- Shared JSON structured logger  
- Fields: `timestamp`, `level`, `message`, `request_id`, `latency`, `query`  

### Configuration
- Central `config/config.yaml`  
- Defines paths, models, retrieval parameters, and API settings  
- Error-safe loading (`loader.py`)  

### Scripts
- `scripts/build_index.py` orchestrates ingestion + future embedding/index building  
- `scripts/reprocess_data.py` resets processed directory  

### Tests
- API contract tests ensuring:
  - Endpoint stability  
  - Header propagation  
  - Response structure  

---

## 3. Architectural Intent

This MVP establishes the **blueprint** for a full legal RAG pipeline.  
All subsystems exist in their production form. Only the logic is missing, by design.

### Retrieval Layer (`retrieval/`)
- `bm25.py`, `faiss_store.py`, `hybrid.py`  
- Interfaces defined  
- Methods stubbed for future:
  - Sparse BM25 retrieval  
  - Dense FAISS vector search  
  - Hybrid ranking/fusion strategies  

### Parsing Layer (`parsers/`)
- `pdf_parser.py`, `html_parser.py`  
- Directory traversal + signatures defined  
- Extraction & chunking are TODO  

### LLM Layer (`llm/client.py`)
- Provider enum (OpenAI, Anthropic, local models)  
- Unified LLM interface  
- Stubbed answer + citation generation  

### Validation Layer (`validators/grounding.py`)
- Grounding interface  
- Citation check hook  

Ready for RAG hallucination prevention once retrieval + LLM integration exist.

---

## 4. Project Structure

```
├── api/                  # FastAPI routes and request models
├── config/               # YAML config + loader
├── data/                 # Raw PDFs, processed chunks, index placeholders
├── llm/                  # LLM client scaffolding
├── parsers/              # PDF / HTML parsing skeletons
├── retrieval/            # BM25, FAISS, Hybrid retrieval scaffolding
├── scripts/              # Index build + data processing orchestration
├── tests/                # API contract tests
├── utils/                # JSON structured logging
├── validators/           # Grounding / citation validation stubs
└── run_api.sh            # Startup script
```

This mirrors real production RAG services where each subsystem evolves independently.

---

## 5. Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

---

## 6. Running the API

### Using the startup script
```bash
./run_api.sh 8000
```

### Using Uvicorn directly
```bash
uvicorn api.routes:app --reload --host 0.0.0.0 --port 8000
```

---

## 7. Reprocessing and Index Building

### Reset processed data
```bash
python scripts/reprocess_data.py
```

### Initialize index structure
```bash
python scripts/build_index.py
```

Both scripts validate configuration & logging pipelines and set up the environment for future
embedding + retrieval logic.

---

## 8. Current Status (MVP Reality)

This repository is in **Phase 1, Architectural Scaffolding**.

Functional components intentionally remain unimplemented:

| Subsystem        | Current Behavior |
|------------------|------------------|
| Retrieval        | Returns empty lists |
| Parsers          | No PDF/HTML extraction |
| LLM generation   | Returns placeholder strings |
| Grounding        | Always passes |
| Index building   | No embeddings or FAISS index |

This design ensures the project is safe for public portfolio use while highlighting real engineering practices.

---

## 9. Roadmap (Production Path)

This MVP is structured so each subsystem can be expanded independently.  
Below is the planned evolution path for turning this architecture into a fully functional legal RAG backend.

### Document Ingestion
- Implement PDF text extraction (pdfplumber / PyPDF2)
- Add HTML parsing (BeautifulSoup)
- Introduce text chunking with size + overlap control

### Retrieval Layer
- Generate embeddings using sentence-transformers
- Build FAISS index for dense retrieval
- Implement BM25 for sparse retrieval
- Add hybrid fusion (RRF, weighted / score-based ranking)

### LLM Integration
- Connect LLMClient to OpenAI / Anthropic / local models
- Create RAG-oriented prompt templates
- Produce answers with citation mapping

### Validation Layer
- Implement citation grounding checks
- Add confidence scoring
- Introduce hallucination detection

### Observability + Testing
- Extend API + retrieval telemetry
- Add metrics (latency, index hit ratios)
- Implement integration + stress tests
- Expand automated QA coverage

---

## 10. Screenshots (MVP Runtime Preview)

These screenshots illustrate the MVP running locally with a functioning FastAPI service.

### 10.1 API Server Running
![API server running](images/server_running.png)

### 10.2 Health Endpoint Response (cURL)
![Health check response](images/01_health_check_curl.png)

### 10.3 Directory Structure (Project Tree)
![Project directory tree](images/02_project_tree_structure.png)

### 10.4 Uvicorn Server Running (Startup Logs)
![Uvicorn server running](images/03_uvicorn_server_running.png)

These visuals confirm the operational backbone of the system  
(routing, logging, environment setup, and server initialization).

---

## 11. License

MIT License 

Copyright (c) 2025 Cesar Augusto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Cesar Augusto
AI Systems Engineer, Mycellium Lab
