# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LegacyLens is a RAG (Retrieval-Augmented Generation) system that helps developers understand the BLAS (Basic Linear Algebra Subprograms) Fortran codebase. Users ask natural-language questions and get AI-generated answers backed by retrieved source code.

## Commands

```bash
# Dev server (auto-reload)
uvicorn app.main:app --reload

# Production (used by Procfile)
uvicorn app.main:app --host 0.0.0.0 --port $PORT

# Ingest BLAS source into Pinecone (one-shot, requires blas_src/ directory)
python scripts/ingest.py

# Tests
pytest
```

## Required Environment Variables (.env)

- `OPENAI_API_KEY` — embeddings (text-embedding-3-small)
- `ANTHROPIC_API_KEY` — answer generation (Claude)
- `PINECONE_API_KEY` — vector database
- `PINECONE_INDEX_NAME` — defaults to "legacylens"

## Architecture

The system follows a RAG pipeline: **Ingest → Embed → Store → Retrieve → Generate**.

### Ingestion Pipeline (offline, `scripts/ingest.py`)
1. **Scanner** (`app/ingestion/scanner.py`) — finds Fortran files (.f, .f90, .f95, .for, .fpp) in `blas_src/`
2. **Chunker** (`app/ingestion/chunker.py`) — splits files at SUBROUTINE/FUNCTION boundaries using regex; enriches chunks with BLAS metadata (level 1/2/3, data type from naming convention S/D/C/Z)
3. **Embedder** (`app/embeddings/openai_embed.py`) — generates vectors via OpenAI text-embedding-3-small (1536 dims)
4. **Vector DB** (`app/vectordb/pinecone_client.py`) — upserts vectors + metadata to Pinecone

### Query Pipeline (runtime, `app/api/routes.py` → POST `/api/query`)
1. **Search** (`app/retrieval/search.py`) — embeds query → searches Pinecone → filters by score threshold (0.2) → assembles context string
2. **Generator** (`app/retrieval/generator.py`) — sends context + query to Claude with a BLAS-expert system prompt

### Frontend
Single-page app at `static/index.html` — dark-themed UI with inline CSS/JS, manual Fortran syntax highlighting, expandable source cards.

### API Endpoints
- `GET /` — serves the frontend
- `GET /api/health` — health check
- `POST /api/query` — accepts `{ "query": "..." }`, returns `{ answer, sources, query_time_ms }`

## Development Rules

### TDD (Test-Driven Development) — MANDATORY
- **Write tests FIRST**, before implementing any new feature or modifying existing code
- Run `pytest tests/ -v` to verify tests fail (red), then implement, then verify they pass (green)
- Every new function in `app/` must have corresponding tests in `tests/`
- Run the full test suite before committing — all tests must pass
- Never skip tests to save time. Tests catch silent RAG failures that don't crash but return bad results

### Testing Commands
```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_chunker.py -v

# Run a specific test class
pytest tests/test_chunker.py::TestExtractPurposeBlock -v
```

## Key Design Decisions

- Chunks are scoped to individual subroutines/functions, not fixed-size windows — this preserves logical boundaries in the Fortran code
- Each chunk's text is prefixed with a metadata header (BLAS level, data type, description) to improve semantic search relevance
- Chunk text is stored in Pinecone metadata (truncated to 10KB) so it can be returned without a separate store
- Config uses `pydantic-settings` with `dotenv.load_dotenv(override=True)` called before class definition in `app/config.py`
