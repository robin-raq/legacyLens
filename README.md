# LegacyLens

A RAG (Retrieval-Augmented Generation) system that makes the BLAS (Basic Linear Algebra Subprograms) Fortran codebase queryable through natural language. Ask questions about subroutines, get AI-generated answers backed by retrieved source code with file/line references.

**Live demo:** [https://legacylens-production-fd39.up.railway.app](https://legacylens-production-fd39.up.railway.app)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python / FastAPI |
| Frontend | Vanilla HTML/CSS/JS (single page) |
| LLM | Google Gemini 2.5 Flash (streaming) |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Vector DB | Pinecone (managed, free tier) |
| Deployment | Railway |

## Features

- **8 code understanding features:** Code Explanation, Documentation Generation, Pattern Detection, Business Logic Extraction, Dependency Mapping, Impact Analysis, Translation Hints, Bug Pattern Search
- **Intelligent retrieval:** Query classifier routes to feature-specific prompts and search parameters
- **Re-ranking:** Score-gap filtering + exact-match boosting for high precision (93% P@5)
- **SSE streaming:** Real-time answer generation with token-by-token display
- **Session memory:** Lightweight conversation tracking across queries
- **Fortran syntax highlighting:** Manual highlighting in the web UI
- **Error handling:** Retry with backoff on rate limits, graceful degradation on service failures

## Architecture

```
Ingestion (one-time):
  Fortran files -> Syntax-aware chunker -> OpenAI embeddings -> Pinecone

Query (per request):
  User question -> Query classifier (8 types)
                -> OpenAI embedding
                -> Pinecone search (top-k)
                -> Score-gap filter + exact-match boost
                -> Context assembly
                -> Gemini 2.5 Flash (streaming)
                -> SSE to frontend
```

## Quick Start

### Prerequisites

- Python 3.11+
- API keys for: OpenAI, Google AI (Gemini), Pinecone

### Installation

```bash
git clone https://github.com/yourusername/legacyLens.git
cd legacyLens

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Embeddings (text-embedding-3-small) |
| `PINECONE_API_KEY` | Yes | Vector database |
| `GOOGLE_API_KEY` | Yes* | Gemini 2.5 Flash for answer generation |
| `ANTHROPIC_API_KEY` | No* | Claude (if `llm_provider=anthropic`) |
| `PINECONE_INDEX_NAME` | No | Default: `legacylens` |
| `PORT` | No | Default: `3000` |

\* At least one of `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY` is required. Default provider is Gemini.

### Ingest the BLAS Codebase

```bash
# Clone the reference BLAS source
git clone https://github.com/Reference-LAPACK/lapack.git
cp -r lapack/BLAS/SRC blas_src/SRC

# Run ingestion (scans, chunks, embeds, upserts to Pinecone)
python scripts/ingest.py
```

### Run the Server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

- `GET /` -- Serves the frontend
- `GET /api/health` -- Health check
- `POST /api/query` -- `{ "query": "..." }` returns `{ answer, sources, query_type, query_time_ms, session_id }`
- `GET /api/source?file=...` -- Returns full Fortran source file content (with path traversal protection)

Supports `?stream=true` for SSE streaming and `?session_id=...` for conversation memory.

## Testing

```bash
# Run all tests (215 tests)
pytest tests/ -v

# Run evaluation (24 queries across 8 features, requires running server)
python scripts/eval.py
python scripts/eval.py --url https://legacylens-production-fd39.up.railway.app

# Verify answers contain valid file/line references
python scripts/verify_refs.py
python scripts/verify_refs.py --url https://legacylens-production-fd39.up.railway.app
python scripts/verify_refs.py --skip-disk  # no blas_src/ needed
```

## Evaluation Results

24 queries across 8 features, tested with Gemini 2.5 Flash:

| Metric | Result |
|--------|--------|
| Retrieval precision (P@5) | 93% |
| Term recall | 96% |
| Queries passed | 24/24 |
| Mean latency (streaming) | 6.3s total (first token <1s) |
| Ingestion throughput | 48,480 LOC in ~2 min (264K LOC/s) |
| Answer accuracy | 100% correct file refs, 0 hallucinations |
| Codebase coverage | 100% (169 files, 48,480 LOC) |
| Test suite | 215 tests, all passing |

## Project Structure

```
legacylens/
  app/
    main.py              # FastAPI entry point
    config.py            # Centralized settings (pydantic-settings)
    models.py            # Request/response schemas with validation
    session.py           # Lightweight conversation memory
    utils.py             # Shared retry utility
    ingestion/
      scanner.py         # Find .f files recursively
      chunker.py         # Syntax-aware subroutine splitting + metadata
    embeddings/
      openai_embed.py    # OpenAI embedding client with retry
    vectordb/
      pinecone_client.py # Pinecone upsert + search
    retrieval/
      query_classifier.py # 8-type regex classifier
      search.py          # Similarity search + re-ranking
      generator.py       # Gemini/Claude generation with streaming
    api/
      routes.py          # API routes + SSE streaming
  static/
    index.html           # Single-page chat UI
  scripts/
    ingest.py            # CLI ingestion script with LOC metrics
    eval.py              # RAG evaluation (24 queries)
    verify_refs.py       # Answer file/line reference verification
  tests/                 # 215 tests
```

## Documentation

- [RAG Architecture](RAG_ARCHITECTURE.md) -- Vector DB selection, chunking strategy, retrieval pipeline, failure modes, performance results
- [AI Cost Analysis](AI_COST_ANALYSIS.md) -- Development spend and production cost projections
- [CLAUDE.md](CLAUDE.md) -- Development rules and detailed architecture reference
