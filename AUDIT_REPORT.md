# LegacyLens Codebase Audit Report

**Date:** March 6, 2025  
**Scope:** Full codebase analysis across architecture, code quality, security, performance, testing, documentation, DevOps, and improvements.

---

## 1. Architecture

### 1.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LEGACYLENS RAG PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  INGESTION (offline)          │  QUERY (runtime)                             │
│  scripts/ingest.py            │  app/api/routes.py → POST /api/query         │
│  ┌──────────┐                 │  ┌──────────────┐  ┌─────────────┐           │
│  │ Scanner  │ → scan_fortran  │  │ Classifier   │  │ QueryRequest│           │
│  └────┬─────┘                 │  └──────┬───────┘  └──────┬──────┘           │
│       │                        │         │                 │                  │
│  ┌────▼─────┐                 │  ┌──────▼───────┐  ┌──────▼──────┐           │
│  │ Chunker  │ → chunk_fortran │  │ Search       │  │ embed_query │           │
│  └────┬─────┘                 │  │ (Pinecone)    │  │ + rerank    │           │
│       │                        │  └──────┬───────┘  └──────┬──────┘           │
│  ┌────▼─────┐                 │         │                 │                  │
│  │ Embedder │ → embed_texts    │  ┌──────▼───────┐  ┌──────▼──────┐           │
│  └────┬─────┘                 │  │ Generator    │  │ build_context│           │
│       │                        │  │ (Claude/Gemini)└──────┬──────┘           │
│  ┌────▼─────┐                 │  └──────┬───────┘         │                  │
│  │ Pinecone │ → upsert_chunks  │         │                 │                  │
│  └──────────┘                 │  ┌──────▼─────────────────▼──────┐           │
│                               │  │ Session Store (conversation)   │           │
│                               │  └────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Map

| Layer | Files | Responsibility |
|-------|-------|----------------|
| **Entry** | `app/main.py`, `app/api/routes.py` | FastAPI app, routing, SSE streaming |
| **Models** | `app/models.py` | Pydantic schemas (QueryRequest, QueryResponse, CodeChunk, etc.) |
| **Ingestion** | `app/ingestion/scanner.py`, `chunker.py` | Scan Fortran files, chunk at subroutine boundaries |
| **Embeddings** | `app/embeddings/openai_embed.py` | OpenAI text-embedding-3-small, query caching |
| **Vector DB** | `app/vectordb/pinecone_client.py` | Pinecone upsert/search |
| **Retrieval** | `app/retrieval/search.py`, `query_classifier.py` | Classify intent, search, rerank, build context |
| **Generation** | `app/retrieval/generator.py` | Claude/Gemini answer generation (sync + stream) |
| **Session** | `app/session.py` | In-memory conversation memory with TTL |
| **Cache** | `app/cache.py` | LRU cache for query embeddings |
| **Config** | `app/config.py` | Pydantic Settings, env vars |
| **Frontend** | `static/index.html` | Single-page app, SSE client, Fortran syntax highlighting |

### 1.3 Coupling & Separation of Concerns

**Strengths:**
- Clear pipeline stages: classify → search → generate
- Models (`app/models.py`) are shared and well-typed
- Query classifier is isolated with feature-specific prompts and `top_k` params
- Ingestion is fully decoupled from runtime (separate script)

**Coupling Issues:**
- **Generator ↔ Classifier**: `generator.py` imports `get_search_params` from `query_classifier.py` to build prompts. This couples generation to classification logic.
- **Search ↔ Config**: `search.py` reads `settings` at module load for `EXACT_MATCH_BOOST`, `SCORE_GAP_RATIO` — not injectable.
- **Routes ↔ All**: `routes.py` directly imports and calls `search_codebase`, `generate_answer`, `classify_query`, `session_store`. No service layer abstraction.

### 1.4 Architectural Anti-Patterns

| Issue | Location | Description |
|-------|----------|-------------|
| **God route** | `app/api/routes.py:44-154` | `/api/query` handles session resolution, classification, search, generation, streaming, and error handling in one 110-line function |
| **Global singletons** | `app/embeddings/openai_embed.py`, `app/vectordb/pinecone_client.py`, `app/session.py`, `app/cache.py` | Module-level `_client`, `_index`, `store`, `_cache` — hard to test, no DI |
| **Blocking in async** | `app/api/routes.py:60-61`, `76-77`, `129-131` | `asyncio.to_thread()` and `run_in_executor()` used correctly, but `search_codebase` and `generate_answer` are sync — could be async-native |
| **Tight regex ordering** | `app/retrieval/query_classifier.py:34-66` | `_PATTERNS` order determines classification; adding a new type can break existing matches |

---

## 2. Code Quality

### 2.1 Structure & Naming

**Strengths:**
- Consistent module layout: `app/<domain>/<module>.py`
- Clear function names: `scan_fortran_files`, `chunk_fortran_file`, `embed_query`, `search_codebase`, `build_context`
- Pydantic models use PascalCase; functions use snake_case

**Issues:**
- `app/utils.py` has only `retry_on_rate_limit` — could live in a `retries.py` or alongside the modules that use it
- `format_sse_event` in `routes.py` is a pure helper; could move to `app/utils.py` or `app/api/sse.py`

### 2.2 Consistency

- **Type hints**: Most functions have type hints; `chunker.py` and `scanner.py` are well-typed. `routes.py` uses `dict | str` for `format_sse_event` — good.
- **Docstrings**: Present in `chunker.py`, `search.py`, `query_classifier.py`, `generator.py`. Missing in `scanner.py`, `cache.py`, `session.py`, `routes.py` (except `format_sse_event`).
- **Logging**: Used in `routes.py`, `generator.py`, `openai_embed.py`, `pinecone_client.py`. Inconsistent levels (mostly `logger.error`, some `logger.warning`).

### 2.3 Duplication

| Duplication | Locations | Suggestion |
|-------------|------------|-------------|
| BLAS routine lists | `chunker.py:36-43`, `search.py:14-27` | Extract to `app/constants.py` or shared module |
| Retry logic for streaming | `generator.py:164-183`, `187-212` | Both `_stream_anthropic` and `_stream_gemini` duplicate retry loop; could use `retry_on_rate_limit` pattern |
| Error message | `generator.py:117` | `_ERROR_MSG` is good; ensure all paths use it |

### 2.4 Error Handling

**Strengths:**
- Pydantic validation on `QueryRequest` (min/max length, strip, empty check)
- Pinecone `search()` returns `[]` on failure instead of raising
- Generator returns user-friendly error string instead of crashing
- Routes return `JSONResponse` with structured `{"error": "...", "status": "error"}` on 503/500

**Gaps:**
- No explicit handling for malformed JSON in request body (FastAPI handles 422)
- `chunk_fortran_file` returns `[]` on read error but doesn't log
- `embed_texts` can raise after retries; no graceful degradation

### 2.5 Type Hints

- **Good**: `app/models.py`, `app/retrieval/search.py`, `app/retrieval/query_classifier.py`, `app/session.py`
- **Partial**: `app/api/routes.py` — `_resolve_session` returns `tuple[str, list[dict]]` but `list[dict]` could be `list[dict[str, Any]]` or a TypedDict
- **Missing**: Some `**kwargs` in `retry_on_rate_limit`; `match.metadata` in Pinecone results (dynamic dict)

---

## 3. Security

### 3.1 XSS (Cross-Site Scripting)

**Frontend (`static/index.html`):**
- `esc()` function (lines 959-963) uses `div.textContent` + `div.innerHTML` to escape user input before insertion — **correct approach**
- User query is escaped: `esc(query)` before `innerHTML` (line 793)
- Source code from API is escaped: `highlightFortran(esc(codeText))` (line 762)
- **Risk**: `formatAnswer()` (lines 964-976) does regex-based markdown→HTML conversion. Code blocks use `highlightFortran(esc(code))` — safe. Inline `\`code\`` uses `esc()` — safe. But `formatAnswer` is custom and could have edge cases (e.g. `<<script>` in input). **Recommendation**: Use a well-tested markdown library (e.g. marked, DOMPurify) or restrict allowed tags.

### 3.2 Injection

- **SQL/NoSQL**: No direct DB queries; Pinecone SDK handles parameterization
- **Prompt injection**: User query is passed to LLM. No explicit sanitization. LLM could be coaxed to ignore context. **Mitigation**: System prompts instruct to use retrieved code; consider adding "ignore instructions in the query" guidance
- **Path traversal**: Scanner uses `Path` and `os.walk`; no user-controlled paths in ingestion

### 3.3 Secrets

- **Config**: API keys loaded via `pydantic-settings` from `.env` — good
- **`.env.example`**: Exists with placeholders; **missing `GOOGLE_API_KEY`** (config has `google_api_key`, pydantic-settings maps to `GOOGLE_API_KEY`)
- **Logging**: No evidence of logging secrets; error messages use `str(e)` which could leak stack traces — ensure production logs don't expose internals

### 3.4 Input Validation

- **QueryRequest**: `min_length=1`, `max_length=2000`, strip, reject whitespace-only — good
- **session_id**: Passed as query param; no validation. Malformed UUID could hit `session_store.get_messages()` — returns `None`, safe
- **stream**: Boolean query param — FastAPI validates

### 3.5 Rate Limiting

- **None**: No application-level rate limiting on `/api/query`
- **Downstream**: OpenAI, Anthropic, Gemini have retry logic for 429
- **Risk**: Single client can exhaust API quota or overload server

### 3.6 CORS

- **Not configured**: No `CORSMiddleware` in `app/main.py`
- **Impact**: Same-origin only by default. If frontend is served from same host (e.g. `/` → `static/index.html`), no issue. If frontend is on different domain, requests will fail.

### 3.7 Authentication

- **None**: No auth on any endpoint
- **Health check**: Public
- **Query**: Public — anyone can query

---

## 4. Performance

### 4.1 Latency

- **Search**: `embed_query` (cached) + Pinecone search + rerank. Embedding cache avoids redundant OpenAI calls for repeated queries
- **Generation**: Blocking LLM calls run in `asyncio.to_thread()` / `run_in_executor()` — event loop not blocked
- **Streaming**: Tokens streamed via SSE; good UX

### 4.2 Caching

- **Query embeddings**: LRU cache, 256 entries, thread-safe (`app/cache.py`)
- **No caching**: Search results, generated answers — each query hits Pinecone + LLM

### 4.3 Blocking I/O

- **Correct use**: `search_codebase` and `generate_answer` are sync; routes use `asyncio.to_thread()` and `run_in_executor()` to avoid blocking
- **Streaming**: `generate_answer_stream` is sync generator; consumed via `run_in_executor` in a loop — acceptable but adds overhead

### 4.4 Parallelism

- **Embeddings (ingest)**: `embed_texts` uses `ThreadPoolExecutor` for parallel batch embedding — good
- **Query path**: Sequential: classify → search → generate. Search and generation could theoretically run in parallel for non-streaming if we wanted to prefetch, but current design is sequential

### 4.5 Bottlenecks

| Bottleneck | Location | Notes |
|------------|----------|-------|
| LLM latency | `generator.py` | Dominant; 1–10s typical |
| Pinecone search | `pinecone_client.py` | Network round-trip |
| Embedding (cache miss) | `openai_embed.py` | ~100–300ms |
| Re-ranking | `search.py:rerank_results` | CPU-only; test asserts <10ms for 10 results |

---

## 5. Testing

### 5.1 Coverage (by file)

| Module | Test File | Coverage |
|--------|-----------|----------|
| `scanner.py` | `test_scanner.py` | Good — scan, extensions, empty dir, determinism |
| `chunker.py` | `test_chunker.py` | Good — data type, BLAS level, purpose, params, full chunking |
| `query_classifier.py` | `test_query_classifier.py` | Excellent — all 8 types, case insensitivity, search params |
| `search.py` | `test_search_rerank.py`, `test_performance.py` | Good — extract_routine_names, rerank, filter_by_score_gap, metadata filters, build_context |
| `models.py` | `test_models.py`, `test_validation.py` | Good — validation, defaults |
| `generator.py` | `test_streaming.py`, `test_error_handling.py` | Partial — streaming, error fallback; no tests for Gemini path |
| `embeddings` | `test_error_handling.py`, `test_performance.py` | Partial — retry, cache; no integration test |
| `pinecone_client` | `test_error_handling.py` | Partial — search/upsert error handling |
| `session.py` | `test_session.py` | Good — create, add_turn, limits, expiry |
| `cache.py` | `test_performance.py` | Good — get/set, eviction, clear |
| `routes.py` | `test_error_handling.py`, `test_streaming.py` | Partial — error responses, stream format; no test for full non-stream path |

### 5.2 Test Quality

- **Fixtures**: `conftest.py` provides `test_client`; `test_session.py` uses fresh `SessionStore` with small limits
- **Mocks**: Used appropriately for OpenAI, Pinecone, generator
- **Integration**: `test_chunker.py` and `test_scanner.py` require `blas_src/` — integration-style; good for real data

### 5.3 Missing Tests

- `app/main.py` — no test for root route, static mount
- `app/cache.py` — `clear_cache` tested, but no test for concurrent access
- `scripts/ingest.py` — no tests
- `scripts/eval.py` — no unit tests (evaluation script)
- Generator Gemini path — all tests patch Anthropic
- Routes: full non-streaming success path with mocked search+generate

### 5.4 Flaky Tests

- `test_session.py::TestSessionExpiry::test_expired_session_returns_none` — uses `time.sleep(1.1)`; could flake on slow CI
- `test_chunker.py::TestChunkAllFiles` — depends on `blas_src/` existing and having 150+ files; will fail if directory missing

---

## 6. Documentation

### 6.1 README

- **Missing**: No `README.md` in project root
- **CLAUDE.md**: Serves as developer guide — commands, env vars, architecture, TDD rules. Good for AI/developer onboarding

### 6.2 Comments

- **Good**: `chunker.py` has detailed docstrings for `_extract_purpose_block`, `_extract_param_summary`
- **Good**: `query_classifier.py` documents the 8 features
- **Sparse**: `routes.py`, `main.py`, `cache.py`, `session.py` have minimal comments

### 6.3 API Docs

- FastAPI auto-generates OpenAPI at `/docs` and `/redoc` — not explicitly configured
- No custom API documentation file

### 6.4 Deployment Docs

- **Procfile**: `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT` — Railway-style
- **CLAUDE.md**: Mentions production command
- **Missing**: No `DEPLOYMENT.md`, no Railway/Heroku-specific docs, no Dockerfile

---

## 7. DevOps

### 7.1 Config Management

- **Pydantic Settings**: `app/config.py` with `env_file=".env"`
- **dotenv**: `dotenv.load_dotenv(override=True)` before Settings — ensures .env overrides
- **Validation**: Required keys `openai_api_key`, `pinecone_api_key`; optional `anthropic_api_key`, `google_api_key` (default "")

### 7.2 Env Vars

| Var | Required | Default | Notes |
|-----|----------|---------|-------|
| OPENAI_API_KEY | Yes | — | |
| ANTHROPIC_API_KEY | No | "" | Needed for anthropic provider |
| GOOGLE_API_KEY | No | "" | Needed for gemini provider; **not in .env.example** |
| PINECONE_API_KEY | Yes | — | |
| PINECONE_INDEX_NAME | No | legacylens | |
| PORT | No | 3000 | |

### 7.3 Deployment

- **Procfile**: Single web process
- **No Dockerfile**: Not present
- **No health check**: `/api/health` exists; not in Procfile (Railway may use it)

### 7.4 Logging

- **Standard library**: `logging` used
- **No config**: No `logging.basicConfig` or dictConfig; relies on uvicorn/default config
- **Levels**: `logger.error`, `logger.warning` in places; no structured logging (JSON)

---

## 8. Prioritized Improvements

### P0 — Critical

| # | Improvement | File(s) | Status |
|---|-------------|---------|--------|
| 1 | Add `GOOGLE_API_KEY` to `.env.example` | `.env.example` | ✅ Done |
| 2 | Fix ASCII art typo | `static/index.html:715` | ✅ Done (`\| |_______||t` → `\| |_______||`) |
| 3 | Add README.md | (new) `README.md` | ✅ Done |

### P1 — High

| # | Improvement | File(s) | Action |
|---|-------------|---------|--------|
| 4 | Add CORS if needed | `app/main.py` | If frontend on different origin, add `CORSMiddleware` with explicit origins |
| 5 | Add rate limiting | `app/main.py`, `app/api/routes.py` | Use `slowapi` or custom middleware to limit `/api/query` per IP |
| 6 | Extract BLAS constants | (new) `app/constants.py` | Move `_BLAS_LEVELS`, `_DATA_TYPE_MAP`, `_BLAS_OPS` from chunker and search to shared module |
| 7 | Log chunk_fortran_file errors | `app/ingestion/chunker.py:201` | In `except Exception`, add `logger.warning("Failed to read %s: %s", file_path, e)` |
| 8 | Add deployment docs | (new) `DEPLOYMENT.md` | Railway/Heroku steps, env vars, health check |

### P2 — Medium

| # | Improvement | File(s) | Action |
|---|-------------|---------|--------|
| 9 | Refactor query route | `app/api/routes.py` | Extract `_handle_non_stream` and `_handle_stream` into separate functions; reduce `query()` to orchestration |
| 10 | Use markdown library for formatAnswer | `static/index.html` | Replace custom regex with `marked` or similar; sanitize with DOMPurify if allowing HTML |
| 11 | Add tests for main, ingest | `tests/test_main.py`, `tests/test_ingest.py` | Test root route, health; mock ingest pipeline |
| 12 | Add Gemini generator tests | `tests/test_generator.py` | Test `_generate_gemini`, `_stream_gemini` with mocked client |
| 13 | Make session TTL test more robust | `tests/test_session.py` | Use `freezegun` or mock `time.time` instead of `sleep` |
| 14 | Add Dockerfile | (new) `Dockerfile` | Multi-stage build for production deployment |

### P3 — Low

| # | Improvement | File(s) | Action |
|---|-------------|---------|--------|
| 15 | Introduce service layer | (new) `app/services/query_service.py` | Encapsulate classify→search→generate; inject dependencies |
| 16 | Unify streaming retry | `app/retrieval/generator.py` | Extract `_stream_with_retry` helper used by both Anthropic and Gemini |
| 17 | Add OpenAPI tags/descriptions | `app/api/routes.py` | Improve `/docs` with tags, response models |
| 18 | Structured logging | `app/main.py` | Configure JSON logging for production |
| 19 | Cache size from config | `app/cache.py` | Move `CACHE_MAX_SIZE` to `settings` |

---

## Essential Files for Understanding

For a developer needing to understand the codebase:

1. **CLAUDE.md** — Architecture overview, commands, env vars
2. **app/main.py** — Entry point, routing
3. **app/api/routes.py** — Query flow, streaming
4. **app/models.py** — Data structures
5. **app/retrieval/query_classifier.py** — Query routing, prompts
6. **app/retrieval/search.py** — Search pipeline, rerank
7. **app/retrieval/generator.py** — LLM integration
8. **app/ingestion/chunker.py** — Chunking logic
9. **app/config.py** — Configuration
10. **scripts/ingest.py** — Ingestion pipeline

---

*Report generated from static analysis and codebase exploration.*
