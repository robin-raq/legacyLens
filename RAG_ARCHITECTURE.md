# LegacyLens RAG Architecture

## Vector Database Selection

**Choice: Pinecone (managed cloud, free tier)**

Pinecone was selected over ChromaDB, Qdrant, and pgvector for this project. Key factors:

- **Zero ops overhead** -- managed service, no Docker containers or sidecar databases alongside Railway
- **Free tier** -- 1 index, 100K vectors. BLAS needs ~170 vectors, well within limits
- **Python SDK** -- first-class support, simple API (create index, upsert, query)
- **Metadata filtering** -- rich filtering on `blas_level`, `data_type`, `subroutine_name`

**Tradeoff accepted:** No hybrid search (vector + keyword). For code search, queries like exact function names rely on semantic similarity rather than exact match. Mitigation: function names are included prominently in chunk metadata text that gets embedded.

## Embedding Strategy

**Choice: OpenAI text-embedding-3-small (1536 dimensions)**

| Criteria | text-embedding-3-small | Voyage Code 2 | sentence-transformers |
|----------|----------------------|----------------|----------------------|
| Cost/1M tokens | $0.02 | $0.12 | Free (local) |
| Dimensions | 1536 | 1536 | Varies |
| Code-optimized | No | Yes | Depends |

**Why this model:** BLAS Fortran files are 40-60% English-language header comments describing purpose, parameters, and mathematical operations. General-purpose embeddings work well on this structured documentation. Cost is 6x cheaper than Voyage Code 2, and our total embedding cost for the full codebase is under $0.01. No local GPU needed -- runs from Railway via API.

**Tradeoff accepted:** Not code-optimized. If retrieval precision were poor on pure code queries, Voyage Code 2 would be the upgrade path. In practice, 93% P@5 shows the header comments carry enough semantic signal.

## Chunking Approach

**Strategy: Subroutine-level splitting (primary) + fixed-size fallback**

BLAS has ideal structure for syntax-aware chunking: each `.f` file typically contains exactly one subroutine with clear boundaries.

- **Start boundary:** `SUBROUTINE name(...)` or `FUNCTION name(...)` (regex, case insensitive)
- **End boundary:** `END` / `END SUBROUTINE` / `END FUNCTION`
- **Fallback:** Non-subroutine content (makefiles, test files) uses 1500-token chunks with 200-token overlap

Each chunk is enriched with metadata before embedding:

```
Subroutine: DGEMM
Level: 3
Type: double
File: BLAS/SRC/dgemm.f
Description: DGEMM performs one of the matrix-matrix operations...

{full header comment block}
{subroutine signature with parameters}
{implementation body (truncated at ~2000 tokens)}
```

Prepending metadata to the embedding text means semantic search finds the right subroutine even when queries use different terminology ("matrix multiply" matches DGEMM's description).

**Result:** 169 chunks from ~150 Fortran source files. Average chunk size is 50-300 lines, fitting well within the embedding model's context window.

## Retrieval Pipeline

```
User query
  -> Query classifier (regex, 8 types)
  -> OpenAI embedding (same model as ingestion)
  -> Pinecone similarity search (top-k varies by query type: 5-10)
  -> Score-gap filtering (drop results below 60% of top score)
  -> Exact-match boosting (2x boost for subroutine name matches)
  -> Context assembly (combine top chunks, max 30K chars)
  -> Gemini 2.5 Flash generation (feature-specific system prompt)
  -> SSE streaming to frontend
```

**Query classifier:** Regex-based routing to one of 8 query types (EXPLAIN, DOCUMENT, PATTERN, LOGIC, DEPENDENCY, IMPACT, TRANSLATION, BUG_PATTERN). Each type has its own system prompt and `top_k` value -- pattern queries retrieve 10 results while explanation queries retrieve 5.

**Re-ranking:** Two-stage post-retrieval filtering:
1. **Score-gap filter** -- if a result's score drops below 60% of the top result's score, it's excluded. This prevents low-relevance noise.
2. **Exact-match boost** -- if the query contains a subroutine name that appears in a result's metadata, that result's score is doubled. This ensures `DGEMM` queries always surface `dgemm.f`.

**Context assembly:** Retrieved chunks are concatenated with metadata headers, truncated to 30K characters to stay within the LLM context window while providing rich code context.

## Failure Modes

| Failure Mode | Likelihood | Mitigation |
|---|---|---|
| Query about non-existent subroutine | High | Score threshold (0.2) filters irrelevant results; empty results return "no relevant code found" |
| Wrong precision variant returned (SSYRK vs DSYRK) | Medium | Data type prefix (S/D/C/Z) included in chunk metadata; exact-match boosting helps |
| Vague query ("show me some code") | Medium | System prompt instructs LLM to ask clarifying questions when context is insufficient |
| Pattern Detection lower precision | Medium | Broader top-k (10) retrieves more candidates, but some may be tangential. P@5 = 47% for this feature vs 100% for others |
| Rate limit on embedding/LLM API | Low | Retry with linear backoff (up to 3 attempts). Embedding results cached in-memory |
| Pinecone service unavailable | Low | Graceful degradation: returns 503 JSON error instead of crashing |

**Known limitation:** Pattern Detection queries ("find similar patterns to X") retrieve broadly, pulling in tangentially related subroutines. This is inherent to the query type -- pattern-finding needs diverse results, but not all are equally relevant.

## Performance Results

Evaluation: 24 queries across 8 features (3 queries per feature), run against Gemini 2.5 Flash.

| Metric | Target | Actual |
|---|---|---|
| Retrieval precision (P@5) | >70% | **93%** |
| Term recall | >70% | **96%** |
| Queries passed | 24/24 | **24/24** |
| Mean latency | <3s | 6.3s total; **<1s to first token** (SSE streaming) |
| Codebase coverage | 100% | **100%** (169 chunks from all files) |
| Ingestion throughput | 10K+ LOC in <5 min | **48,480 LOC in ~2 min** (264K LOC/s scan+chunk) |
| Answer accuracy | Correct file/line refs | **100%** (6/6 queries, 0 hallucinated files) |

**Per-feature breakdown:**

| Feature | P@5 | Term Recall | Avg Latency |
|---|---|---|---|
| Code Explanation | 100% | 94% | 3.2s |
| Documentation Gen | 100% | 92% | 3.6s |
| Dependency Mapping | 100% | 100% | 4.0s |
| Bug Pattern Search | 100% | 89% | 4.4s |
| Impact Analysis | 100% | 92% | 7.3s |
| Pattern Detection | 47% | 100% | 8.2s |
| Business Logic | 100% | 100% | 9.3s |
| Translation Hints | 100% | 100% | 10.1s |

**Latency note:** Mean time-to-last-token is 6.3s, above the 3s target. However, SSE streaming delivers the **first visible token in <1s**, so the user sees the answer building in real time rather than waiting for the full response. The latency breakdown: embedding ~200ms, Pinecone search + rerank ~300ms, LLM generation 4-9s (the bottleneck). Features with longer answers (Translation Hints, Business Logic) take longer because Gemini generates more tokens, not because retrieval is slow.
