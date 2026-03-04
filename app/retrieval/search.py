"""Search pipeline: embed query -> search Pinecone -> re-rank -> assemble context."""

import re
import time
from app.embeddings.openai_embed import embed_query
from app.vectordb.pinecone_client import search as pinecone_search
from app.models import CodeChunk, ChunkMetadata, SearchResult


# ── Routine name extraction for re-ranking ──

_BLAS_PREFIXES = ["S", "D", "C", "Z"]
_BLAS_OPS = [
    # Level 1
    "ROTG", "ROT", "ROTMG", "ROTM", "SWAP", "SCAL", "COPY", "AXPY",
    "DOT", "DOTU", "DOTC", "NRM2", "ASUM", "AMAX", "IAMAX", "AXPBY",
    # Level 2
    "GEMV", "GBMV", "HEMV", "HBMV", "HPMV", "SYMV", "SBMV", "SPMV",
    "TRMV", "TBMV", "TPMV", "TRSV", "TBSV", "TPSV", "GER", "GERU",
    "GERC", "HER", "HPR", "HER2", "HPR2", "SYR", "SPR", "SYR2", "SPR2",
    # Level 3
    "GEMM", "SYMM", "HEMM", "SYRK", "HERK", "SYR2K", "HER2K",
    "TRMM", "TRSM", "GEMMTR",
]
_UTILITY_NAMES = {"XERBLA", "XERBLA_ARRAY", "LSAME"}

_KNOWN_BLAS_NAMES: set[str] = set()
for _pfx in _BLAS_PREFIXES:
    for _op in _BLAS_OPS:
        _KNOWN_BLAS_NAMES.add(_pfx + _op)
_KNOWN_BLAS_NAMES.update(_UTILITY_NAMES)

_ROUTINE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{3,})\b")

EXACT_MATCH_BOOST = 2.0
SCORE_GAP_RATIO = 0.6  # keep results scoring ≥ 60% of the top result


_LEVEL_PATTERN = re.compile(r"\blevel\s+([1-3])\b", re.IGNORECASE)


def detect_query_metadata_filters(query: str) -> dict | None:
    """Detect BLAS level mentions in a query and return a Pinecone filter dict.

    Returns e.g. {"blas_level": {"$eq": "3"}} or None if no level is mentioned.
    """
    m = _LEVEL_PATTERN.search(query)
    if m:
        return {"blas_level": {"$eq": m.group(1)}}
    return None


def extract_routine_names(query: str) -> set[str]:
    """Extract BLAS routine names mentioned in a query string."""
    upper_query = query.upper()
    candidates = set(_ROUTINE_PATTERN.findall(upper_query))
    return candidates & _KNOWN_BLAS_NAMES


def rerank_results(
    results: list[SearchResult], query: str, top_k: int
) -> list[SearchResult]:
    """Re-rank results by boosting chunks whose subroutine_name appears in the query.

    The boost is used only for sorting order — the original Pinecone score
    is preserved for display so users see honest relevance scores.
    """
    if not results:
        return []
    mentioned = extract_routine_names(query)
    if not mentioned:
        return results[:top_k]

    # Sort by boosted score, but keep original score for display
    decorated = []
    for r in results:
        sort_score = r.score
        if r.chunk.metadata.subroutine_name.upper() in mentioned:
            sort_score *= EXACT_MATCH_BOOST
        decorated.append((sort_score, r))

    decorated.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in decorated[:top_k]]


def filter_by_score_gap(
    results: list[SearchResult], ratio: float = SCORE_GAP_RATIO
) -> list[SearchResult]:
    """Drop results whose score falls below `ratio` × the top score.

    After re-ranking boosts exact matches (e.g. 0.49 → 0.99), semantic
    neighbors stay at ~0.47.  A 0.6 ratio keeps only the boosted results
    while preserving clusters of similar-scoring non-boosted hits.
    """
    if not results:
        return []
    threshold = results[0].score * ratio
    return [r for r in results if r.score >= threshold]


# ── Search pipeline ──


def search_codebase(query: str, top_k: int = 5, threshold: float = 0.2) -> tuple[list[SearchResult], float]:
    """Search the codebase and return re-ranked results with timing."""
    start = time.time()

    # Embed the query
    query_embedding = embed_query(query)

    # Over-fetch from Pinecone for re-ranking, with optional metadata filter
    fetch_k = top_k * 2
    metadata_filter = detect_query_metadata_filters(query)
    matches = pinecone_search(query_embedding, top_k=fetch_k, metadata_filter=metadata_filter)

    # Convert to SearchResult objects, filtering by threshold
    results = []
    for match in matches:
        if match.score < threshold:
            continue
        meta = match.metadata
        chunk = CodeChunk(
            id=match.id,
            text=meta.get("text", ""),
            metadata=ChunkMetadata(
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                subroutine_name=meta.get("subroutine_name", ""),
                blas_level=meta.get("blas_level", "unknown"),
                data_type=meta.get("data_type", "unknown"),
                description=meta.get("description", ""),
                line_count=meta.get("line_count", 0),
            ),
        )
        results.append(SearchResult(chunk=chunk, score=match.score))

    # Filter by score gap on raw Pinecone scores (before re-ranking boost),
    # then re-rank to promote exact name matches to the top.
    results = filter_by_score_gap(results)
    results = rerank_results(results, query, top_k)

    elapsed_ms = (time.time() - start) * 1000
    return results, elapsed_ms


def build_context(results: list[SearchResult], max_chars: int = 30000) -> str:
    """Assemble retrieved chunks into a context string for the LLM.

    Args:
        results: Retrieved search results.
        max_chars: Maximum character limit for the context string.
                   Prevents exceeding model context windows and reduces latency.
                   The first source is always included even if it exceeds the limit.
    """
    separator = "\n\n---\n\n"
    parts = []
    total_len = 0

    for i, r in enumerate(results, 1):
        m = r.chunk.metadata
        header = f"[Source {i}] {m.file_path}:{m.start_line}-{m.end_line}"
        if m.subroutine_name:
            header += f" | {m.subroutine_name}"
        header += f" (score: {r.score:.3f})"
        part = f"{header}\n{r.chunk.text}"

        # Always include the first source; after that, check the limit
        if i > 1:
            added_len = len(separator) + len(part)
            if total_len + added_len > max_chars:
                break

        parts.append(part)
        total_len = len(separator.join(parts))

    return separator.join(parts)
