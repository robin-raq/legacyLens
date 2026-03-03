"""Search pipeline: embed query -> search Pinecone -> assemble context."""

import time
from app.embeddings.openai_embed import embed_query
from app.vectordb.pinecone_client import search as pinecone_search
from app.models import CodeChunk, ChunkMetadata, SearchResult


def search_codebase(query: str, top_k: int = 5, threshold: float = 0.2) -> tuple[list[SearchResult], float]:
    """Search the codebase and return results with timing."""
    start = time.time()

    # Embed the query
    query_embedding = embed_query(query)

    # Search Pinecone
    matches = pinecone_search(query_embedding, top_k=top_k)

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
