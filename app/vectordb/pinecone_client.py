"""Pinecone vector database client with error handling."""

import logging
from pinecone import Pinecone
from app.config import settings
from app.models import CodeChunk

logger = logging.getLogger(__name__)

_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = _pc.Index(settings.pinecone_index_name)

BATCH_SIZE = settings.pinecone_batch_size


def upsert_chunks(chunks: list[CodeChunk], embeddings: list[list[float]]) -> int:
    """Upsert chunks with their embeddings into Pinecone.

    Handles batch failures gracefully — logs errors and continues with
    remaining batches instead of crashing the entire ingest.
    """
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.id,
            "values": embedding,
            "metadata": {
                "file_path": chunk.metadata.file_path,
                "start_line": chunk.metadata.start_line,
                "end_line": chunk.metadata.end_line,
                "subroutine_name": chunk.metadata.subroutine_name,
                "blas_level": chunk.metadata.blas_level,
                "data_type": chunk.metadata.data_type,
                "description": chunk.metadata.description,
                "line_count": chunk.metadata.line_count,
                "text": chunk.text[:settings.pinecone_metadata_max_chars],
            },
        })

    upserted = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        try:
            _index.upsert(vectors=batch)
            upserted += len(batch)
        except Exception as e:
            logger.error("Pinecone upsert failed for batch at offset %d: %s", i, e)

    return upserted


def search(query_embedding: list[float], top_k: int = 5, metadata_filter: dict | None = None) -> list:
    """Search for similar chunks, optionally filtered by metadata.

    Returns empty list on failure instead of crashing, so the search
    pipeline can still return a graceful "no results" response.

    Args:
        metadata_filter: Pinecone filter dict, e.g. {"blas_level": {"$eq": "3"}}.
                         Applied server-side before ANN search.
    """
    kwargs = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    try:
        results = _index.query(**kwargs)
        return results.matches
    except Exception as e:
        logger.error("Pinecone search failed: %s", e)
        return []


def get_index_stats() -> dict:
    """Get index statistics."""
    return _index.describe_index_stats()
