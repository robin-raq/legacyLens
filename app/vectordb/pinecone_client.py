"""Pinecone vector database client."""

from pinecone import Pinecone
from app.config import settings
from app.models import CodeChunk

_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = _pc.Index(settings.pinecone_index_name)

BATCH_SIZE = 100


def upsert_chunks(chunks: list[CodeChunk], embeddings: list[list[float]]) -> int:
    """Upsert chunks with their embeddings into Pinecone."""
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
                "text": chunk.text[:10000],  # Pinecone metadata limit ~40KB
            },
        })

    upserted = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        _index.upsert(vectors=batch)
        upserted += len(batch)

    return upserted


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for similar chunks."""
    results = _index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    return results.matches


def get_index_stats() -> dict:
    """Get index statistics."""
    return _index.describe_index_stats()
