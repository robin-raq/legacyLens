"""OpenAI embedding client for text-embedding-3-small."""

from openai import OpenAI
from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048, but keep batches manageable


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching as needed."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = _client.embeddings.create(model=MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    response = _client.embeddings.create(model=MODEL, input=[query])
    return response.data[0].embedding
