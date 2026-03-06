"""OpenAI embedding client with retry logic for rate limits."""

import logging
from openai import OpenAI, RateLimitError
from app.config import settings
from app.cache import get_cached_embedding, set_cached_embedding
from app.utils import retry_on_rate_limit

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key, timeout=settings.embedding_timeout)

MODEL = settings.embedding_model
DIMENSIONS = settings.embedding_dimensions
BATCH_SIZE = settings.embedding_batch_size


def _embed_batch_call(batch: list[str]) -> list[list[float]]:
    """Raw API call to embed a batch — no retry logic."""
    response = _client.embeddings.create(model=MODEL, input=batch)
    return [item.embedding for item in response.data]


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch of texts with retry on rate limit."""
    return retry_on_rate_limit(_embed_batch_call, batch, exc_type=RateLimitError)


def embed_texts(texts: list[str], parallel: bool = True) -> list[list[float]]:
    """Embed a list of texts, batching as needed.

    Args:
        texts: List of text strings to embed.
        parallel: If True, process batches concurrently using a thread pool.
    """
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    if parallel and len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            batch_results = list(pool.map(_embed_batch, batches))
        all_embeddings = []
        for batch_emb in batch_results:
            all_embeddings.extend(batch_emb)
        return all_embeddings

    # Sequential fallback (single batch or parallel=False)
    all_embeddings = []
    for batch in batches:
        all_embeddings.extend(_embed_batch(batch))
    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string, with caching and retry on rate limit."""
    cached = get_cached_embedding(query)
    if cached is not None:
        return cached

    def _call():
        response = _client.embeddings.create(model=MODEL, input=[query])
        return response.data[0].embedding

    embedding = retry_on_rate_limit(_call, exc_type=RateLimitError)
    set_cached_embedding(query, embedding)
    return embedding
