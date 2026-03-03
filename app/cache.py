"""In-memory LRU cache for query embeddings.

Avoids redundant OpenAI API calls for repeated or identical queries.
Uses OrderedDict for O(1) LRU eviction.
"""

from collections import OrderedDict
import threading

CACHE_MAX_SIZE = 256

_lock = threading.Lock()
_cache: OrderedDict[str, list[float]] = OrderedDict()


def get_cached_embedding(query: str) -> list[float] | None:
    """Return cached embedding for a query, or None on miss."""
    with _lock:
        if query in _cache:
            _cache.move_to_end(query)  # Mark as recently used
            return _cache[query]
    return None


def set_cached_embedding(query: str, embedding: list[float]) -> None:
    """Store an embedding in the cache, evicting the oldest if full."""
    with _lock:
        if query in _cache:
            _cache.move_to_end(query)
        _cache[query] = embedding
        while len(_cache) > CACHE_MAX_SIZE:
            _cache.popitem(last=False)  # Evict oldest


def clear_cache() -> None:
    """Remove all entries from the cache."""
    with _lock:
        _cache.clear()
