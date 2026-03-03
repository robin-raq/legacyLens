"""Tests for performance improvements: caching, context limits, timeouts."""

import pytest
from app.retrieval.search import build_context
from app.models import CodeChunk, ChunkMetadata, SearchResult


# ── Helpers ──

def _make_result(name: str, text: str, score: float = 0.5) -> SearchResult:
    """Create a SearchResult with given subroutine name and text."""
    return SearchResult(
        chunk=CodeChunk(
            id=f"test::{name}",
            text=text,
            metadata=ChunkMetadata(
                file_path=f"SRC/{name.lower()}.f",
                start_line=1,
                end_line=50,
                subroutine_name=name,
                blas_level="1",
                data_type="double real",
            ),
        ),
        score=score,
    )


# ── 2.5: Context size limit ──

class TestBuildContextLimit:
    def test_default_has_no_limit(self):
        """Without max_chars, all chunks are included."""
        results = [_make_result("DGEMM", "x" * 500), _make_result("DGEMV", "y" * 500)]
        context = build_context(results)
        assert "DGEMM" in context
        assert "DGEMV" in context

    def test_max_chars_truncates(self):
        """With max_chars, context is truncated to fit."""
        results = [
            _make_result("DGEMM", "x" * 5000),
            _make_result("DGEMV", "y" * 5000),
            _make_result("DTRSV", "z" * 5000),
        ]
        context = build_context(results, max_chars=6000)
        assert len(context) <= 6000

    def test_max_chars_keeps_at_least_first(self):
        """Even with a tight limit, at least the first source is included."""
        results = [_make_result("DGEMM", "x" * 200)]
        context = build_context(results, max_chars=50)
        # First source is always included even if over limit
        assert "DGEMM" in context

    def test_max_chars_excludes_later_sources(self):
        """Sources that would exceed max_chars are dropped."""
        results = [
            _make_result("DGEMM", "x" * 3000),
            _make_result("DGEMV", "y" * 3000),
            _make_result("DTRSV", "z" * 3000),
        ]
        context = build_context(results, max_chars=4000)
        assert "DGEMM" in context
        # At least one later source should be excluded
        source_count = context.count("[Source")
        assert source_count < 3


# ── 2.2: Query embedding cache ──

class TestQueryCache:
    def test_cache_import(self):
        """Cache module exists and has expected interface."""
        from app.cache import get_cached_embedding, set_cached_embedding, clear_cache
        assert callable(get_cached_embedding)
        assert callable(set_cached_embedding)
        assert callable(clear_cache)

    def test_cache_miss_returns_none(self):
        """Cache returns None for unseen queries."""
        from app.cache import get_cached_embedding, clear_cache
        clear_cache()
        result = get_cached_embedding("never seen before query xyz")
        assert result is None

    def test_cache_hit_returns_value(self):
        """After setting, cache returns the stored embedding."""
        from app.cache import get_cached_embedding, set_cached_embedding, clear_cache
        clear_cache()
        embedding = [0.1, 0.2, 0.3]
        set_cached_embedding("test query", embedding)
        result = get_cached_embedding("test query")
        assert result == embedding

    def test_cache_different_queries_independent(self):
        """Different queries have independent cache entries."""
        from app.cache import get_cached_embedding, set_cached_embedding, clear_cache
        clear_cache()
        set_cached_embedding("query A", [1.0, 2.0])
        set_cached_embedding("query B", [3.0, 4.0])
        assert get_cached_embedding("query A") == [1.0, 2.0]
        assert get_cached_embedding("query B") == [3.0, 4.0]

    def test_cache_max_size(self):
        """Cache evicts old entries when max size is reached."""
        from app.cache import get_cached_embedding, set_cached_embedding, clear_cache, CACHE_MAX_SIZE
        clear_cache()
        # Fill beyond max
        for i in range(CACHE_MAX_SIZE + 10):
            set_cached_embedding(f"query_{i}", [float(i)])
        # Latest entries should exist
        assert get_cached_embedding(f"query_{CACHE_MAX_SIZE + 9}") is not None

    def test_clear_cache(self):
        """clear_cache removes all entries."""
        from app.cache import get_cached_embedding, set_cached_embedding, clear_cache
        set_cached_embedding("test", [1.0])
        clear_cache()
        assert get_cached_embedding("test") is None


# ── 2.4: Timeout configuration ──

class TestTimeoutConfig:
    def test_openai_timeout_configured(self):
        """OpenAI client has explicit timeout."""
        from app.embeddings.openai_embed import _client
        assert _client.timeout is not None

    def test_anthropic_timeout_configured(self):
        """Anthropic client has explicit timeout."""
        from app.retrieval.generator import _client
        assert _client.timeout is not None


# ── 2.1: Async route uses thread pool ──

class TestAsyncRoute:
    def test_route_is_async(self):
        """The query route should be an async function."""
        import inspect
        from app.api.routes import query
        assert inspect.iscoroutinefunction(query)
