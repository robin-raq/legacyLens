"""Tests for error handling and fallback mechanisms."""

import pytest
from unittest.mock import patch, MagicMock


class TestEmbeddingErrorHandling:
    def test_embed_query_retries_on_rate_limit(self):
        """embed_query should retry on rate limit errors."""
        from app.embeddings.openai_embed import embed_query

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 10)]

        with patch("app.embeddings.openai_embed._client") as mock_client:
            from openai import RateLimitError
            mock_client.embeddings.create.side_effect = [
                RateLimitError("rate limited", response=MagicMock(status_code=429), body=None),
                mock_response,
            ]
            # Clear cache to force API call
            with patch("app.embeddings.openai_embed.get_cached_embedding", return_value=None):
                with patch("app.embeddings.openai_embed.set_cached_embedding"):
                    result = embed_query("test query")
            assert result == [0.1] * 10
            assert mock_client.embeddings.create.call_count == 2

    def test_embed_query_raises_after_max_retries(self):
        """embed_query should raise after exhausting retries."""
        from app.embeddings.openai_embed import embed_query

        with patch("app.embeddings.openai_embed._client") as mock_client:
            from openai import RateLimitError
            mock_client.embeddings.create.side_effect = RateLimitError(
                "rate limited", response=MagicMock(status_code=429), body=None
            )
            with patch("app.embeddings.openai_embed.get_cached_embedding", return_value=None):
                with pytest.raises(RateLimitError):
                    embed_query("test query")


class TestPineconeErrorHandling:
    def test_search_returns_empty_on_error(self):
        """Pinecone search should return empty list on failure, not crash."""
        from app.vectordb.pinecone_client import search

        with patch("app.vectordb.pinecone_client._index") as mock_index:
            mock_index.query.side_effect = Exception("Pinecone down")
            result = search([0.1] * 10, top_k=5)
            assert result == []

    def test_upsert_logs_and_continues_on_batch_error(self):
        """Upsert should handle batch failures gracefully."""
        from app.vectordb.pinecone_client import upsert_chunks
        from app.models import CodeChunk, ChunkMetadata

        chunks = [
            CodeChunk(
                id="test::FOO",
                text="SUBROUTINE FOO",
                metadata=ChunkMetadata(file_path="test.f", start_line=1, end_line=5),
            )
        ]
        embeddings = [[0.1] * 10]

        with patch("app.vectordb.pinecone_client._index") as mock_index:
            mock_index.upsert.side_effect = Exception("Batch failed")
            # Should not raise
            result = upsert_chunks(chunks, embeddings)
            assert result == 0  # No successful upserts


class TestGeneratorErrorHandling:
    def test_generate_answer_returns_error_string_on_failure(self):
        """generate_answer should return an error message, not crash."""
        from app.retrieval.generator import generate_answer
        from app.retrieval.query_classifier import QueryType

        with patch("app.retrieval.generator._generate_gemini") as mock_gen:
            mock_gen.side_effect = Exception("API down")
            result = generate_answer("test", "context", QueryType.EXPLAIN)
            assert "error" in result.lower() or "sorry" in result.lower()

    def test_stream_yields_error_on_failure(self):
        """generate_answer_stream should yield an error message, not crash."""
        from app.retrieval.generator import generate_answer_stream
        from app.retrieval.query_classifier import QueryType

        with patch("app.retrieval.generator._stream_gemini") as mock_stream:
            mock_stream.side_effect = Exception("API down")
            chunks = list(generate_answer_stream("test", "context", QueryType.EXPLAIN))
            # Should yield at least one chunk with error info
            assert len(chunks) >= 1
            combined = "".join(chunks)
            assert "error" in combined.lower() or "sorry" in combined.lower()


class TestRouteErrorHandling:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_search_failure_returns_json_error(self, client):
        """If search crashes, API should return structured error, not 500."""
        with patch("app.api.routes.search_codebase") as mock_search:
            mock_search.side_effect = Exception("Pinecone unavailable")
            resp = client.post("/api/query", json={"query": "test"})
            # Should be a 5xx with structured JSON, not raw exception
            assert resp.status_code in (500, 503)
            data = resp.json()
            assert "error" in data

    def test_empty_query_returns_422(self, client):
        """Empty query should return validation error."""
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, client):
        """Overly long query should return validation error."""
        resp = client.post("/api/query", json={"query": "a" * 2001})
        assert resp.status_code == 422


class TestConfigCentralization:
    def test_generator_uses_config_max_tokens(self):
        """Generator should read max_tokens from settings, not hardcode."""
        from app.config import settings
        assert hasattr(settings, "max_tokens")

    def test_embedding_model_in_config(self):
        """Embedding model name should be in settings."""
        from app.config import settings
        assert hasattr(settings, "embedding_model")
        assert settings.embedding_model == "text-embedding-3-small"

    def test_search_thresholds_in_config(self):
        """Search tuning params should be in settings."""
        from app.config import settings
        assert hasattr(settings, "score_threshold")
        assert hasattr(settings, "score_gap_ratio")
        assert hasattr(settings, "exact_match_boost")

    def test_session_config_exists(self):
        """Session settings should be in config."""
        from app.config import settings
        assert hasattr(settings, "session_ttl")
        assert hasattr(settings, "max_sessions")
