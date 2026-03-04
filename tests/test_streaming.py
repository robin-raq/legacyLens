"""Tests for SSE streaming: generator streaming + route streaming."""

from unittest.mock import MagicMock, patch
import pytest


# ── generate_answer_stream ──


class TestGenerateAnswerStream:
    def test_function_exists(self):
        """generate_answer_stream should be importable."""
        from app.retrieval.generator import generate_answer_stream
        assert callable(generate_answer_stream)

    def test_returns_generator(self):
        """Should return a generator that yields text deltas."""
        from app.retrieval.generator import generate_answer_stream

        # Mock the Anthropic streaming client
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " world", "!"])
        mock_client.messages.stream.return_value = mock_stream

        with patch("app.retrieval.generator._get_anthropic_client", return_value=mock_client), \
             patch("app.retrieval.generator.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            mock_settings.anthropic_model = "test-model"
            gen = generate_answer_stream("test query", "test context")
            chunks = list(gen)
            assert chunks == ["Hello", " world", "!"]

    def test_passes_correct_model(self):
        """Should use settings.anthropic_model for the stream call."""
        from app.retrieval.generator import generate_answer_stream

        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["ok"])
        mock_client.messages.stream.return_value = mock_stream

        with patch("app.retrieval.generator._get_anthropic_client", return_value=mock_client), \
             patch("app.retrieval.generator.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            mock_settings.anthropic_model = "claude-haiku-4-5-20251001"
            list(generate_answer_stream("q", "ctx"))
            call_kwargs = mock_client.messages.stream.call_args[1]
            assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_uses_feature_specific_prompt(self):
        """Should use the system prompt for the given query type."""
        from app.retrieval.generator import generate_answer_stream
        from app.retrieval.query_classifier import QueryType

        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["ok"])
        mock_client.messages.stream.return_value = mock_stream

        with patch("app.retrieval.generator._get_anthropic_client", return_value=mock_client), \
             patch("app.retrieval.generator.settings") as mock_settings:
            mock_settings.llm_provider = "anthropic"
            mock_settings.anthropic_model = "test-model"
            list(generate_answer_stream("q", "ctx", QueryType.DOCUMENT))
            call_kwargs = mock_client.messages.stream.call_args[1]
            assert "documentation" in call_kwargs["system"].lower() or "document" in call_kwargs["system"].lower()


# ── SSE format helper ──


class TestFormatSSEEvent:
    def test_format_sources_event(self):
        """SSE sources event should have correct format."""
        from app.api.routes import format_sse_event
        event = format_sse_event("sources", {"items": [1, 2]})
        assert event.startswith("event: sources\n")
        assert "data: " in event
        assert event.endswith("\n\n")

    def test_format_token_event(self):
        """SSE token event should contain the text chunk."""
        from app.api.routes import format_sse_event
        event = format_sse_event("token", {"content": "Hello"})
        assert "event: token\n" in event
        assert '"Hello"' in event or "'Hello'" in event

    def test_format_done_event(self):
        """SSE done event should signal completion."""
        from app.api.routes import format_sse_event
        event = format_sse_event("done", {"total_time_ms": 1234.5})
        assert "event: done\n" in event


# ── Streaming route ──


class TestStreamingRoute:
    def test_query_endpoint_accepts_stream_param(self):
        """POST /api/query?stream=true should not error."""
        from app.main import app
        from fastapi.testclient import TestClient

        # Just verify the parameter is accepted (will fail on API calls,
        # but should not return 422 for the param itself)
        client = TestClient(app)
        # Mock the entire pipeline to avoid real API calls
        with patch("app.api.routes.search_codebase") as mock_search, \
             patch("app.api.routes.generate_answer_stream") as mock_gen:
            mock_search.return_value = ([], 10.0)
            mock_gen.return_value = iter(["test"])
            response = client.post(
                "/api/query?stream=true",
                json={"query": "test"},
            )
            # Should get a streaming response (200) not a validation error (422)
            assert response.status_code == 200

    def test_stream_response_content_type(self):
        """Streaming response should have text/event-stream content type."""
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch("app.api.routes.search_codebase") as mock_search, \
             patch("app.api.routes.generate_answer_stream") as mock_gen:
            mock_search.return_value = ([], 10.0)
            mock_gen.return_value = iter([])
            response = client.post(
                "/api/query?stream=true",
                json={"query": "test"},
            )
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_non_stream_still_returns_json(self):
        """Without stream=true, endpoint should return JSON as before."""
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        with patch("app.api.routes.search_codebase") as mock_search, \
             patch("app.api.routes.generate_answer") as mock_gen:
            mock_search.return_value = ([], 10.0)
            mock_gen.return_value = "test answer"
            response = client.post(
                "/api/query",
                json={"query": "test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data

    def test_stream_emits_done_event_after_tokens(self):
        """SSE stream must end with a 'done' event (PEP 479 regression guard)."""
        from app.main import app
        from fastapi.testclient import TestClient
        from app.models import CodeChunk, ChunkMetadata

        fake_chunk = CodeChunk(
            id="test::FOO", text="test code",
            metadata=ChunkMetadata(file_path="test.f", start_line=1, end_line=10,
                                   subroutine_name="FOO"),
        )
        fake_result = MagicMock(score=0.95, chunk=fake_chunk)
        fake_result.model_dump.return_value = {
            "score": 0.95,
            "chunk": {"id": "test::FOO", "text": "test code",
                      "metadata": {"file_path": "test.f", "start_line": 1,
                                   "end_line": 10, "subroutine_name": "FOO",
                                   "blas_level": "unknown", "data_type": "unknown",
                                   "description": "", "line_count": 10}},
        }

        client = TestClient(app)
        with patch("app.api.routes.search_codebase") as mock_search, \
             patch("app.api.routes.generate_answer_stream") as mock_gen:
            mock_search.return_value = ([fake_result], 50.0)
            mock_gen.return_value = iter(["Hello", " world"])
            response = client.post(
                "/api/query?stream=true",
                json={"query": "test"},
            )
            body = response.text
            # Must contain all three event types in order
            assert "event: sources" in body
            assert "event: token" in body
            assert "event: done" in body
            # Done must come after the last token
            last_token_pos = body.rfind("event: token")
            done_pos = body.rfind("event: done")
            assert done_pos > last_token_pos
