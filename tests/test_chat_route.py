"""Tests for /api/chat endpoint — TDD RED phase."""

import pytest
import inspect
from unittest.mock import patch, MagicMock
from app.models import SearchResult, CodeChunk, ChunkMetadata


@pytest.fixture
def mock_agent():
    """Mock the run_agent function."""
    with patch("app.api.routes.run_agent") as mock:
        mock.return_value = (
            "DGEMM performs matrix multiplication.",
            [
                SearchResult(
                    chunk=CodeChunk(
                        id="dgemm.f::DGEMM",
                        text="SUBROUTINE DGEMM",
                        metadata=ChunkMetadata(
                            file_path="SRC/dgemm.f",
                            start_line=1,
                            end_line=50,
                            subroutine_name="DGEMM",
                            blas_level="3",
                            data_type="double real",
                        ),
                    ),
                    score=0.85,
                )
            ],
            [
                {
                    "tool_name": "search_codebase",
                    "tool_input": {"query": "DGEMM"},
                    "tool_result": {"results": []},
                }
            ],
        )
        yield mock


class TestChatEndpoint:
    def test_returns_200(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        assert resp.status_code == 200

    def test_returns_session_id(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_returns_answer(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        data = resp.json()
        assert "answer" in data
        assert "DGEMM" in data["answer"]

    def test_returns_sources(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        data = resp.json()
        assert "sources" in data
        assert len(data["sources"]) == 1

    def test_returns_tool_calls(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        data = resp.json()
        assert "tool_calls" in data
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool_name"] == "search_codebase"

    def test_returns_query_time(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "What is DGEMM?"})
        data = resp.json()
        assert "query_time_ms" in data
        assert data["query_time_ms"] > 0

    def test_new_session_created_without_session_id(self, test_client, mock_agent):
        resp = test_client.post("/api/chat", json={"query": "Hello"})
        sid1 = resp.json()["session_id"]

        resp2 = test_client.post("/api/chat", json={"query": "Hello"})
        sid2 = resp2.json()["session_id"]

        # Different sessions since no session_id was sent
        assert sid1 != sid2

    def test_reuses_session_with_id(self, test_client, mock_agent):
        resp1 = test_client.post("/api/chat", json={"query": "Hello"})
        sid = resp1.json()["session_id"]

        resp2 = test_client.post(
            "/api/chat", json={"query": "Follow up", "session_id": sid}
        )
        assert resp2.json()["session_id"] == sid

    def test_unknown_session_creates_new(self, test_client, mock_agent):
        resp = test_client.post(
            "/api/chat", json={"query": "Hello", "session_id": "fake-id"}
        )
        data = resp.json()
        assert data["session_id"] != "fake-id"

    def test_empty_query_returns_422(self, test_client):
        resp = test_client.post("/api/chat", json={})
        assert resp.status_code == 422


class TestChatRouteIsAsync:
    def test_chat_is_async(self):
        from app.api.routes import chat

        assert inspect.iscoroutinefunction(chat)


class TestOldEndpointUnchanged:
    def test_health_still_works(self, test_client):
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
