"""Tests for /api/chat/stream SSE endpoint — TDD RED phase."""

import json
import pytest
from unittest.mock import patch


def parse_sse_events(content: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    for block in content.strip().split("\n\n"):
        event_type = None
        data_str = None
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
        if event_type and data_str:
            events.append({"event": event_type, "data": json.loads(data_str)})
    return events


@pytest.fixture
def mock_agent_stream():
    """Mock run_agent_stream to yield controlled SSE events."""
    with patch("app.api.routes.run_agent_stream") as mock:

        def fake_stream(query, messages):
            yield {
                "event": "tool_start",
                "data": {
                    "tool_name": "search_codebase",
                    "tool_input": {"query": query},
                },
            }
            yield {
                "event": "tool_result",
                "data": {"tool_name": "search_codebase", "status": "ok"},
            }
            yield {"event": "text_delta", "data": {"chunk": "DGEMM performs "}}
            yield {"event": "text_delta", "data": {"chunk": "matrix multiply."}}
            yield {"event": "sources", "data": []}

        mock.side_effect = fake_stream
        yield mock


class TestStreamEndpoint:
    def test_returns_200_event_stream(self, test_client, mock_agent_stream):
        resp = test_client.post(
            "/api/chat/stream", json={"query": "What is DGEMM?"}
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_first_event_is_session(self, test_client, mock_agent_stream):
        resp = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events = parse_sse_events(resp.text)
        assert events[0]["event"] == "session"
        assert "session_id" in events[0]["data"]
        assert len(events[0]["data"]["session_id"]) > 0

    def test_contains_text_delta_events(self, test_client, mock_agent_stream):
        resp = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events = parse_sse_events(resp.text)
        deltas = [e for e in events if e["event"] == "text_delta"]
        assert len(deltas) == 2
        assert deltas[0]["data"]["chunk"] == "DGEMM performs "

    def test_contains_tool_events(self, test_client, mock_agent_stream):
        resp = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events = parse_sse_events(resp.text)
        tool_starts = [e for e in events if e["event"] == "tool_start"]
        tool_results = [e for e in events if e["event"] == "tool_result"]
        assert len(tool_starts) == 1
        assert len(tool_results) == 1
        assert tool_starts[0]["data"]["tool_name"] == "search_codebase"
        assert tool_results[0]["data"]["status"] == "ok"

    def test_ends_with_done_event(self, test_client, mock_agent_stream):
        resp = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events = parse_sse_events(resp.text)
        assert events[-1]["event"] == "done"
        assert "query_time_ms" in events[-1]["data"]
        assert events[-1]["data"]["query_time_ms"] > 0

    def test_reuses_session_with_id(self, test_client, mock_agent_stream):
        resp1 = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events1 = parse_sse_events(resp1.text)
        sid = events1[0]["data"]["session_id"]

        resp2 = test_client.post(
            "/api/chat/stream",
            json={"query": "Follow up", "session_id": sid},
        )
        events2 = parse_sse_events(resp2.text)
        assert events2[0]["data"]["session_id"] == sid

    def test_unknown_session_creates_new(self, test_client, mock_agent_stream):
        resp = test_client.post(
            "/api/chat/stream",
            json={"query": "Hello", "session_id": "fake-id"},
        )
        events = parse_sse_events(resp.text)
        assert events[0]["data"]["session_id"] != "fake-id"

    def test_empty_query_returns_422(self, test_client):
        resp = test_client.post("/api/chat/stream", json={})
        assert resp.status_code == 422

    def test_event_order_is_correct(self, test_client, mock_agent_stream):
        """Events arrive in order: session → tools → text → sources → done."""
        resp = test_client.post("/api/chat/stream", json={"query": "Hello"})
        events = parse_sse_events(resp.text)
        event_types = [e["event"] for e in events]
        assert event_types == [
            "session",
            "tool_start",
            "tool_result",
            "text_delta",
            "text_delta",
            "sources",
            "done",
        ]


class TestStreamEndpointIsAsync:
    def test_chat_stream_is_async(self):
        import inspect
        from app.api.routes import chat_stream

        assert inspect.iscoroutinefunction(chat_stream)
