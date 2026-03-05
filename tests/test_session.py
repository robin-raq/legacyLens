"""Tests for lightweight conversation memory (session store)."""

import time
import pytest
from app.session import SessionStore


@pytest.fixture
def store():
    """Fresh session store for each test."""
    return SessionStore(max_sessions=5, ttl=10, max_messages=4)


class TestSessionCreate:
    def test_create_returns_uuid(self, store):
        sid = store.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID format

    def test_create_multiple_unique(self, store):
        ids = {store.create_session() for _ in range(3)}
        assert len(ids) == 3


class TestSessionMessages:
    def test_new_session_empty(self, store):
        sid = store.create_session()
        assert store.get_messages(sid) == []

    def test_add_turn(self, store):
        sid = store.create_session()
        store.add_turn(sid, "What does DGEMM do?", "DGEMM performs matrix multiplication.")
        msgs = store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What does DGEMM do?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "DGEMM performs matrix multiplication."

    def test_multiple_turns(self, store):
        sid = store.create_session()
        store.add_turn(sid, "Q1", "A1")
        store.add_turn(sid, "Q2", "A2")
        msgs = store.get_messages(sid)
        assert len(msgs) == 4
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "Q2"

    def test_unknown_session_returns_none(self, store):
        assert store.get_messages("nonexistent-id") is None

    def test_add_turn_unknown_session_ignored(self, store):
        """Adding to a nonexistent session should not crash."""
        store.add_turn("nonexistent-id", "Q", "A")  # Should not raise


class TestSessionLimits:
    def test_max_messages_truncates(self, store):
        """Messages beyond max_messages are dropped (oldest first)."""
        sid = store.create_session()
        # max_messages=4, so 3 turns = 6 messages, should keep last 4
        store.add_turn(sid, "Q1", "A1")
        store.add_turn(sid, "Q2", "A2")
        store.add_turn(sid, "Q3", "A3")
        msgs = store.get_messages(sid)
        assert len(msgs) == 4
        # Oldest messages (Q1/A1) should be dropped, keeping [Q2, A2, Q3, A3]
        assert msgs[0]["content"] == "Q2"
        assert msgs[3]["content"] == "A3"

    def test_max_sessions_evicts_oldest(self, store):
        """Creating more sessions than max evicts the oldest."""
        sids = [store.create_session() for _ in range(6)]  # max=5
        # First session should be evicted
        assert store.get_messages(sids[0]) is None
        # Latest should still exist
        assert store.get_messages(sids[5]) == []


class TestSessionExpiry:
    def test_expired_session_returns_none(self):
        store = SessionStore(ttl=1)  # 1 second TTL
        sid = store.create_session()
        store.add_turn(sid, "Q", "A")
        time.sleep(1.1)
        assert store.get_messages(sid) is None
