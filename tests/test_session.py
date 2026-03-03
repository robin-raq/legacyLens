"""Tests for in-memory session store — TDD RED phase."""

import time
from unittest.mock import patch


class TestSessionStore:
    def test_create_session_returns_string_id(self):
        from app.agent.session import SessionStore

        store = SessionStore()
        sid = store.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_get_messages_returns_empty_list_for_new_session(self):
        from app.agent.session import SessionStore

        store = SessionStore()
        sid = store.create_session()
        msgs = store.get_messages(sid)
        assert msgs == []

    def test_get_messages_unknown_session_returns_none(self):
        from app.agent.session import SessionStore

        store = SessionStore()
        assert store.get_messages("nonexistent-id") is None

    def test_set_messages_persists(self):
        from app.agent.session import SessionStore

        store = SessionStore()
        sid = store.create_session()
        store.set_messages(sid, [{"role": "user", "content": "hello"}])
        msgs = store.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_get_messages_returns_copy(self):
        from app.agent.session import SessionStore

        store = SessionStore()
        sid = store.create_session()
        store.set_messages(sid, [{"role": "user", "content": "hi"}])
        msgs = store.get_messages(sid)
        msgs.append({"role": "assistant", "content": "bye"})
        # Original should be unchanged
        assert len(store.get_messages(sid)) == 1

    def test_ttl_eviction(self):
        from app.agent.session import SessionStore

        store = SessionStore(ttl=1)  # 1 second TTL
        sid = store.create_session()
        store.set_messages(sid, [{"role": "user", "content": "test"}])

        # Should exist immediately
        assert store.get_messages(sid) is not None

        # After TTL expires
        time.sleep(1.1)
        assert store.get_messages(sid) is None

    def test_max_sessions_evicts_oldest(self):
        from app.agent.session import SessionStore

        store = SessionStore(max_sessions=3)
        sid1 = store.create_session()
        sid2 = store.create_session()
        sid3 = store.create_session()

        # All three exist
        assert store.get_messages(sid1) is not None
        assert store.get_messages(sid2) is not None
        assert store.get_messages(sid3) is not None

        # Adding a 4th should evict the oldest (sid1)
        sid4 = store.create_session()
        assert store.get_messages(sid1) is None
        assert store.get_messages(sid4) is not None

    def test_max_messages_truncates(self):
        from app.agent.session import SessionStore

        store = SessionStore(max_messages=5)
        sid = store.create_session()
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(10)]
        store.set_messages(sid, msgs)
        result = store.get_messages(sid)
        assert len(result) == 5
        # Should keep the LAST 5 (most recent)
        assert result[0]["content"] == "msg-5"

    def test_get_session_updates_last_accessed(self):
        from app.agent.session import SessionStore

        store = SessionStore(ttl=2)
        sid = store.create_session()

        # Access within TTL keeps it alive
        time.sleep(0.5)
        assert store.get_messages(sid) is not None
        time.sleep(0.5)
        assert store.get_messages(sid) is not None
        # Still alive because each access resets the clock
        time.sleep(0.5)
        assert store.get_messages(sid) is not None
