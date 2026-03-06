"""Lightweight in-memory session store for conversation memory.

Stores prior Q&A turns so the LLM can reference earlier questions.
Thread-safe, with TTL-based expiry and max session caps.
"""

import time
import uuid
import threading
from collections import OrderedDict

from app.config import settings


class SessionStore:
    def __init__(
        self,
        max_sessions: int = None,
        ttl: int = None,
        max_messages: int = None,
    ):
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions if max_sessions is not None else settings.max_sessions
        self._ttl = ttl if ttl is not None else settings.session_ttl
        self._max_messages = max_messages if max_messages is not None else settings.max_messages_per_session

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "messages": [],
                "created_at": time.time(),
                "last_accessed": time.time(),
            }
            self._evict_expired()
        return session_id

    def get_messages(self, session_id: str) -> list[dict] | None:
        """Get a copy of messages for a session.

        Returns None if session not found or expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session["last_accessed"] > self._ttl:
                del self._sessions[session_id]
                return None
            session["last_accessed"] = time.time()
            self._sessions.move_to_end(session_id)
            return list(session["messages"])

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        """Add a Q&A turn to a session.

        Silently ignores unknown session IDs.
        Truncates to max_messages (oldest first).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session["messages"].append({"role": "user", "content": query})
            session["messages"].append({"role": "assistant", "content": answer})
            # Keep only the most recent messages
            session["messages"] = session["messages"][-self._max_messages:]
            session["last_accessed"] = time.time()

    def _evict_expired(self) -> None:
        """Remove expired sessions and enforce max size."""
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s["last_accessed"] > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)


# Module-level singleton
store = SessionStore()
