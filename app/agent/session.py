"""In-memory session store for multi-turn conversations."""

import time
import uuid
import threading
from collections import OrderedDict

SESSION_TTL_SECONDS = 3600  # 1 hour
MAX_SESSIONS = 200
MAX_MESSAGES_PER_SESSION = 50


class SessionStore:
    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        ttl: int = SESSION_TTL_SECONDS,
        max_messages: int = MAX_MESSAGES_PER_SESSION,
    ):
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._max_messages = max_messages

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
        """Get a copy of messages for a session. Returns None if not found or expired."""
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

    def set_messages(self, session_id: str, messages: list[dict]) -> None:
        """Update messages for a session, truncating to max_messages."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session["messages"] = messages[-self._max_messages :]

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
