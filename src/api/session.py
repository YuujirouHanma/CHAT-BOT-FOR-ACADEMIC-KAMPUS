"""In-memory session manager for chat context."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

TTL_SECONDS = 3600


@dataclass
class Session:
    session_id: str
    course: str | None = None
    week: int | None = None
    source_filter: str | None = None
    history: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_active) > TTL_SECONDS

    def set_context(
        self, course: str | None = None, week: int | None = None,
        source_filter: str | None = None,
    ) -> None:
        if course != self.course or week != self.week:
            self.source_filter = None
        if course is not None:
            self.course = course
        if week is not None:
            self.week = week
        if source_filter is not None:
            self.source_filter = source_filter
        self.touch()

    def add_turn(self, role: str, content: str, max_turns: int = 10) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]
        self.touch()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        s = Session(session_id=str(uuid.uuid4()))
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        if s.is_expired():
            del self._sessions[session_id]
            return None
        s.touch()
        return s

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            s = self.get(session_id)
            if s:
                return s
        return self.create()


session_store = SessionStore()