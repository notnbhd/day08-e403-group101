from __future__ import annotations

import threading
import time
import uuid
from typing import Any


class ConversationStore:
    """
    In-memory conversation store for local demo.

    This is enough for a classroom/local FastAPI demo.
    For production, replace this with Redis/Postgres.
    """

    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def ensure_session(self, session_id: str | None = None) -> str:
        with self._lock:
            sid = session_id or str(uuid.uuid4())
            self._sessions.setdefault(sid, [])
            return sid

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            summaries: list[dict[str, Any]] = []
            for session_id, history in self._sessions.items():
                first_user = next((turn.get("user") for turn in history if turn.get("user")), "")
                last_turn = history[-1] if history else {}
                summaries.append(
                    {
                        "session_id": session_id,
                        "title": (first_user or "Hội thoại mới")[:80],
                        "turn_count": len(history),
                        "updated_at": float(last_turn.get("created_at") or 0),
                    }
                )

            summaries.sort(key=lambda item: item["updated_at"], reverse=True)
            return summaries

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_answer: str,
        standalone_question: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        turn = {
            "user": user_message,
            "assistant": assistant_answer,
            "standalone_question": standalone_question,
            "sources": sources or [],
            "created_at": time.time(),
        }

        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(turn)
            if len(history) > self.max_turns:
                self._sessions[session_id] = history[-self.max_turns :]

    def reset(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id:
                self._sessions.pop(session_id, None)
            else:
                self._sessions.clear()


CONVERSATIONS = ConversationStore()
