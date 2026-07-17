"""Logs chat interactions and student feedback to local JSONL files.

Not a training pipeline — just durable records that a dosen/asisten can
later review to validate answers and build fine-tuning datasets offline.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

HITL_DIR = PROJECT_ROOT / "data" / "hitl_logs"
CONVERSATION_LOG_PATH = HITL_DIR / "conversation_logs.jsonl"
FEEDBACK_LOG_PATH = HITL_DIR / "student_feedback_logs.jsonl"
QUIZ_ATTEMPT_LOG_PATH = HITL_DIR / "quiz_attempts.jsonl"


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")


def log_interaction(
    question: str,
    dq: dict[str, Any],
    answer: str,
    sources: list[dict],
    recommendations: list[str],
    elapsed_seconds: float,
    content_id: str | None = None,
    session_id: str | None = None,
) -> str:
    """Log one chat interaction. Returns the interaction_id for feedback linking."""
    interaction_id = f"hitl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "content_id": content_id,
        "question": question,
        "decomposition": dq,
        "answer": answer,
        "sources": sources,
        "recommendations": recommendations,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    _append_jsonl(CONVERSATION_LOG_PATH, record)
    return interaction_id


def log_quiz_attempt(
    content_id: str,
    source_file: str,
    correct: int,
    total: int,
    score: float,
    session_id: str | None = None,
    student_id: str | None = None,
) -> str:
    """Log one quiz attempt (for progress tracking). Returns the attempt_id."""
    attempt_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = {
        "attempt_id": attempt_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "student_id": student_id,
        "content_id": content_id,
        "source_file": source_file,
        "correct": correct,
        "total": total,
        "score": score,
    }
    _append_jsonl(QUIZ_ATTEMPT_LOG_PATH, record)
    return attempt_id


def log_feedback(
    interaction_id: str,
    rating: str,
    issues: list[str] | None = None,
    comment: str | None = None,
) -> None:
    """Log student feedback for a previously logged interaction."""
    record = {
        "feedback_id": f"fb_{uuid.uuid4().hex[:8]}",
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "issues": issues or [],
        "comment": comment or "",
    }
    _append_jsonl(FEEDBACK_LOG_PATH, record)
