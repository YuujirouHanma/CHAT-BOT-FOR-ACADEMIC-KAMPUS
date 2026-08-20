"""Logs chat interactions and student feedback to local JSONL files.

Not a training pipeline — just durable records that a dosen/asisten can
later review to validate answers and build fine-tuning datasets offline.

Dipisah per tenant: `data/hitl_logs/{tenant_id}/`. Berkas ini memuat pertanyaan
mahasiswa beserta jawaban lengkapnya — isi paling sensitif di seluruh sistem,
karena merekam apa yang sedang dipelajari seseorang dan materi internal kampus
mana yang dipakai menjawabnya. Menaruhnya dalam satu berkas bersama berarti
siapa pun yang boleh membaca log satu kampus dapat membaca log kampus lain.

Identitas mahasiswa disunting lebih dulu lewat `security.redact`, sehingga baris
log tetap dapat ditelusuri dan dibandingkan tanpa memuat identitas utuhnya.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.security import redact
from src.tenancy import require_tenant_id, storage_prefix

HITL_DIR = PROJECT_ROOT / "data" / "hitl_logs"

CONVERSATION_LOG = "conversation_logs.jsonl"
FEEDBACK_LOG = "student_feedback_logs.jsonl"
QUIZ_ATTEMPT_LOG = "quiz_attempts.jsonl"


def log_dir(tenant_id: str) -> Path:
    return HITL_DIR / storage_prefix(tenant_id)


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
    *,
    tenant_id: str,
) -> str:
    """Log one chat interaction. Returns the interaction_id for feedback linking."""
    tenant_id = require_tenant_id(tenant_id, operation="log_interaction")
    interaction_id = f"hitl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tenant_id": tenant_id,
        "session_id": session_id,
        "content_id": content_id,
        "question": question,
        "decomposition": dq,
        "answer": answer,
        "sources": sources,
        "recommendations": recommendations,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    _append_jsonl(log_dir(tenant_id) / CONVERSATION_LOG, redact.scrub(record))
    return interaction_id


def log_quiz_attempt(
    content_id: str,
    source_file: str,
    correct: int,
    total: int,
    score: float,
    session_id: str | None = None,
    student_id: str | None = None,
    *,
    tenant_id: str,
) -> str:
    """Log one quiz attempt (for progress tracking). Returns the attempt_id."""
    tenant_id = require_tenant_id(tenant_id, operation="log_quiz_attempt")
    attempt_id = f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = {
        "attempt_id": attempt_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tenant_id": tenant_id,
        "session_id": session_id,
        "student_id": student_id,
        "content_id": content_id,
        "source_file": source_file,
        "correct": correct,
        "total": total,
        "score": score,
    }
    _append_jsonl(log_dir(tenant_id) / QUIZ_ATTEMPT_LOG, redact.scrub(record))
    return attempt_id


def log_feedback(
    interaction_id: str,
    rating: str,
    issues: list[str] | None = None,
    comment: str | None = None,
    *,
    tenant_id: str,
) -> None:
    """Log student feedback for a previously logged interaction."""
    tenant_id = require_tenant_id(tenant_id, operation="log_feedback")
    record = {
        "feedback_id": f"fb_{uuid.uuid4().hex[:8]}",
        "interaction_id": interaction_id,
        "tenant_id": tenant_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rating": rating,
        "issues": issues or [],
        "comment": comment or "",
    }
    _append_jsonl(log_dir(tenant_id) / FEEDBACK_LOG, redact.scrub(record))
