"""On-disk cache for auto-generated starter questions.

Generating starter questions calls the LLM, so we cache the result per
(content_id, source_file) and reuse it. Cache lives under data/starter_questions/
as one JSON file per material. Delete a file to force regeneration.
"""
from __future__ import annotations

import hashlib
import json

from src.config import PROJECT_ROOT
from src.utils.logger import logger

_CACHE_DIR = PROJECT_ROOT / "data" / "starter_questions"


def _cache_path(content_id: str, source_file: str):
    key = hashlib.sha1(f"{content_id}\x00{source_file}".encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{key}.json"


def load(content_id: str, source_file: str) -> list[str] | None:
    """Return cached questions, or None if not cached yet."""
    path = _cache_path(content_id, source_file)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            return [str(q) for q in data["questions"]]
    except Exception as exc:
        logger.warning("Failed to read starter cache {}: {}", path.name, exc)
    return None


def save(content_id: str, source_file: str, questions: list[str]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(content_id, source_file)
    payload = {
        "content_id": content_id,
        "source_file": source_file,
        "questions": questions,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write starter cache {}: {}", path.name, exc)
