"""On-disk cache for auto-generated, per-material content (starter questions, quiz).

Generating these calls the LLM, so we cache the result per (kind, content_id,
source_file) and reuse it. Cache lives under data/gen_cache/ as one JSON file
per entry. Delete a file to force regeneration.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config import PROJECT_ROOT
from src.utils.logger import logger

_CACHE_DIR = PROJECT_ROOT / "data" / "gen_cache"


def _cache_path(kind: str, content_id: str, source_file: str):
    key = hashlib.sha1(f"{kind}\x00{content_id}\x00{source_file}".encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{kind}_{key}.json"


def load(kind: str, content_id: str, source_file: str) -> Any | None:
    """Return cached data for this kind/material, or None if not cached yet."""
    path = _cache_path(kind, content_id, source_file)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            return data["data"]
    except Exception as exc:
        logger.warning("Failed to read gen cache {}: {}", path.name, exc)
    return None


def save(kind: str, content_id: str, source_file: str, data: Any) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(kind, content_id, source_file)
    payload = {
        "kind": kind,
        "content_id": content_id,
        "source_file": source_file,
        "data": data,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write gen cache {}: {}", path.name, exc)
