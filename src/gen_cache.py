"""On-disk cache for auto-generated, per-material content (starter questions, quiz).

Generating these calls the LLM, so we cache the result per (tenant, kind,
content_id, source_file) and reuse it. Cache lives under data/gen_cache/ as one
JSON file per entry. Delete a file to force regeneration.

`tenant_id` ikut masuk ke kunci hash. Tanpa itu, dua kampus yang memakai pola
`content_id` yang sama — dan pola `<matkul>-minggu-<n>` memang dianjurkan,
sehingga tabrakan justru lumrah — akan berbagi satu entri cache: kuis yang
disusun dari materi kampus A tersaji kepada mahasiswa kampus B. Kebocoran seperti
ini tidak akan tertangkap oleh penyaringan di lapisan pencarian, karena isinya
tidak pernah melewati Qdrant lagi setelah tersimpan di sini.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.tenancy import require_tenant_id
from src.utils.logger import logger

_CACHE_DIR = PROJECT_ROOT / "data" / "gen_cache"


def _cache_path(kind: str, content_id: str, source_file: str, tenant_id: str) -> Path:
    mentah = f"{tenant_id}\x00{kind}\x00{content_id}\x00{source_file}"
    key = hashlib.sha256(mentah.encode()).hexdigest()[:24]
    return _CACHE_DIR / f"{kind}_{key}.json"


def load(
    kind: str, content_id: str, source_file: str, *, tenant_id: str,
) -> Any | None:
    """Return cached data for this kind/material, or None if not cached yet."""
    tenant_id = require_tenant_id(tenant_id, operation="gen_cache.load")
    path = _cache_path(kind, content_id, source_file, tenant_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Pemeriksaan ulang pemilik: kunci hash sudah memisahkan tenant, tetapi
        # entri yang tersalin antar lingkungan tidak boleh terpakai diam-diam.
        if isinstance(data, dict) and data.get("tenant_id") not in (None, tenant_id):
            return None
        if isinstance(data, dict) and "data" in data:
            return data["data"]
    except Exception as exc:
        logger.warning("Failed to read gen cache {}: {}", path.name, exc)
    return None


def save(
    kind: str, content_id: str, source_file: str, data: Any, *, tenant_id: str,
) -> None:
    tenant_id = require_tenant_id(tenant_id, operation="gen_cache.save")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(kind, content_id, source_file, tenant_id)
    payload = {
        "tenant_id": tenant_id,
        "kind": kind,
        "content_id": content_id,
        "source_file": source_file,
        "data": data,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write gen cache {}: {}", path.name, exc)
