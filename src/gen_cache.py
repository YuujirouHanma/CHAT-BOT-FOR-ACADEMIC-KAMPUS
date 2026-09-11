"""On-disk cache for auto-generated, per-material content (starter questions, quiz).

Generating these calls the LLM, so we cache the result per (tenant, kind,
content_id, source_file, material_hash) and reuse it. Cache lives under
data/gen_cache/ as one JSON file per entry. Delete a file to force regeneration.

`material_hash` adalah sidik jari teks materi yang benar-benar dipakai saat
membangkitkan. Tanpa itu, kunci cache hanya menyebut nama berkas — dan setiap
perbaikan pada cara materi dirakit (urutan potongan, penanda sumber, kuota
karakter per minggu) tidak akan pernah terasa pada materi yang terlanjur
ter-cache. Entri lama tidak dihapus, hanya berhenti terpanggil.

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


def hash_material(text: str) -> str:
    """Sidik jari isi materi yang dipakai membangkitkan sebuah entri cache.

    Dipakai sebagai bagian kunci cache supaya entri lama berhenti terpakai
    begitu materinya berubah — termasuk ketika yang berubah adalah *cara*
    materi itu dirakit, bukan berkasnya. Perbaikan pada perakit teks (urutan
    potongan, penanda sumber, kuota per minggu) tidak akan terasa sama sekali
    selama kuncinya hanya menyebut nama berkas.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_path(
    kind: str, content_id: str, source_file: str, tenant_id: str,
    material_hash: str,
) -> Path:
    mentah = (
        f"{tenant_id}\x00{kind}\x00{content_id}\x00{source_file}\x00{material_hash}"
    )
    key = hashlib.sha256(mentah.encode()).hexdigest()[:24]
    return _CACHE_DIR / f"{kind}_{key}.json"


def load(
    kind: str, content_id: str, source_file: str, *, tenant_id: str,
    material_hash: str,
) -> Any | None:
    """Kembalikan entri cache untuk materi ini, atau None bila belum ada.

    `material_hash` wajib: pemanggil harus sudah memegang teks materinya. Itu
    berarti teks diambil lebih dulu meski nanti berujung cache hit — biayanya
    satu scroll Qdrant (0,001-0,014 detik), jauh lebih murah daripada
    menyajikan soal yang disusun dari materi versi lama.
    """
    tenant_id = require_tenant_id(tenant_id, operation="gen_cache.load")
    path = _cache_path(kind, content_id, source_file, tenant_id, material_hash)
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
    material_hash: str,
) -> None:
    tenant_id = require_tenant_id(tenant_id, operation="gen_cache.save")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(kind, content_id, source_file, tenant_id, material_hash)
    payload = {
        "tenant_id": tenant_id,
        "kind": kind,
        "content_id": content_id,
        "source_file": source_file,
        # Disimpan apa adanya supaya entri lama dapat ditelusuri asalnya ketika
        # kuncinya berubah dan berkasnya menjadi yatim.
        "material_hash": material_hash,
        "data": data,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write gen cache {}: {}", path.name, exc)
