"""Jejak audit yang perubahannya dapat terdeteksi (*tamper-evident*).

Setiap peristiwa penting — autentikasi, unggah materi, penghapusan percakapan,
penolakan akses — dicatat sebagai satu baris JSON, dan setiap baris memuat MAC
dari isinya digabung MAC baris sebelumnya. Rantai itulah yang membuat perubahan
terdeteksi: mengubah atau menghapus satu baris memutus semua MAC sesudahnya, dan
memalsukannya kembali mustahil tanpa *pepper* sisi server.

Rantai memakai HMAC-SHA256, bukan SHA-256 biasa. Dengan hash biasa, siapa pun
yang bisa menulis berkasnya juga bisa menghitung ulang seluruh rantai setelah
mengubah isinya — jejaknya tampak utuh padahal sudah dipalsukan.

BATASAN JUJUR: ini membuat perubahan TERDETEKSI, bukan MUSTAHIL. Penyerang yang
menguasai server tetap bisa menghapus berkasnya. Untuk kepatuhan sungguhan,
alirkan juga ke penyimpanan yang hanya bisa ditulis sekali (S3 Object Lock,
CloudWatch Logs, atau SIEM) — lihat docs/SECURITY.md.

Isi peristiwa selalu melewati `redact.scrub()` lebih dulu, sehingga kunci API dan
identitas mahasiswa tidak pernah masuk ke jejak audit dalam bentuk utuh.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from src.config import PROJECT_ROOT, settings
from src.security import redact
from src.utils.logger import logger

AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
# Peristiwa yang belum punya tenant (mis. autentikasi gagal) tetap harus tercatat.
SYSTEM_SCOPE: Final = "_system"

GENESIS: Final = "0" * 64

# --- jenis peristiwa ---
AUTH_SUCCESS: Final = "auth.success"
AUTH_FAILURE: Final = "auth.failure"
AUTH_FORBIDDEN: Final = "auth.forbidden"
TENANT_MISMATCH: Final = "auth.tenant_mismatch"
RATE_LIMITED: Final = "auth.rate_limited"
CONTENT_UPLOAD: Final = "content.upload"
CONTENT_INDEX: Final = "content.index"
CONVERSATION_READ: Final = "conversation.read"
CONVERSATION_DELETE: Final = "conversation.delete"
CHAT_ASK: Final = "chat.ask"

# Berkas terakhir per cakupan → (path, hash terakhir). Menghindari membaca ulang
# seluruh berkas hanya untuk mengetahui mata rantai sebelumnya.
_tail_cache: dict[str, tuple[Path, str]] = {}


def _mac_key() -> bytes:
    return settings.tenant_key_pepper.get_secret_value().encode("utf-8") or b"dev-audit"


def _canonical(record: dict[str, Any]) -> bytes:
    """Bentuk baku sebuah catatan untuk dihitung MAC-nya.

    `sort_keys` wajib: tanpa urutan yang pasti, catatan yang sama bisa
    menghasilkan MAC berbeda dan verifikasi rantai gagal tanpa ada yang diubah.
    """
    return json.dumps(record, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _chain_mac(record: dict[str, Any], prev_hash: str) -> str:
    body = _canonical(record) + b"|" + prev_hash.encode("ascii")
    return hmac.new(_mac_key(), body, hashlib.sha256).hexdigest()


def _path_for(scope: str) -> Path:
    hari = datetime.now(UTC).strftime("%Y-%m-%d")
    return AUDIT_DIR / scope / f"{hari}.jsonl"


def _last_hash(path: Path) -> str:
    """MAC baris terakhir pada berkas; GENESIS bila berkasnya baru."""
    if not path.exists():
        return GENESIS
    try:
        terakhir = ""
        with path.open("r", encoding="utf-8") as f:
            for baris in f:
                if baris.strip():
                    terakhir = baris
        if not terakhir:
            return GENESIS
        return json.loads(terakhir).get("hash") or GENESIS
    except Exception as exc:
        logger.error("Jejak audit {} tidak terbaca: {}", path.name, exc)
        return GENESIS


def record(
    event: str,
    *,
    tenant_id: str | None = None,
    actor: str = "",
    outcome: str = "ok",
    request_id: str = "",
    ip: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Catat satu peristiwa audit.

    Kegagalan menulis TIDAK dilempar: sebuah unggahan yang sudah berhasil tidak
    boleh dibatalkan hanya karena pencatatannya gagal. Kegagalan itu sendiri
    dicatat sebagai galat agar tidak lolos tanpa terlihat.
    """
    scope = tenant_id or SYSTEM_SCOPE
    path = _path_for(scope)

    cached = _tail_cache.get(scope)
    prev = cached[1] if cached and cached[0] == path else _last_hash(path)

    catatan: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": event,
        "tenant_id": tenant_id,
        "actor": actor,               # key_id, bukan kunci itu sendiri
        "outcome": outcome,
        "request_id": request_id,
        "ip": ip,
        "detail": redact.scrub(detail or {}),
        "prev_hash": prev,
    }
    catatan["hash"] = _chain_mac(catatan, prev)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(catatan, ensure_ascii=False) + "\n")
        _tail_cache[scope] = (path, catatan["hash"])
    except Exception as exc:
        logger.error("Gagal menulis jejak audit ({}): {}", event, exc)


def verify_chain(path: Path) -> tuple[bool, int, str]:
    """Periksa keutuhan satu berkas jejak audit.

    Mengembalikan (utuh, jumlah_baris, keterangan). Dipakai pemeriksaan berkala
    dan saat menyiapkan bukti kepatuhan.
    """
    if not path.exists():
        return False, 0, "berkas tidak ada"

    prev = GENESIS
    jumlah = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for nomor, baris in enumerate(f, start=1):
                if not baris.strip():
                    continue
                jumlah += 1
                catatan = json.loads(baris)
                tersimpan = catatan.pop("hash", "")
                if catatan.get("prev_hash") != prev:
                    return False, jumlah, f"rantai putus di baris {nomor}"
                if not hmac.compare_digest(_chain_mac(catatan, prev), tersimpan):
                    return False, jumlah, f"MAC tidak cocok di baris {nomor}"
                prev = tersimpan
    except Exception as exc:
        return False, jumlah, f"gagal membaca: {exc}"

    return True, jumlah, "utuh"
