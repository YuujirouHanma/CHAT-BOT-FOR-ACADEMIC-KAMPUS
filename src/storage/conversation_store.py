"""Penyimpanan percakapan agar dapat dibuka kembali di lain hari.

Sebelumnya percakapan hanya hidup di memori dan hangus setelah satu jam, sehingga
mahasiswa tidak bisa melanjutkan belajar keesokan harinya. Di sini setiap
percakapan disimpan sebagai satu berkas JSON, dinamai `session_id`-nya.

Satu berkas per percakapan — bukan satu berkas besar berisi semuanya — supaya
menghapus satu percakapan cukup menghapus berkasnya, tanpa perlu menulis ulang
seluruh data dan tanpa risiko dua permintaan saling menimpa.

Dua daftar pesan disimpan terpisah dan sengaja berbeda isinya:

- `transcript` — SEMUA yang tampil di layar, termasuk langkah memilih menu.
  Dipakai untuk menampilkan ulang percakapan apa adanya.
- `history`    — hanya tanya-jawab sungguhan. Dipakai sebagai konteks LLM;
  memasukkan klik menu ke sini hanya memboroskan token dan mengaburkan alur
  belajar yang ingin disambung oleh pertanyaan lanjutan.

ISOLASI DAN DATA PRIBADI
------------------------
Berkas dipisah per tenant: `data/conversations/{tenant_id}/{id}.json`. Karena
`tenant_id` menjadi bagian dari path, membaca percakapan tenant lain tidak
tertahan oleh sebuah pemeriksaan yang bisa terlupa — path-nya memang tidak
menunjuk ke sana. Nilai `tenant_id` di dalam berkas diperiksa ulang sebagai
lapis kedua, untuk menangkap berkas yang salah tempat akibat pemulihan cadangan.

`student_id` adalah data pribadi: ia merangkai seluruh riwayat belajar satu
orang. Ia tidak disimpan apa adanya, melainkan sebagai dua turunan —
`student_ref` (indeks buta, untuk menyaring) dan `student_id` tersandi (untuk
ditampilkan kembali). Lihat `src/security/crypto.py`.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.security import crypto
from src.tenancy import require_tenant_id, storage_prefix
from src.utils.logger import logger

CONVERSATION_DIR = PROJECT_ROOT / "data" / "conversations"

# session_id kita adalah UUID; pola ini menolak apa pun di luar itu supaya
# nama berkas tidak bisa dipakai keluar dari direktori (path traversal).
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_TITLE_MAX = 60


def is_valid_id(conversation_id: str) -> bool:
    return bool(_ID_RE.match(conversation_id or ""))


def tenant_dir(tenant_id: str) -> Path:
    return CONVERSATION_DIR / storage_prefix(tenant_id)


def _path_for(conversation_id: str, tenant_id: str) -> Path | None:
    if not is_valid_id(conversation_id):
        logger.warning("Id percakapan ditolak: {!r}", conversation_id)
        return None
    return tenant_dir(tenant_id) / f"{conversation_id}.json"


def make_title(text: str) -> str:
    """Judul percakapan dari pesan pertama mahasiswa.

    Dipakai di daftar riwayat. Sapaan seperti "halo" tidak memberi tahu apa pun,
    jadi judul baru ditetapkan dari pesan yang benar-benar berisi.
    """
    bersih = " ".join((text or "").split())
    if len(bersih) <= _TITLE_MAX:
        return bersih or "Percakapan baru"
    return bersih[:_TITLE_MAX].rstrip() + "…"


def _protect_pii(record: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    """Ganti `student_id` mentah dengan bentuk tersandi + indeks butanya."""
    hasil = dict(record)
    mentah = hasil.get("student_id")
    hasil["student_ref"] = crypto.blind_index(mentah, tenant_id=tenant_id)
    hasil["student_id"] = crypto.encrypt_field(mentah, aad=tenant_id)
    return hasil


def _reveal_pii(record: dict[str, Any], tenant_id: str) -> dict[str, Any]:
    hasil = dict(record)
    hasil["student_id"] = crypto.decrypt_field(hasil.get("student_id"), aad=tenant_id)
    return hasil


def save(record: dict[str, Any], *, tenant_id: str) -> None:
    """Tulis satu percakapan. Kegagalan dicatat, tidak dilempar.

    Menyimpan riwayat tidak boleh menggagalkan jawaban yang sudah berhasil
    dihasilkan — dari sudut pandang mahasiswa, kehilangan riwayat jauh lebih
    ringan daripada kehilangan jawabannya.
    """
    tenant_id = require_tenant_id(tenant_id, operation="conversation.save")
    path = _path_for(record.get("conversation_id", ""), tenant_id)
    if path is None:
        return
    try:
        isi = _protect_pii(record, tenant_id)
        isi["tenant_id"] = tenant_id
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(isi, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        tmp.replace(path)   # atomik: berkas tidak pernah terbaca separuh tertulis
    except Exception as exc:
        logger.warning("Gagal menyimpan percakapan {}: {}", path.name, exc)


def load(conversation_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Baca satu percakapan milik tenant ini; None bila tidak ada atau rusak."""
    tenant_id = require_tenant_id(tenant_id, operation="conversation.load")
    path = _path_for(conversation_id, tenant_id)
    if path is None or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Percakapan {} tidak terbaca: {}", path.name, exc)
        return None

    # Lapis kedua: berkas yang tersalin ke direktori tenant lain (mis. akibat
    # pemulihan cadangan yang keliru) ditolak, bukan disajikan.
    pemilik = d.get("tenant_id")
    if pemilik is not None and pemilik != tenant_id:
        logger.error(
            "Percakapan {} berisi tenant_id '{}' di direktori '{}' — ditolak",
            conversation_id, pemilik, tenant_id,
        )
        return None
    return _reveal_pii(d, tenant_id)


def delete(conversation_id: str, *, tenant_id: str) -> bool:
    """Hapus satu percakapan. True bila berkasnya memang ada dan terhapus."""
    tenant_id = require_tenant_id(tenant_id, operation="conversation.delete")
    path = _path_for(conversation_id, tenant_id)
    if path is None or not path.exists():
        return False
    try:
        path.unlink()
        logger.info("Percakapan dihapus: {} (tenant={})", conversation_id, tenant_id)
        return True
    except Exception as exc:
        logger.warning("Gagal menghapus percakapan {}: {}", conversation_id, exc)
        return False


def list_summaries(
    student_id: str | None = None, limit: int = 50, *, tenant_id: str,
) -> list[dict[str, Any]]:
    """Ringkasan percakapan milik satu tenant, terbaru lebih dulu.

    Hanya bagian ringkasnya yang dikembalikan — daftar riwayat tidak perlu
    memuat seluruh isi pesan, dan mengirimkannya akan memberatkan klien.

    `student_id` menyaring milik siapa, dicocokkan lewat indeks buta sehingga
    penyaringan berjalan tanpa perlu mendekripsi satu berkas pun. Percakapan
    tanpa pemilik (mahasiswa belum dikenali) hanya muncul bila penyaring kosong.
    """
    tenant_id = require_tenant_id(tenant_id, operation="conversation.list")
    root = tenant_dir(tenant_id)
    if not root.exists():
        return []

    rujukan = crypto.blind_index(student_id, tenant_id=tenant_id) if student_id else None

    hasil: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue                      # berkas rusak dilewati, bukan menggagalkan daftar
        if d.get("tenant_id") not in (None, tenant_id):
            continue
        if rujukan is not None and d.get("student_ref") != rujukan:
            continue
        hasil.append({
            "conversation_id": d.get("conversation_id", path.stem),
            "title": d.get("title") or "Percakapan baru",
            "student_id": crypto.decrypt_field(d.get("student_id"), aad=tenant_id),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
            "message_count": len(d.get("transcript") or []),
            "context": d.get("context") or {},
        })

    hasil.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return hasil[:limit]


def delete_all_for_tenant(*, tenant_id: str) -> int:
    """Hapus seluruh percakapan satu tenant. Dipakai saat berhenti berlangganan."""
    tenant_id = require_tenant_id(tenant_id, operation="conversation.purge")
    root = tenant_dir(tenant_id)
    if not root.exists():
        return 0
    jumlah = 0
    for path in root.glob("*.json"):
        try:
            path.unlink()
            jumlah += 1
        except OSError as exc:
            logger.warning("Gagal menghapus {}: {}", path.name, exc)
    logger.warning("{} percakapan tenant '{}' dihapus", jumlah, tenant_id)
    return jumlah


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
