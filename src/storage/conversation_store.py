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
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.utils.logger import logger

CONVERSATION_DIR = PROJECT_ROOT / "data" / "conversations"

# session_id kita adalah UUID; pola ini menolak apa pun di luar itu supaya
# nama berkas tidak bisa dipakai keluar dari direktori (path traversal).
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_TITLE_MAX = 60


def is_valid_id(conversation_id: str) -> bool:
    return bool(_ID_RE.match(conversation_id or ""))


def _path_for(conversation_id: str) -> Path | None:
    if not is_valid_id(conversation_id):
        logger.warning("Id percakapan ditolak: {!r}", conversation_id)
        return None
    return CONVERSATION_DIR / f"{conversation_id}.json"


def make_title(text: str) -> str:
    """Judul percakapan dari pesan pertama mahasiswa.

    Dipakai di daftar riwayat. Sapaan seperti "halo" tidak memberi tahu apa pun,
    jadi judul baru ditetapkan dari pesan yang benar-benar berisi.
    """
    bersih = " ".join((text or "").split())
    if len(bersih) <= _TITLE_MAX:
        return bersih or "Percakapan baru"
    return bersih[:_TITLE_MAX].rstrip() + "…"


def save(record: dict[str, Any]) -> None:
    """Tulis satu percakapan. Kegagalan dicatat, tidak dilempar.

    Menyimpan riwayat tidak boleh menggagalkan jawaban yang sudah berhasil
    dihasilkan — dari sudut pandang mahasiswa, kehilangan riwayat jauh lebih
    ringan daripada kehilangan jawabannya.
    """
    path = _path_for(record.get("conversation_id", ""))
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        tmp.replace(path)   # atomik: berkas tidak pernah terbaca separuh tertulis
    except Exception as exc:
        logger.warning("Gagal menyimpan percakapan {}: {}", path.name, exc)


def load(conversation_id: str) -> dict[str, Any] | None:
    """Baca satu percakapan; None bila tidak ada atau rusak."""
    path = _path_for(conversation_id)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Percakapan {} tidak terbaca: {}", path.name, exc)
        return None


def delete(conversation_id: str) -> bool:
    """Hapus satu percakapan. True bila berkasnya memang ada dan terhapus."""
    path = _path_for(conversation_id)
    if path is None or not path.exists():
        return False
    try:
        path.unlink()
        logger.info("Percakapan dihapus: {}", conversation_id)
        return True
    except Exception as exc:
        logger.warning("Gagal menghapus percakapan {}: {}", conversation_id, exc)
        return False


def list_summaries(
    student_id: str | None = None, limit: int = 50,
) -> list[dict[str, Any]]:
    """Ringkasan percakapan, terbaru lebih dulu.

    Hanya bagian ringkasnya yang dikembalikan — daftar riwayat tidak perlu
    memuat seluruh isi pesan, dan mengirimkannya akan memberatkan klien.

    `student_id` menyaring milik siapa. Percakapan tanpa pemilik (mahasiswa
    belum dikenali) hanya muncul bila penyaring tidak diisi.
    """
    if not CONVERSATION_DIR.exists():
        return []

    hasil: list[dict[str, Any]] = []
    for path in CONVERSATION_DIR.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue                      # berkas rusak dilewati, bukan menggagalkan daftar
        if student_id is not None and d.get("student_id") != student_id:
            continue
        hasil.append({
            "conversation_id": d.get("conversation_id", path.stem),
            "title": d.get("title") or "Percakapan baru",
            "student_id": d.get("student_id"),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
            "message_count": len(d.get("transcript") or []),
            "context": d.get("context") or {},
        })

    hasil.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return hasil[:limit]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
