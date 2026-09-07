"""Validasi dosen atas jawaban yang diberikan AI kepada mahasiswa.

Kenapa ini ada
--------------
`docs/CHANGELOG_TIM_BE.md` mencatat apa adanya: ketepatan pengambilan materi
belum terjamin — pernah terukur pertanyaan tentang perulangan dijawab memakai
materi SQL. Selama itu masih mungkin terjadi, jawaban AI tidak boleh
diperlakukan sebagai kebenaran final, dan harus ada jalan bagi dosen untuk
menyatakan mana yang keliru.

Ada tiga manfaat sekaligus, dan yang ketiga yang paling menentukan:

1. Mahasiswa terlindungi dari jawaban salah yang tidak ada yang menyadarinya.
2. Dosen melihat apa yang sebenarnya dipelajari kelasnya dari AI.
3. Terkumpul data berlabel manusia — pasangan (pertanyaan, jawaban, putusan
   dosen, koreksi). Inilah yang membuat klaim "sistem ini akurat" dalam sebuah
   makalah dapat diuji, bukan sekadar diyakini.

Bentuk penyimpanan
------------------
Satu baris JSON per putusan, ditambahkan di akhir berkas, dipisah per tenant.
Putusan TIDAK menimpa putusan lama: dosen yang mengubah penilaiannya
menghasilkan baris baru, dan yang berlaku adalah yang terakhir. Riwayat
perubahan penilaian itu sendiri adalah data — pada penelitian, mengetahui bahwa
seorang dosen berubah pikiran lebih berharga daripada kerapian berkas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from src.config import PROJECT_ROOT
from src.hitl.logger import CONVERSATION_LOG, log_dir
from src.security import redact
from src.tenancy import require_tenant_id, storage_prefix
from src.utils.logger import logger

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
VERDICT_LOG: Final = "verdicts.jsonl"

Verdict = Literal["sesuai", "perlu_perbaikan", "tidak_sesuai"]
VERDICTS: Final[tuple[str, ...]] = ("sesuai", "perlu_perbaikan", "tidak_sesuai")

# Batas atas pembacaan log. Log interaksi tumbuh terus; tanpa batas, satu
# permintaan daftar bisa menarik berkas berukuran ratusan MB ke memori.
_MAX_SCAN = 5_000


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _verdict_path(tenant_id: str) -> Path:
    return VALIDATION_DIR / storage_prefix(tenant_id) / VERDICT_LOG


@dataclass(frozen=True)
class ValidationRecord:
    """Satu putusan dosen atas satu jawaban AI."""
    interaction_id: str
    verdict: str
    catatan: str = ""              # koreksi/alasan; wajib bila bukan "sesuai"
    dosen_id: str = ""
    course_id: str | None = None
    content_id: str | None = None
    at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "verdict": self.verdict,
            "catatan": self.catatan,
            "dosen_id": self.dosen_id,
            "course_id": self.course_id,
            "content_id": self.content_id,
            "at": self.at or _now(),
        }


def _read_jsonl(path: Path, limit: int = _MAX_SCAN) -> list[dict[str, Any]]:
    """Baca berkas JSONL, terbaru lebih dulu, dibatasi jumlah baris.

    Baris rusak dilewati, bukan menggagalkan seluruh pembacaan: satu tulisan
    yang terpotong (mis. proses mati di tengah) tidak boleh membuat seluruh
    riwayat validasi tidak terbaca.
    """
    if not path.exists():
        return []
    try:
        baris = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Gagal membaca {}: {}", path.name, exc)
        return []

    hasil: list[dict[str, Any]] = []
    for b in reversed(baris):
        b = b.strip()
        if not b:
            continue
        try:
            hasil.append(json.loads(b))
        except json.JSONDecodeError:
            continue
        if limit and len(hasil) >= limit:
            break
    return hasil


def latest_verdicts(*, tenant_id: str) -> dict[str, dict[str, Any]]:
    """Putusan TERAKHIR per interaction_id.

    Dibaca dari yang terbaru, jadi entri pertama yang ditemui untuk sebuah
    interaction_id adalah yang berlaku.
    """
    tenant_id = require_tenant_id(tenant_id, operation="latest_verdicts")
    terakhir: dict[str, dict[str, Any]] = {}
    for v in _read_jsonl(_verdict_path(tenant_id)):
        iid = v.get("interaction_id")
        if iid and iid not in terakhir:
            terakhir[iid] = v
    return terakhir


def record_verdict(rec: ValidationRecord, *, tenant_id: str) -> dict[str, Any]:
    """Simpan satu putusan dosen. Mengembalikan catatan yang tersimpan."""
    tenant_id = require_tenant_id(tenant_id, operation="record_verdict")
    if rec.verdict not in VERDICTS:
        raise ValueError(f"Putusan tidak dikenal: {rec.verdict!r}")
    if rec.verdict != "sesuai" and not rec.catatan.strip():
        # Menandai salah tanpa menjelaskan salahnya di mana tidak menolong
        # siapa pun — tidak mahasiswa, tidak juga analisis datanya nanti.
        raise ValueError("Catatan wajib diisi bila jawaban dinilai belum sesuai")

    isi = rec.as_dict()
    isi["tenant_id"] = tenant_id
    path = _verdict_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redact.scrub(isi), ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Gagal menyimpan putusan validasi: {}", exc)
        raise
    logger.info(
        "Validasi dosen | tenant={} interaction={} verdict={}",
        tenant_id, rec.interaction_id, rec.verdict,
    )
    return isi


def _course_of(interaction: dict[str, Any]) -> str | None:
    """Mata kuliah sebuah interaksi.

    Log tidak menyimpan `course_id` langsung, jadi diturunkan dari `content_id`
    yang berpola `<matkul>-minggu-<n>`.
    """
    from src.catalog import resolve_course_week

    course_id, _, _ = resolve_course_week(interaction.get("content_id"))
    return course_id


def list_interactions(
    *,
    tenant_id: str,
    course_id: str | None = None,
    only_pending: bool = True,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Jawaban AI beserta status validasinya, terbaru lebih dulu.

    `only_pending=True` menyisakan yang belum pernah dinilai — itulah antrean
    kerja dosen. `False` menampilkan semuanya beserta putusan yang sudah ada.
    """
    tenant_id = require_tenant_id(tenant_id, operation="list_interactions")
    putusan = latest_verdicts(tenant_id=tenant_id)
    interaksi = _read_jsonl(log_dir(tenant_id) / CONVERSATION_LOG)

    hasil: list[dict[str, Any]] = []
    for it in interaksi:
        iid = it.get("interaction_id")
        if not iid:
            continue
        v = putusan.get(iid)
        if only_pending and v is not None:
            continue
        if course_id and _course_of(it) != course_id:
            continue
        hasil.append({
            "interaction_id": iid,
            "at": it.get("timestamp"),
            "question": it.get("question", ""),
            "answer": it.get("answer", ""),
            "sources": it.get("sources") or [],
            "content_id": it.get("content_id"),
            "course_id": _course_of(it),
            "session_id": it.get("session_id"),
            "verdict": (v or {}).get("verdict"),
            "catatan": (v or {}).get("catatan", ""),
            "dosen_id": (v or {}).get("dosen_id", ""),
        })
        if len(hasil) >= limit:
            break
    return hasil


def stats(*, tenant_id: str, course_id: str | None = None) -> dict[str, Any]:
    """Ringkasan hasil validasi — angka yang langsung bisa dikutip di makalah."""
    tenant_id = require_tenant_id(tenant_id, operation="validation.stats")
    semua = list_interactions(
        tenant_id=tenant_id, course_id=course_id,
        only_pending=False, limit=_MAX_SCAN,
    )
    dinilai = [r for r in semua if r["verdict"]]
    hitung = {v: sum(1 for r in dinilai if r["verdict"] == v) for v in VERDICTS}
    total_dinilai = len(dinilai)
    return {
        "total_jawaban": len(semua),
        "sudah_dinilai": total_dinilai,
        "belum_dinilai": len(semua) - total_dinilai,
        "rincian": hitung,
        # Proporsi yang dinilai benar oleh dosen — inti klaim akurasi.
        # None (bukan 0) bila belum ada yang dinilai, supaya "belum diukur"
        # tidak terbaca sebagai "akurasinya nol".
        "akurasi": (
            round(hitung["sesuai"] / total_dinilai, 3) if total_dinilai else None
        ),
    }
