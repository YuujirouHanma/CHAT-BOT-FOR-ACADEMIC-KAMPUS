"""Filesystem operations for the storage/ directory — dipisah per tenant.

Convention:
    storage/
      {tenant_id}/
        {content_id}/
          {any file — pdf, docx, mp4, zip, etc.}

Files are categorized by extension but ALL files are listed.
Only "document" category files are indexable for RAG.

ISOLASI: pemisahan dilakukan secara FISIK lewat direktori, bukan lewat penyaringan
saat membaca. Dua pelanggan yang kebetulan memakai `content_id` sama — misalnya
`sbd-minggu-2`, pola yang justru dianjurkan — akan menulis ke direktori berbeda,
sehingga tidak mungkin saling menimpa berkas. Penyaringan saat baca tidak akan
menolong di sini: kerusakannya sudah terjadi pada saat penulisan.

`content_id` dan nama berkas berasal dari parameter URL, jadi keduanya
divalidasi ketat DAN path hasilnya diperiksa ulang agar benar-benar berada di
dalam direktori tenant. Dua lapis, karena satu kesalahan di sini berarti pembaca
dapat menjangkau berkas milik tenant lain — atau berkas sistem — lewat `../`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.schemas import FileCategory
from src.tenancy import require_tenant_id, storage_prefix

# content_id menjadi nama direktori: huruf, angka, titik, strip, garis bawah.
# Tanpa garis miring dan tanpa ".." — keduanya jalur keluar dari direktori induk.
_CONTENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def is_valid_content_id(content_id: str | None) -> bool:
    if not content_id or ".." in content_id:
        return False
    return bool(_CONTENT_ID_RE.match(content_id))


def _classify(filename: str) -> FileCategory:
    """Classify a file by its extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in settings.indexable_extensions_set:
        return FileCategory.DOCUMENT
    if ext in settings.video_extensions_set:
        return FileCategory.VIDEO
    if ext in settings.audio_extensions_set:
        return FileCategory.AUDIO
    if ext in settings.image_extensions_set:
        return FileCategory.IMAGE
    return FileCategory.OTHER


@dataclass(frozen=True)
class FileEntry:
    filename: str
    content_id: str
    path: Path
    size_bytes: int
    category: FileCategory

    @property
    def is_indexable(self) -> bool:
        """Documents parse directly; video/audio are indexed via transcription."""
        return self.category in (
            FileCategory.DOCUMENT,
            FileCategory.VIDEO,
            FileCategory.AUDIO,
        )

    @property
    def size_display(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        return f"{self.size_bytes / (1024 * 1024):.1f} MB"


def storage_root(*, tenant_id: str) -> Path:
    """Direktori milik satu tenant, dibuat bila belum ada."""
    root = settings.storage_path / storage_prefix(tenant_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def content_dir(content_id: str, *, tenant_id: str) -> Path | None:
    """Direktori satu content_id, atau None bila namanya tidak sah.

    Selain memvalidasi bentuk nama, hasil akhirnya diperiksa ulang dengan
    `resolve()` untuk memastikan ia sungguh berada di dalam direktori tenant —
    penjagaan berlapis terhadap bentuk path yang lolos regex namun tetap keluar
    lewat symlink atau penulisan khas Windows.
    """
    if not is_valid_content_id(content_id):
        return None
    root = storage_root(tenant_id=tenant_id).resolve()
    calon = (root / content_id).resolve()
    if calon != root and root not in calon.parents:
        return None
    return calon


def list_contents(*, tenant_id: str) -> list[str]:
    """Seluruh content_id milik SATU tenant."""
    root = storage_root(tenant_id=tenant_id)
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_files(content_id: str, *, tenant_id: str) -> list[FileEntry]:
    dir_ = content_dir(content_id, tenant_id=tenant_id)
    if dir_ is None or not dir_.exists():
        return []

    entries: list[FileEntry] = []
    for f in dir_.iterdir():
        if f.is_file() and not f.name.startswith("."):
            entries.append(
                FileEntry(
                    filename=f.name,
                    content_id=content_id,
                    path=f,
                    size_bytes=f.stat().st_size,
                    category=_classify(f.name),
                )
            )
    return sorted(entries, key=lambda e: e.filename)


def list_files_by_category(
    content_id: str, *, tenant_id: str,
) -> dict[FileCategory, list[FileEntry]]:
    """Group files by category."""
    files = list_files(content_id, tenant_id=tenant_id)
    result: dict[FileCategory, list[FileEntry]] = {c: [] for c in FileCategory}
    for f in files:
        result[f.category].append(f)
    return result


def resolve_file(content_id: str, filename: str, *, tenant_id: str) -> Path | None:
    """Path sebuah berkas, atau None bila namanya tidak sah / berkasnya tidak ada.

    Nama yang memuat pemisah path DITOLAK, bukan dinormalkan menjadi nama
    dasarnya. Menormalkan terasa lebih ramah, tetapi berarti permintaan
    `../bab1.pdf` diam-diam dilayani dengan berkas yang berbeda dari yang
    diminta — menyembunyikan klien yang bermasalah sekaligus klien yang sedang
    menjajaki batas. Menolak membuat keduanya terlihat.

    Berbeda dari jalur unggah, yang memang menormalkan: peramban kadang
    mengirim path lengkap, dan menolak unggahan karenanya hanya menyusahkan
    tanpa menambah keamanan, sebab nama tujuannya kita yang tentukan.
    """
    dir_ = content_dir(content_id, tenant_id=tenant_id)
    if dir_ is None:
        return None

    nama = filename or ""
    if not nama or nama.startswith(".") or ".." in nama:
        return None
    if "/" in nama or "\\" in nama or nama != Path(nama).name:
        return None

    p = (dir_ / nama).resolve()
    # Lapis terakhir: nama boleh bersih tetapi tetap berupa symlink ke luar.
    if dir_ not in p.parents:
        return None
    return p if p.exists() and p.is_file() else None


def tenant_usage_bytes(*, tenant_id: str) -> int:
    """Total ukuran berkas milik satu tenant. Dipakai menegakkan kuota simpan."""
    require_tenant_id(tenant_id, operation="tenant_usage_bytes")
    root = storage_root(tenant_id=tenant_id)
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total
