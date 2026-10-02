"""Filesystem operations for the storage/ directory.

Convention (multi-kampus):
    storage/
      {tenant_id}/
        {content_id}/
          {any file — pdf, docx, mp4, zip, etc.}
        _uploads/            ← unggahan tanpa content_id

Tiap kampus punya direktorinya sendiri, dipisah secara FISIK: dua kampus yang
sama-sama punya "class-3" tidak mungkin saling menimpa berkas, dan penyaringan
saat baca tidak akan menolong bila kerusakannya sudah terjadi saat menulis.

`content_id` dan nama berkas datang dari pemanggil lalu menjadi bagian path, jadi
keduanya diperiksa ketat dan path hasilnya dipastikan (lewat `resolve()`) tetap
berada di dalam direktori kampus pemanggil.

Files are categorized by extension but ALL files are listed.
Only "document" category files are indexable for RAG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from src.config import settings
from src.schemas import FileCategory
from src.tenancy import storage_prefix

UPLOADS_DIRNAME = "_uploads"

# Huruf, angka, titik, strip, garis bawah; diawali huruf/angka. Tanpa "/" dan "\"
# — keduanya jalan keluar dari direktori induk. Pola ini juga mustahil bertabrakan
# dengan `_uploads`.
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
        return self.category == FileCategory.DOCUMENT

    @property
    def size_display(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        return f"{self.size_bytes / (1024 * 1024):.1f} MB"


def storage_root(*, tenant_id: str) -> Path:
    """Direktori milik satu kampus: storage/{tenant_id}/."""
    root = settings.storage_path / storage_prefix(tenant_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def content_dir(content_id: str, *, tenant_id: str) -> Path:
    """storage/{tenant_id}/{content_id}/. ValueError bila content_id tidak sah."""
    if not is_valid_content_id(content_id):
        raise ValueError("content_id tidak valid")
    root = storage_root(tenant_id=tenant_id)
    path = root / content_id
    # Lapis terakhir: nama boleh bersih tetapi direktorinya tetap bisa berupa
    # symlink ke luar kampus ini.
    if path.exists() and root.resolve() not in path.resolve().parents:
        raise ValueError("content_id tidak valid")
    return path


def list_contents(*, tenant_id: str) -> list[str]:
    root = storage_root(tenant_id=tenant_id)
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and is_valid_content_id(d.name)
    )


def list_files(content_id: str, *, tenant_id: str) -> list[FileEntry]:
    # content_id datang dari jalur URL, jadi bentuknya tidak dijamin. Dibiarkan
    # melempar, permintaan dengan id tak sah dibalas 500 — seolah layanan rusak,
    # padahal yang salah permintaannya. Daftar kosong membuat rute menjawab 404.
    try:
        directory = content_dir(content_id, tenant_id=tenant_id)
    except ValueError:
        return []
    if not directory.exists():
        return []

    entries: list[FileEntry] = []
    for f in directory.iterdir():
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


def clean_filename(filename: str | None) -> str | None:
    """Nama dasar berkas unggahan, atau None bila tidak layak dipakai.

    Peramban dan klien Windows kadang mengirim path lengkap ("C:\\x\\bab1.pdf"),
    jadi pemisah kedua gaya dibuang lebih dulu. Nama tersembunyi, "..", dan nama
    bermuatan karakter kendali ditolak.
    """
    nama = PureWindowsPath((filename or "").strip()).name
    if not nama or nama.startswith(".") or nama in {".", ".."}:
        return None
    if any(ord(c) < 32 for c in nama) or len(nama) > 255:
        return None
    return nama


def resolve_file(content_id: str, filename: str, *, tenant_id: str) -> Path | None:
    """Path sebuah berkas milik kampus ini, atau None bila tidak sah / tidak ada.

    Nama bermuatan pemisah path DITOLAK, bukan dinormalkan: permintaan
    `../bab1.pdf` yang diam-diam dilayani berkas lain menyembunyikan klien yang
    bermasalah maupun yang sedang menjajaki batas.
    """
    try:
        directory = content_dir(content_id, tenant_id=tenant_id)
    except ValueError:
        return None
    if clean_filename(filename) != filename:
        return None
    p = (directory / filename).resolve()
    if directory.resolve() not in p.parents:
        return None
    return p if p.exists() and p.is_file() else None


def tenant_usage_bytes(*, tenant_id: str) -> int:
    """Total ukuran berkas milik satu kampus. Dipakai menegakkan kuota simpan."""
    total = 0
    for p in storage_root(tenant_id=tenant_id).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
