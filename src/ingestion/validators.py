"""File validation for safe upload and indexing."""
from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.ingestion.dedup import find_duplicate


class FileValidationError(ValueError):
    """Raised when a file fails validation."""


class DuplicateFileError(FileValidationError):
    """Isi berkas sudah ada di direktori materi yang sama.

    Sengaja turunan `FileValidationError`, bukan galat baru yang berdiri
    sendiri: rute unggah dan rute indeks massal sama-sama sudah menangani
    `FileValidationError` — yang pertama membalas 400 sambil menghapus berkas
    rangkapnya, yang kedua mencatatnya sebagai berkas dilewati berikut
    alasannya. Keduanya persis perlakuan yang kita inginkan untuk duplikat,
    jadi tidak ada penanganan baru yang perlu ditambahkan di lapisan atas.
    """


def validate_file(path: Path) -> None:
    """Validate that a file is safe to parse.

    Checks: exists, is file, not empty, not oversized, bukan salinan berkas
    yang sudah ada di direktori materi yang sama (lihat `dedup`).
    Extension check is done separately by the caller since
    non-indexable files (video, etc.) are still valid for storage.
    """
    if not path.exists():
        raise FileValidationError(f"File not found: {path}")

    if not path.is_file():
        raise FileValidationError(f"Not a regular file: {path}")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise FileValidationError(f"File is empty: {path.name}")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileValidationError(
            f"File size {size_mb:.1f} MB exceeds limit of "
            f"{settings.max_upload_size_mb} MB ({path.name})"
        )

    validate_not_duplicate(path)


def validate_not_duplicate(path: Path) -> None:
    """Tolak berkas yang isinya sudah ada di direktori materi yang sama.

    Diperiksa di sini — di ambang validasi — supaya kedua jalur ingestion
    (dokumen lewat `parser`, media lewat `transcriber`) tercakup sekaligus,
    dan supaya penolakannya terjadi SEBELUM parsing, peringkasan LLM, dan
    embedding dijalankan atas isi yang sudah kita punya.
    """
    kembar = find_duplicate(path)
    if kembar is None:
        return
    raise DuplicateFileError(
        f"Isi '{path.name}' sama persis dengan '{kembar.name}' yang sudah ada "
        f"di materi ini; berkas tidak diindeks ulang. Bila ini memang materi "
        f"untuk mata kuliah atau minggu yang berbeda, unggah dengan content_id "
        f"tersendiri agar tersimpan terpisah."
    )


def validate_indexable(path: Path) -> None:
    """Validate that a file can be parsed and indexed."""
    validate_file(path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in settings.indexable_extensions_set:
        raise FileValidationError(
            f"Extension '.{ext}' is not indexable. "
            f"Indexable: {sorted(settings.indexable_extensions_set)}"
        )