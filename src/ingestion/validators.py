"""File validation for safe upload and indexing."""
from __future__ import annotations

from pathlib import Path

from src.config import settings


class FileValidationError(ValueError):
    """Raised when a file fails validation."""


def validate_file(path: Path) -> None:
    """Validate that a file is safe to parse.

    Only checks: exists, is file, not empty, not oversized.
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


def validate_indexable(path: Path) -> None:
    """Validate that a file can be parsed and indexed."""
    validate_file(path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in settings.indexable_extensions_set:
        raise FileValidationError(
            f"Extension '.{ext}' is not indexable. "
            f"Indexable: {sorted(settings.indexable_extensions_set)}"
        )