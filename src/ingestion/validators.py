"""File validation utilities for safe upload handling.

Validates extension, size, and existence BEFORE handing the file to
Unstructured.io. Fail fast with clear errors.
"""
from __future__ import annotations

from pathlib import Path

from src.config import settings


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_file(path: Path) -> None:
    """Validate that a file is safe to parse.

    Checks performed:
    1. File exists and is a regular file (not a directory or symlink to dir).
    2. Extension is in the allowed set from settings.
    3. Size does not exceed the configured upload limit.

    Args:
        path: Path to the file to validate.

    Raises:
        FileValidationError: If any check fails.
    """
    if not path.exists():
        raise FileValidationError(f"File not found: {path}")

    if not path.is_file():
        raise FileValidationError(f"Not a regular file: {path}")

    ext = path.suffix.lower().lstrip(".")
    if ext not in settings.allowed_extensions_set:
        raise FileValidationError(
            f"Extension '.{ext}' not allowed. "
            f"Allowed: {sorted(settings.allowed_extensions_set)}"
        )

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise FileValidationError(f"File is empty: {path.name}")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileValidationError(
            f"File size {size_mb:.1f} MB exceeds limit of "
            f"{settings.max_upload_size_mb} MB ({path.name})"
        )