"""Filesystem operations for the storage/ directory.

Convention:
    storage/
      {content_id}/
        {any file — pdf, docx, mp4, zip, etc.}

Files are categorized by extension but ALL files are listed.
Only "document" category files are indexable for RAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.schemas import FileCategory


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


def storage_root() -> Path:
    root = settings.storage_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_contents() -> list[str]:
    root = storage_root()
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_files(content_id: str) -> list[FileEntry]:
    content_dir = storage_root() / content_id
    if not content_dir.exists():
        return []

    entries: list[FileEntry] = []
    for f in content_dir.iterdir():
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


def list_files_by_category(content_id: str) -> dict[FileCategory, list[FileEntry]]:
    """Group files by category."""
    files = list_files(content_id)
    result: dict[FileCategory, list[FileEntry]] = {c: [] for c in FileCategory}
    for f in files:
        result[f.category].append(f)
    return result


def resolve_file(content_id: str, filename: str) -> Path | None:
    p = storage_root() / content_id / filename
    return p if p.exists() and p.is_file() else None
