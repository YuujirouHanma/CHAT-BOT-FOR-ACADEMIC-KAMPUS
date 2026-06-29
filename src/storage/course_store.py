"""Filesystem operations for the storage/ directory.

Convention:
    storage/
      {course}/
        minggu_{week}/
          {any file — pdf, docx, mp4, zip, etc.}

Files are categorized by extension but ALL files are listed.
Only "document" category files are indexable for RAG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.schemas import FileCategory

_WEEK_PATTERN = re.compile(r"^minggu_(\d+)$", re.IGNORECASE)


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
    course: str
    week: int
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


def storage_root() -> Path:
    root = settings.storage_path
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_courses() -> list[str]:
    root = storage_root()
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_weeks(course: str) -> list[int]:
    course_dir = storage_root() / course
    if not course_dir.exists():
        return []

    weeks: list[int] = []
    for d in course_dir.iterdir():
        m = _WEEK_PATTERN.match(d.name)
        if d.is_dir() and m:
            weeks.append(int(m.group(1)))
    return sorted(weeks)


def list_files(course: str, week: int) -> list[FileEntry]:
    week_dir = storage_root() / course / f"minggu_{week}"
    if not week_dir.exists():
        return []

    entries: list[FileEntry] = []
    for f in week_dir.iterdir():
        if f.is_file() and not f.name.startswith("."):
            entries.append(
                FileEntry(
                    filename=f.name,
                    course=course,
                    week=week,
                    path=f,
                    size_bytes=f.stat().st_size,
                    category=_classify(f.name),
                )
            )
    return sorted(entries, key=lambda e: e.filename)


def list_files_by_category(
    course: str, week: int
) -> dict[FileCategory, list[FileEntry]]:
    """Group files by category."""
    files = list_files(course, week)
    result: dict[FileCategory, list[FileEntry]] = {c: [] for c in FileCategory}
    for f in files:
        result[f.category].append(f)
    return result


def resolve_file(course: str, week: int, filename: str) -> Path | None:
    p = storage_root() / course / f"minggu_{week}" / filename
    return p if p.exists() and p.is_file() else None