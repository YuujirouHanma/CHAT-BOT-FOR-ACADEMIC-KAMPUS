"""Browse endpoints — navigate courses, weeks, and files.

GET /browse/courses
GET /browse/courses/{course}/weeks
GET /browse/courses/{course}/weeks/{week}/files
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import (
    CourseListResponse,
    FileInfo,
    FileListResponse,
    WeekListResponse,
)
from src.pipeline import RAGPipeline
from src.storage.course_store import list_courses, list_files, list_weeks

router = APIRouter(prefix="/browse", tags=["browse"])


@router.get("/courses", response_model=CourseListResponse)
async def get_courses() -> CourseListResponse:
    return CourseListResponse(courses=list_courses())


@router.get("/courses/{course}/weeks", response_model=WeekListResponse)
async def get_weeks(course: str) -> WeekListResponse:
    weeks = list_weeks(course)
    if not weeks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Course '{course}' not found or has no weeks",
        )
    return WeekListResponse(course=course, weeks=weeks)


@router.get(
    "/courses/{course}/weeks/{week}/files",
    response_model=FileListResponse,
)
async def get_files(
    course: str,
    week: int,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> FileListResponse:
    """List ALL files for a course/week — documents, videos, audio, etc.
    Documents get an `indexed` flag from Qdrant."""
    fs_files = list_files(course, week)
    if not fs_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No files found for {course} minggu_{week}",
        )

    # Get indexed filenames from Qdrant
    indexed_records = await pipeline._store.list_indexed_files(
        course=course, week=week
    )
    indexed_names = {r["source_file"] for r in indexed_records}

    files = [
        FileInfo(
            filename=f.filename,
            course=f.course,
            week=f.week,
            size_bytes=f.size_bytes,
            size_display=f.size_display,
            category=f.category.value,
            is_indexable=f.is_indexable,
            indexed=f.filename in indexed_names,
        )
        for f in fs_files
    ]

    total_indexable = sum(1 for f in files if f.is_indexable)
    total_indexed = sum(1 for f in files if f.indexed)

    return FileListResponse(
        course=course,
        week=week,
        files=files,
        total_files=len(files),
        total_indexable=total_indexable,
        total_indexed=total_indexed,
    )