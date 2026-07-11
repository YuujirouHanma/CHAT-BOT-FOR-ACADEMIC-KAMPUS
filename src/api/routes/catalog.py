"""Catalog navigation for the guided UI.

Powers the click-through flow for students who don't know how to prompt an AI:
    mata kuliah → minggu → materi → pertanyaan template.

Everything is derived from what is actually indexed, so it auto-updates as new
material is uploaded and never lists a week/material with no answerable content.

    GET /catalog/courses
    GET /catalog/courses/{course_id}/weeks
    GET /catalog/courses/{course_id}/weeks/{week}/materials
    GET /catalog/materials/{content_id}/{source_file}/starter-questions
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import (
    CourseInfo,
    CourseListResponse,
    MaterialInfo,
    MaterialListResponse,
    StarterQuestionsResponse,
    WeekListResponse,
)
from src.pipeline import RAGPipeline

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/courses", response_model=CourseListResponse)
async def get_courses(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> CourseListResponse:
    """Step 1 — daftar mata kuliah yang punya materi terindex."""
    courses = await pipeline._store.list_courses()
    return CourseListResponse(courses=[CourseInfo(**c) for c in courses])


@router.get("/courses/{course_id}/weeks", response_model=WeekListResponse)
async def get_weeks(
    course_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> WeekListResponse:
    """Step 2 — minggu yang tersedia untuk satu mata kuliah (auto-update)."""
    weeks = await pipeline._store.list_weeks(course_id)
    if not weeks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tidak ada minggu terindex untuk mata kuliah '{course_id}'",
        )
    return WeekListResponse(course_id=course_id, weeks=weeks)


@router.get(
    "/courses/{course_id}/weeks/{week}/materials",
    response_model=MaterialListResponse,
)
async def get_materials(
    course_id: str,
    week: int,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> MaterialListResponse:
    """Step 3 — daftar materi (dokumen) untuk satu minggu."""
    materials = await pipeline._store.list_materials(course_id, week)
    if not materials:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tidak ada materi untuk '{course_id}' minggu {week}",
        )
    return MaterialListResponse(
        course_id=course_id,
        week=week,
        materials=[MaterialInfo(**m) for m in materials],
    )


@router.get(
    "/materials/{content_id}/{source_file}/starter-questions",
    response_model=StarterQuestionsResponse,
)
async def get_starter_questions(
    content_id: str,
    source_file: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StarterQuestionsResponse:
    """Step 4 — pertanyaan template untuk satu materi (auto-generate + cache)."""
    questions = await pipeline.starter_questions(content_id, source_file)
    return StarterQuestionsResponse(
        content_id=content_id,
        source_file=source_file,
        questions=questions,
    )
