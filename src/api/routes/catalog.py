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

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import assert_body_tenant, request_id, tenant_catalog
from src.api.dependencies import get_pipeline
from src.api.schemas import (
    CourseInfo,
    CourseListResponse,
    MaterialInfo,
    MaterialListResponse,
    QuizQuestion,
    QuizResponse,
    QuizResultItem,
    QuizSubmitRequest,
    QuizSubmitResponse,
    StarterQuestionsResponse,
    WeekListResponse,
)
from src.pipeline import RAGPipeline
from src.tenancy import TenantContext

router = APIRouter(prefix="/catalog", tags=["catalog"])


async def _guard_material(
    tenant: TenantContext, pipeline: RAGPipeline, content_id: str,
) -> None:
    """ABAC untuk endpoint yang hanya menerima `content_id`, bukan `course_id`.

    Pertanyaan pembuka dan kuis dibangkitkan dari ISI materi, jadi membiarkannya
    terbuka membuat pembatasan mata kuliah pada endpoint lain kehilangan arti —
    kunci yang dibatasi ke satu fakultas tetap dapat membaca ringkasan materi
    fakultas lain lewat jalan memutar ini.

    Kunci yang tidak dibatasi keluar lebih dulu, sehingga jalur umum tidak
    membayar satu kueri tambahan.
    """
    if tenant.allowed_courses is None:
        return
    course_id = await pipeline._store.get_content_course(
        content_id, tenant_id=tenant.tenant_id,
    )
    tenant.require_course(course_id)


@router.get("/courses", response_model=CourseListResponse)
async def get_courses(
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_catalog),
) -> CourseListResponse:
    """Step 1 — daftar mata kuliah milik tenant ini yang punya materi terindex."""
    courses = await pipeline._store.list_courses(tenant_id=tenant.tenant_id)
    # ABAC: kunci yang dibatasi hanya melihat mata kuliah jatahnya.
    if tenant.allowed_courses is not None:
        courses = [c for c in courses if c["course_id"] in tenant.allowed_courses]
    return CourseListResponse(courses=[CourseInfo(**c) for c in courses])


@router.get("/courses/{course_id}/weeks", response_model=WeekListResponse)
async def get_weeks(
    course_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_catalog),
) -> WeekListResponse:
    """Step 2 — minggu yang tersedia untuk satu mata kuliah (auto-update)."""
    tenant.require_course(course_id)
    weeks = await pipeline._store.list_weeks(course_id, tenant_id=tenant.tenant_id)
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
    tenant: TenantContext = Depends(tenant_catalog),
) -> MaterialListResponse:
    """Step 3 — daftar materi (dokumen) untuk satu minggu."""
    tenant.require_course(course_id)
    materials = await pipeline._store.list_materials(
        course_id, week, tenant_id=tenant.tenant_id,
    )
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
    model: str | None = None,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_catalog),
) -> StarterQuestionsResponse:
    """Step 4 — pertanyaan template untuk satu materi (auto-generate + cache).

    `model` (query param) opsional — key model dari /models untuk memilih LLM.
    """
    await _guard_material(tenant, pipeline, content_id)
    questions = await pipeline.starter_questions(
        content_id, source_file, model=model, tenant_id=tenant.tenant_id,
    )
    return StarterQuestionsResponse(
        content_id=content_id,
        source_file=source_file,
        questions=questions,
    )


@router.get(
    "/materials/{content_id}/{source_file}/quiz",
    response_model=QuizResponse,
)
async def get_quiz(
    content_id: str,
    source_file: str,
    model: str | None = None,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_catalog),
) -> QuizResponse:
    """Kuis pilihan ganda untuk satu materi (auto-generate + cache).

    `model` (query param) opsional. Tiap soal: pertanyaan, 4 opsi, answer_index, penjelasan.
    """
    await _guard_material(tenant, pipeline, content_id)
    quiz = await pipeline.quiz(
        content_id, source_file, model=model, tenant_id=tenant.tenant_id,
    )
    return QuizResponse(
        content_id=content_id,
        source_file=source_file,
        questions=[QuizQuestion(**q) for q in quiz],
    )


@router.post(
    "/materials/{content_id}/{source_file}/quiz/submit",
    response_model=QuizSubmitResponse,
)
async def submit_quiz(
    request: Request,
    content_id: str,
    source_file: str,
    body: QuizSubmitRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_catalog),
) -> QuizSubmitResponse:
    """Nilai jawaban kuis mahasiswa → skor + pembahasan per soal, dan simpan progres.

    `answers` = indeks opsi (0-3) yang dipilih, urut sesuai soal dari GET .../quiz.
    """
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    await _guard_material(tenant, pipeline, content_id)
    try:
        result = await pipeline.grade_quiz(
            content_id, source_file, body.answers,
            session_id=body.session_id, student_id=body.student_id,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return QuizSubmitResponse(
        content_id=content_id,
        source_file=source_file,
        total=result["total"],
        correct=result["correct"],
        score=result["score"],
        attempt_id=result["attempt_id"],
        results=[QuizResultItem(**r) for r in result["results"]],
    )
