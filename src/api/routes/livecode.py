"""Latihan koding: mahasiswa menulis program, AI menilai dan mengoreksi.

    POST /livecode/exercises    daftar latihan untuk mata kuliah + minggu
    POST /livecode/submit       kirim kode, dapatkan koreksi
    GET  /livecode/submissions  riwayat kiriman
    GET  /livecode/stats        statistik percobaan (bahan analisis)

Kode mahasiswa TIDAK dijalankan di server — lihat `src/livecode.py` untuk
alasannya dan jalur yang benar bila eksekusi nanti dibutuhkan.

Satu hal yang menentukan di sini: saat mengirim kode, klien hanya menyebut
`exercise_id`. Definisi latihannya — perilaku yang diharapkan, rubrik, kasus
uji — diambil ulang di sisi server dari cache, TIDAK diterima dari klien. Kalau
klien boleh mengirimkannya, mahasiswa cukup mengirim rubrik kosong untuk lulus.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src import livecode as livecode_mod
from src.api.auth import assert_body_tenant, request_id, tenant_chat
from src.api.dependencies import get_pipeline
from src.api.schemas import (
    LivecodeExerciseInfo,
    LivecodeExercisesRequest,
    LivecodeExercisesResponse,
    LivecodeStatsResponse,
    LivecodeSubmissionInfo,
    LivecodeSubmissionListResponse,
    LivecodeSubmitRequest,
    LivecodeSubmitResponse,
)
from src.pipeline import RAGPipeline
from src.tenancy import TenantContext

router = APIRouter(prefix="/livecode", tags=["livecode"])


async def _exercises(
    pipeline: RAGPipeline, body, tenant: TenantContext,
) -> list[livecode_mod.Exercise]:
    return await pipeline.livecode_exercises(
        body.course_id, list(body.weeks), tenant_id=tenant.tenant_id,
        count=getattr(body, "count", 3), model=body.model,
    )


@router.post("/exercises", response_model=LivecodeExercisesResponse)
async def get_exercises(
    body: LivecodeExercisesRequest,
    request: Request,
    tenant: TenantContext = Depends(tenant_chat),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> LivecodeExercisesResponse:
    """Latihan koding yang diturunkan dari materi minggu tertentu."""
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    tenant.require_course(body.course_id)

    latihan = await _exercises(pipeline, body, tenant)
    if not latihan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Belum ada materi terindeks untuk '{body.course_id}' pada minggu "
                f"{body.weeks}, sehingga latihan tidak dapat disusun"
            ),
        )
    return LivecodeExercisesResponse(
        course_id=body.course_id,
        exercises=[LivecodeExerciseInfo(**e.as_public()) for e in latihan],
    )


@router.post("/submit", response_model=LivecodeSubmitResponse)
async def submit_code(
    body: LivecodeSubmitRequest,
    request: Request,
    tenant: TenantContext = Depends(tenant_chat),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> LivecodeSubmitResponse:
    """Kirim kode, dapatkan koreksi.

    Umpan baliknya bertingkat: apa yang sudah benar, di mana yang keliru, lalu
    petunjuk. Kode perbaikan yang lengkap sengaja TIDAK diberikan — menyodorkan
    jawaban menyelesaikan soalnya, tetapi menghapus proses belajarnya.
    """
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    tenant.require_course(body.course_id)

    latihan = await _exercises(pipeline, body, tenant)
    dipilih = next(
        (e for e in latihan if e.exercise_id == body.exercise_id), None,
    )
    if dipilih is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Latihan '{body.exercise_id}' tidak ditemukan",
        )

    hasil = await pipeline.grade_livecode(
        dipilih, body.code, tenant_id=tenant.tenant_id,
        student_id=body.student_id, session_id=body.session_id, model=body.model,
    )
    return LivecodeSubmitResponse(**hasil)


@router.get("/submissions", response_model=LivecodeSubmissionListResponse)
async def get_submissions(
    student_id: str | None = Query(default=None, max_length=128),
    exercise_id: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=50, ge=1, le=200),
    tenant: TenantContext = Depends(tenant_chat),
) -> LivecodeSubmissionListResponse:
    """Riwayat kiriman, terbaru lebih dulu.

    Kiriman yang GAGAL ikut tersimpan dan ditampilkan — berapa kali seorang
    mahasiswa mencoba sebelum berhasil justru data paling berguna untuk menilai
    apakah sebuah gaya belajar benar-benar membantu.
    """
    rows = livecode_mod.list_submissions(
        tenant_id=tenant.tenant_id, student_id=student_id,
        exercise_id=exercise_id, limit=limit,
    )
    return LivecodeSubmissionListResponse(
        total=len(rows),
        submissions=[
            LivecodeSubmissionInfo(
                submission_id=r.get("submission_id", ""),
                at=r.get("at"),
                exercise_id=r.get("exercise_id", ""),
                student_id=r.get("student_id"),
                lulus=r.get("lulus"),
                skor=r.get("skor"),
                code=r.get("code", ""),
            )
            for r in rows
        ],
    )


@router.get("/stats", response_model=LivecodeStatsResponse)
async def get_stats(
    exercise_id: str = Query(max_length=256),
    tenant: TenantContext = Depends(tenant_chat),
) -> LivecodeStatsResponse:
    """Statistik percobaan sebuah latihan."""
    return LivecodeStatsResponse(
        exercise_id=exercise_id,
        **livecode_mod.attempt_stats(
            tenant_id=tenant.tenant_id, exercise_id=exercise_id,
        ),
    )
