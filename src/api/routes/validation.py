"""Validasi dosen atas jawaban AI.

    GET  /validation/answers          antrean jawaban untuk ditinjau
    POST /validation/answers/{id}     dosen menyatakan putusannya
    GET  /validation/stats            ringkasan untuk laporan/makalah

Butuh hak `answer:validate`, yang sengaja TIDAK termasuk dalam kunci aplikasi
mahasiswa. Kunci dosen diterbitkan terpisah lewat `tenantctl` — mahasiswa tidak
boleh menilai jawaban yang ia terima sendiri, dan lebih penting lagi, tidak boleh
membaca antrean berisi pertanyaan seluruh kelas.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from src import validation
from src.api.auth import tenant_validate
from src.api.schemas import (
    ValidationAnswerInfo,
    ValidationListResponse,
    ValidationRequest,
    ValidationStatsResponse,
    ValidationVerdictResponse,
)
from src.tenancy import TenantContext

router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("/answers", response_model=ValidationListResponse)
async def list_answers(
    course_id: str | None = Query(default=None, max_length=128),
    only_pending: bool = Query(
        default=True,
        description="True = hanya yang belum dinilai (antrean kerja dosen).",
    ),
    limit: int = Query(default=50, ge=1, le=200),
    tenant: TenantContext = Depends(tenant_validate),
) -> ValidationListResponse:
    """Jawaban AI beserta status validasinya, terbaru lebih dulu."""
    tenant.require_course(course_id) if course_id else None
    rows = validation.list_interactions(
        tenant_id=tenant.tenant_id, course_id=course_id,
        only_pending=only_pending, limit=limit,
    )
    return ValidationListResponse(
        total=len(rows),
        answers=[ValidationAnswerInfo(**r) for r in rows],
    )


@router.post("/answers/{interaction_id}", response_model=ValidationVerdictResponse)
async def submit_verdict(
    body: ValidationRequest,
    interaction_id: str = Path(max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
    tenant: TenantContext = Depends(tenant_validate),
) -> ValidationVerdictResponse:
    """Simpan putusan dosen atas satu jawaban.

    Putusan lama tidak ditimpa — dosen yang berubah pikiran menghasilkan catatan
    baru, dan yang berlaku adalah yang terakhir.
    """
    try:
        tersimpan = validation.record_verdict(
            validation.ValidationRecord(
                interaction_id=interaction_id,
                verdict=body.verdict,
                catatan=body.catatan or "",
                dosen_id=body.dosen_id or "",
                course_id=body.course_id,
                content_id=body.content_id,
            ),
            tenant_id=tenant.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return ValidationVerdictResponse(
        interaction_id=interaction_id,
        verdict=tersimpan["verdict"],
        catatan=tersimpan["catatan"],
        at=tersimpan["at"],
    )


@router.get("/stats", response_model=ValidationStatsResponse)
async def get_stats(
    course_id: str | None = Query(default=None, max_length=128),
    tenant: TenantContext = Depends(tenant_validate),
) -> ValidationStatsResponse:
    """Ringkasan hasil validasi — angka yang bisa langsung dikutip di makalah."""
    return ValidationStatsResponse(**validation.stats(
        tenant_id=tenant.tenant_id, course_id=course_id,
    ))
