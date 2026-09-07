"""Evaluasi berkala: kuis, ETS, dan EAS atas rentang minggu.

    GET  /evaluation/schedule    jadwal evaluasi satu semester
    POST /evaluation/questions   ambil soalnya
    POST /evaluation/submit      kirim jawaban, dapatkan nilai + pembahasan

Berbeda dari kuis materi (`/catalog/materials/.../quiz`) yang menguji SATU
berkas: evaluasi di sini mencakup beberapa minggu sekaligus, dan soalnya
bercampur jenis — pilihan ganda, benar/salah, isian singkat, esai, dan koding.

Soal di-cache per (mata kuliah, rentang minggu, komposisi), sehingga seluruh
mahasiswa satu kelas mengerjakan soal yang sama dan nilainya dapat dibandingkan.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src import evaluation as evaluation_mod
from src.api.auth import assert_body_tenant, request_id, tenant_chat
from src.api.dependencies import get_pipeline
from src.api.schemas import (
    EvaluationItemInfo,
    EvaluationPlanInfo,
    EvaluationQuestionsRequest,
    EvaluationQuestionsResponse,
    EvaluationScheduleResponse,
    EvaluationSubmitRequest,
    EvaluationSubmitResponse,
    GradedItemInfo,
)
from src.pipeline import RAGPipeline
from src.tenancy import TenantContext

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _plan_info(p: evaluation_mod.EvaluationPlan) -> EvaluationPlanInfo:
    return EvaluationPlanInfo(
        week=p.week,
        kind=p.kind,
        label=p.label,
        weeks_covered=list(p.weeks_covered),
        range_text=p.range_text,
        counts=dict(p.blueprint.counts),
        total=p.blueprint.total,
    )


def _resolve_plan(
    week: int | None, weeks: list[int] | None, kind: str | None,
    counts: dict[str, int] | None,
) -> evaluation_mod.EvaluationPlan:
    """Tentukan rencana evaluasi dari permintaan klien.

    `week` memakai jadwal bawaan; `weeks` membuat rentang bebas. Salah satunya
    wajib — tanpa itu tidak jelas materi mana yang diuji.
    """
    if weeks:
        try:
            return evaluation_mod.custom_plan(weeks, kind or "kuis", counts)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
            ) from exc

    if week is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Isi 'week' (jadwal bawaan) atau 'weeks' (rentang bebas)",
        )
    plan = evaluation_mod.plan_for_week(week)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Minggu {week} bukan minggu evaluasi. "
                f"Minggu evaluasi: {list(evaluation_mod.EVALUATION_WEEKS)}"
            ),
        )
    return plan


@router.get("/schedule", response_model=EvaluationScheduleResponse)
async def get_schedule(
    tenant: TenantContext = Depends(tenant_chat),
) -> EvaluationScheduleResponse:
    """Jadwal evaluasi satu semester beserta cakupan minggunya."""
    return EvaluationScheduleResponse(
        evaluation_weeks=list(evaluation_mod.EVALUATION_WEEKS),
        plans=[
            _plan_info(evaluation_mod.DEFAULT_SCHEDULE[w])
            for w in evaluation_mod.EVALUATION_WEEKS
        ],
    )


@router.post("/questions", response_model=EvaluationQuestionsResponse)
async def get_questions(
    body: EvaluationQuestionsRequest,
    request: Request,
    tenant: TenantContext = Depends(tenant_chat),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvaluationQuestionsResponse:
    """Soal evaluasi untuk sebuah mata kuliah.

    Kunci jawaban TIDAK disertakan — hanya soal dan opsinya. Mengirimkan kunci
    ke klien berarti mahasiswa mana pun yang membuka panel jaringan peramban
    dapat membacanya sebelum menjawab.
    """
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    tenant.require_course(body.course_id)

    plan = _resolve_plan(body.week, body.weeks, body.kind, body.counts)
    items = await pipeline.evaluation(
        body.course_id, plan, tenant_id=tenant.tenant_id, model=body.model,
    )
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Belum ada materi terindeks untuk '{body.course_id}' "
                f"{plan.range_text}, sehingga soal tidak dapat disusun"
            ),
        )

    return EvaluationQuestionsResponse(
        course_id=body.course_id,
        plan=_plan_info(plan),
        items=[
            EvaluationItemInfo(
                index=i,
                type=it.get("type", "pilihan_ganda"),
                question=it.get("question", ""),
                options=it.get("options") or [],
                starter_code=it.get("starter_code") or "",
            )
            for i, it in enumerate(items)
        ],
    )


@router.post("/submit", response_model=EvaluationSubmitResponse)
async def submit(
    body: EvaluationSubmitRequest,
    request: Request,
    tenant: TenantContext = Depends(tenant_chat),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvaluationSubmitResponse:
    """Nilai jawaban mahasiswa atas sebuah evaluasi.

    Dinilai terhadap soal yang SAMA dengan yang diterima mahasiswa — soalnya
    diambil dari cache, bukan disusun ulang.
    """
    assert_body_tenant(tenant, body.tenant_id, request_id_=request_id(request))
    tenant.require_course(body.course_id)

    plan = _resolve_plan(body.week, body.weeks, body.kind, body.counts)
    items = await pipeline.evaluation(
        body.course_id, plan, tenant_id=tenant.tenant_id, model=body.model,
    )
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Soal evaluasi tidak tersedia untuk rentang minggu tersebut",
        )

    hasil = await pipeline.grade_evaluation(
        items, list(body.answers), tenant_id=tenant.tenant_id, model=body.model,
    )
    return EvaluationSubmitResponse(
        course_id=body.course_id,
        plan=_plan_info(plan),
        total=hasil["total"],
        benar=hasil["benar"],
        skor=hasil["skor"],
        per_jenis=hasil["per_jenis"],
        perlu_tinjauan=hasil["perlu_tinjauan"],
        items=[GradedItemInfo(**b) for b in hasil["butir"]],
    )
