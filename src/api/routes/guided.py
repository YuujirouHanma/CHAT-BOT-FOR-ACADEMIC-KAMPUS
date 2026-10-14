"""Guided Learning routes — personalized template-based AI tutoring.

Endpoints:
  GET  /guided/templates      — list templates instantly (NO LLM call)
  GET  /guided/topic-hints    — fetch material topic hints (may be slow, call async)
  POST /guided/ask            — run RAG with personalized system prompt + template
  POST /guided/ask/stream     — sama, tapi jawabannya dialirkan (Server-Sent Events)
  POST /guided/exercise       — buat latihan praktik yang bisa diunduh (lambat)
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from src import learning_styles, usage
from src.api.auth import request_id, tenant_chat
from src.api.dependencies import get_pipeline, guard_material
from src.api.schemas import (
    ExerciseRequest,
    ExerciseResponse,
    GuidedQueryRequest,
    GuidedQueryResponse,
    GuidedTemplatesResponse,
    SourceInfo,
    TemplateInfo,
)
from src.api.session import session_store
from src.generation.exercise import render_exercise, safe_filename
from src.generation.prompt_templates import (
    build_guided_system_prompt,
    get_templates_for_response,
)
from src.generation.prompts import label_kedalaman
from src.pipeline import RAGPipeline
from src.tenancy import TenantContext
from src.utils.logger import logger

router = APIRouter(prefix="/guided", tags=["guided-learning"])

# Learning style display labels

@router.get("/templates", response_model=GuidedTemplatesResponse)
async def get_templates(
    section_name: str = Query(default="Minggu Aktif"),
    class_name: str = Query(default=""),
    _tenant: TenantContext = Depends(tenant_chat),
) -> GuidedTemplatesResponse:
    """Return the 6 universal template cards immediately.

    This endpoint is intentionally FAST — no LLM or retrieval call.
    Material topic hints are fetched separately via GET /guided/topic-hints.
    """
    templates = [TemplateInfo(**t) for t in get_templates_for_response()]

    return GuidedTemplatesResponse(
        templates=templates,
        section_name=section_name,
        class_name=class_name,
        material_topics=[],  # returned empty; loaded separately via /topic-hints
    )


@router.get("/topic-hints")
async def get_topic_hints(
    content_id: str | None = Query(default=None),
    section_ids: str | None = Query(
        default=None,
        description="Minggu aktif, dipisah koma (mis. '3,6'). Kosong = seluruh kelas.",
    ),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_chat),
) -> dict:
    """Tebak topik dari cuplikan materi. CADANGAN, bukan sumber utama.

    Sumber utama topik sekarang tabel `concept` di MariaDB — peta konsep yang
    disusun dari SELURUH materi satu minggu, disimpan, dan bisa ditinjau dosen
    (lihat frontend/api/concepts_store.php). Frontend hanya jatuh ke sini bila
    minggu yang dipilih belum pernah disusun petanya.

    Endpoint ini mengambil 8 chunk sembarang lalu meminta model menebak. Untuk
    minggu berisi 21 chunk itu 32% materi yang dipilih tanpa urutan dan tanpa
    jaminan cakupan, dihitung ulang tiap panggilan, sehingga daftarnya berbeda
    antar kunjungan. Cukup untuk menghindari layar kosong, tidak cukup untuk
    dijadikan kunci apa pun.

    Menghormati minggu aktif: menyarankan topik dari minggu yang justru tidak
    bisa dijawab AI hanya membuat mahasiswa menabrak "materi tidak ditemukan".
    """
    if not content_id:
        return {"topics": []}

    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, content_id)

    # Dikirim sebagai string dipisah koma, bukan list, karena ini query GET —
    # bentuk apa pun yang tidak berupa angka diabaikan diam-diam.
    minggu: list[int] = []
    for bagian in (section_ids or "").split(","):
        bagian = bagian.strip()
        if bagian.isdigit():
            minggu.append(int(bagian))

    try:
        # Cuplikan chunk tanpa embedding/vektor, lewat store supaya filter
        # kampus tidak mungkin terlewat.
        cuplikan = await pipeline._store.sample_texts(
            content_id=content_id, section_ids=minggu or None, limit=8,
            tenant_id=tenant.tenant_id,
        )
        texts = [t[:500] for t in cuplikan if t]

        if not texts:
            return {"topics": []}

        combined = "\n\n---\n\n".join(texts)
        prompt = (
            f"Berdasarkan cuplikan materi berikut, sebutkan 5-7 topik/konsep utama "
            f"yang dibahas. Jawab HANYA dengan daftar topik singkat (satu per baris, "
            f"tanpa nomor, tanpa penjelasan tambahan):\n\n{combined}"
        )

        usage.pasang_tujuan("petunjuk-topik")
        raw_answer = await pipeline._generator._call_with_retry(
            [
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten yang mengekstrak topik utama dari teks "
                        "materi kuliah. Jawab singkat dalam bahasa Indonesia."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            reasoning=False,
        )

        topics = [
            line.strip().lstrip("-•*0123456789. ").strip()
            for line in raw_answer.splitlines()
            if line.strip() and len(line.strip()) > 3
        ]
        return {"topics": [t for t in topics if t][:8]}

    except Exception as exc:
        logger.warning("Could not extract material topics for hints: {}", exc)

    return {"topics": []}


@router.post("/ask/stream")
async def guided_ask_stream(
    request: Request,
    body: GuidedQueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_chat),
) -> StreamingResponse:
    """Versi /guided/ask yang mengalirkan jawabannya sebagai Server-Sent Events.

    Endpoint terpisah, bukan pengganti: /guided/ask tetap ada supaya pemanggil
    yang tidak bisa membaca aliran (dan jalur latihan praktik) tidak ikut
    terganggu.

    Tiap baris berbentuk ``data: {json}\\n\\n`` dengan kunci ``type``:
    ``sources`` sekali di awal, ``delta`` berkali-kali, lalu ``done`` sekali.
    ``error`` dikirim sebagai peristiwa biasa, bukan kode status HTTP — begitu
    aliran dimulai, statusnya sudah terkirim dan tidak bisa diubah lagi.
    """
    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, body.content_id)

    session = await session_store.get_or_create(body.session_id, tenant_id=tenant.tenant_id)
    profile = body.user_profile
    student_name = profile.student_name or "Mahasiswa"
    section_ids = body.active_section_ids()
    rid = request_id(request)

    logger.info(
        "Guided stream | tenant={} session={} template={} content_id={} sections={}",
        tenant.tenant_id, session.session_id[:8], body.template_id, body.content_id,
        section_ids,
    )

    guided_system_prompt = build_guided_system_prompt(
        student_name=student_name,
        class_name=body.class_name or "Mata Kuliah",
        section_name=body.section_name or "Minggu Aktif",
        learning_style=profile.learning_style,
        explanation_depth=profile.explanation_depth,
        material_topics=None,
    )

    personalized_label = (
        f"Untuk {student_name} "
        f"({learning_styles.resolve(profile.learning_style).label}, "
        f"{label_kedalaman(profile.explanation_depth)})"
    )
    riwayat = list(session.history)

    async def peristiwa() -> AsyncIterator[str]:
        def kirim(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield kirim({
            "type": "meta",
            "session_id": session.session_id,
            "template_id": body.template_id,
            "personalized_label": personalized_label,
        })

        jawaban = ""
        try:
            async for ev in pipeline.guided_query_stream(
                topic=body.topic,
                template_id=body.template_id,
                class_name=body.class_name or "Mata Kuliah",
                guided_system_prompt=guided_system_prompt,
                content_id=body.content_id,
                session_id=session.session_id,
                section_ids=section_ids,
                history=riwayat,
                student_name=student_name,
                user_id=body.user_id,
                tenant_id=tenant.tenant_id,
            ):
                if ev["type"] == "done":
                    jawaban = ev.get("answer", "")
                yield kirim(ev)
        except Exception:  # noqa: BLE001 - kegagalan harus sampai ke mahasiswa
            logger.exception("Guided stream gagal (tenant={} request_id={})", tenant.tenant_id, rid)
            # Pesan umum + request_id, bukan teks galat: aliran ini sampai ke
            # layar mahasiswa, dan pesan pustaka memuat path serta versi server.
            yield kirim({
                "type": "error",
                "message": "Gagal memproses pertanyaan",
                "request_id": rid,
            })
            return

        # Sesi disimpan setelah aliran selesai, bukan di tengah: menyimpan
        # jawaban separuh jadi akan mencemari konteks giliran berikutnya.
        if jawaban:
            session.add_turn("user", f"[{body.template_id}] {body.topic}")
            session.add_turn("assistant", jawaban)
            await session_store.save(session)

    return StreamingResponse(
        peristiwa(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tanpa ini, proxy penyangga (termasuk beberapa konfigurasi nginx)
            # menahan seluruh aliran sampai selesai — yang persis meniadakan
            # gunanya streaming.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ask", response_model=GuidedQueryResponse)
async def guided_ask(
    body: GuidedQueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_chat),
) -> GuidedQueryResponse:
    """Run RAG with a personalized system prompt based on template + user profile.

    This endpoint differs from /chat/ask in that:
    1. The system prompt is rebuilt per-request based on user_profile.
    2. The question is constructed from the selected template + topic.
    3. The response includes template_id and personalized_label for UI use.
    """
    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, body.content_id)

    session = await session_store.get_or_create(body.session_id, tenant_id=tenant.tenant_id)

    profile = body.user_profile
    student_name = profile.student_name or "Mahasiswa"
    learning_style = profile.learning_style
    explanation_depth = profile.explanation_depth

    section_ids = body.active_section_ids()

    logger.info(
        "Guided ask | session={} template={} style={} depth={} content_id={} sections={}",
        session.session_id[:8],
        body.template_id,
        learning_style,
        explanation_depth,
        body.content_id,
        section_ids,
    )

    # Build personalized system prompt
    guided_system_prompt = build_guided_system_prompt(
        student_name=student_name,
        class_name=body.class_name or "Mata Kuliah",
        section_name=body.section_name or "Minggu Aktif",
        learning_style=learning_style,
        explanation_depth=explanation_depth,
        material_topics=None,
    )

    try:
        result = await pipeline.guided_query(
            topic=body.topic,
            template_id=body.template_id,
            class_name=body.class_name or "Mata Kuliah",
            guided_system_prompt=guided_system_prompt,
            content_id=body.content_id,
            session_id=session.session_id,
            section_ids=section_ids,
            # Diambil sebelum giliran ini ditambahkan ke sesi.
            history=list(session.history),
            model_provider=body.model_provider,
            model_name=body.model_name,
            student_name=student_name,
            user_id=body.user_id,
            tenant_id=tenant.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("Guided query failed (tenant={})", tenant.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memproses pertanyaan",
        ) from None

    session.add_turn("user", f"[{body.template_id}] {body.topic}")
    session.add_turn("assistant", result.answer)
    # Wajib eksplisit: dengan backend Redis, Session hanyalah salinan lepas.
    await session_store.save(session)

    style_label = learning_styles.resolve(learning_style).label
    depth_label = label_kedalaman(explanation_depth)
    personalized_label = (
        f"Untuk {student_name} ({style_label}, {depth_label})"
    )

    return GuidedQueryResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
        recommendations=result.recommendations,
        session_id=session.session_id,
        interaction_id=result.interaction_id,
        template_id=body.template_id,
        personalized_label=personalized_label,
    )


@router.post("/exercise", response_model=ExerciseResponse)
async def guided_exercise(
    body: ExerciseRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_chat),
) -> ExerciseResponse:
    """Buat latihan praktik yang bisa diunduh untuk sebuah topik.

    Dipisah dari /guided/ask karena pembuatannya jauh lebih lama dan hasilnya
    berupa berkas, bukan jawaban chat. Format (.ipynb atau .md) ditentukan
    model dari sifat materi — notebook hanya untuk materi pemrograman.
    """
    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, body.content_id)

    logger.info(
        "Buat latihan | topik={} content_id={} sections={}",
        body.topic[:60], body.content_id, body.section_ids,
    )

    try:
        latihan, sources, alasan = await pipeline.generate_exercise(
            topic=body.topic,
            class_name=body.class_name or "Mata Kuliah",
            content_id=body.content_id,
            section_ids=body.section_ids,
            tenant_id=tenant.tenant_id,
        )
    except Exception:
        logger.exception("Pembuatan latihan gagal (tenant={})", tenant.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pembuatan latihan gagal",
        ) from None

    if latihan is None:
        # Dua sebab kegagalan yang menuntut tindakan berbeda: materi belum ada
        # (mahasiswa harus menunggu dosen) versus keluaran model kacau (cukup
        # diulang). Pesan yang menyamakan keduanya menyesatkan.
        if alasan == "tanpa_materi":
            pesan = (
                "Tidak ada materi tentang topik ini di minggu yang sedang kamu "
                "aktifkan. Aktifkan minggu lain, atau ganti topiknya."
                if body.section_ids
                else "Materi untuk topik ini belum tersedia di kelas. Coba topik lain, "
                     "atau tunggu dosen mengunggah materinya."
            )
        else:
            pesan = (
                "Latihan gagal disusun meski materinya ada. Ini biasanya sementara — "
                "coba tekan tombolnya sekali lagi."
            )
        # 422, bukan 500: permintaannya sah, hasilnya yang belum bisa dipakai.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=pesan)

    return ExerciseResponse(
        format=latihan.format,
        filename=safe_filename(latihan.judul, latihan.ekstensi),
        mime_type=latihan.mime,
        title=latihan.judul,
        intro=latihan.pengantar,
        task_count=len(latihan.tugas),
        content=render_exercise(latihan),
        sources=[SourceInfo(**s) for s in sources],
    )
