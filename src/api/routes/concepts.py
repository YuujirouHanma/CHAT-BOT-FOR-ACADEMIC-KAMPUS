"""Peta konsep tingkat minggu dan bank soal per konsep.

GET  /concepts/preview   — susun peta konsep satu minggu dan kembalikan apa adanya.
POST /concepts/questions — susun soal pilihan ganda untuk konsep yang dikirim.

Namanya "preview" karena memang belum menyimpan apa pun: tiap panggilan
menghitung ulang dan membayar satu panggilan LLM. Bentuk ini disengaja selama
ketepatan konsepnya masih dinilai manusia — begitu hasilnya dipercaya, peta akan
disimpan di MariaDB dan endpoint ini berubah jadi penghasil, bukan pembaca.

Karena itu jangan dipanggil dari jalur mahasiswa: ia lambat (beberapa detik) dan
hasilnya tidak stabil antar panggilan.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.auth import tenant_content_write
from src.api.dependencies import get_pipeline, guard_material
from src.api.schemas import (
    ConceptInfo,
    ConceptPreviewResponse,
    ConceptQuestionInfo,
    ConceptQuestionsRequest,
    ConceptQuestionsResponse,
    ConceptSourceInfo,
    FileCoverage,
    QuestionOptionInfo,
)
from src.indexing.concepts import susun_peta_konsep
from src.indexing.questions import KonsepMasukan, susun_bank_soal
from src.pipeline import RAGPipeline
from src.tenancy import TenantContext
from src.utils.logger import logger

router = APIRouter(prefix="/concepts", tags=["concepts"])


@router.get("/preview", response_model=ConceptPreviewResponse)
async def preview_concepts(
    content_id: str = Query(..., min_length=1),
    section_id: int | None = Query(default=None),
    class_name: str = Query(default="Mata Kuliah"),
    section_name: str = Query(default="Minggu Ini"),
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> ConceptPreviewResponse:
    """Susun peta konsep untuk satu minggu dari chunk yang sudah terindeks."""

    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, content_id)
    try:
        peta = await susun_peta_konsep(
            store=pipeline._store,
            generator=pipeline._generator,
            content_id=content_id,
            section_id=section_id,
            class_name=class_name,
            section_name=section_name,
            tenant_id=tenant.tenant_id,
        )
    except Exception:
        logger.exception("Penyusunan peta konsep gagal (tenant={})", tenant.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Penyusunan peta konsep gagal",
        ) from None

    tercakup = {cid for k in peta.konsep for cid in k.chunk_ids}
    cakupan = [
        FileCoverage(
            source_file=berkas,
            total=len(ids),
            tercakup=len(tercakup & set(ids)),
        )
        for berkas, ids in sorted(peta.chunk_per_berkas.items())
    ]

    return ConceptPreviewResponse(
        content_id=peta.content_id,
        section_id=peta.section_id,
        jumlah_chunk=peta.jumlah_chunk,
        chunk_dipakai=peta.chunk_dipakai,
        concepts=[
            ConceptInfo(
                nama=k.nama,
                deskripsi=k.deskripsi,
                urutan=k.urutan,
                jumlah_potongan=len(k.chunk_ids),
                sumber=[
                    ConceptSourceInfo(
                        source_file=s.source_file,
                        halaman_awal=s.halaman_awal,
                        halaman_akhir=s.halaman_akhir,
                    )
                    for s in k.sumber
                ],
            )
            for k in peta.konsep
        ],
        cakupan=cakupan,
    )


@router.post("/questions", response_model=ConceptQuestionsResponse)
async def buat_soal_konsep(
    payload: ConceptQuestionsRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> ConceptQuestionsResponse:
    """Susun soal pilihan ganda (default dua: ingatan + penerapan) untuk tiap konsep yang dikirim.

    POST, bukan GET, karena konsepnya ikut dikirim dalam badan permintaan:
    yang jadi bahan soal adalah konsep yang SUDAH tersimpan dan ditinjau dosen,
    bukan hasil ekstraksi baru. Menyusun ulang peta di sini justru akan membuat
    soal lahir dari konsep yang belum dilihat siapa pun.

    Satu panggilan LLM per konsep, dijalankan bersamaan — hitung beberapa detik
    untuk satu minggu penuh. Jangan dipanggil dari jalur mahasiswa.
    """

    # Dijaga sebelum apa pun dikerjakan: menolak setelah jawaban terlanjur
    # dibuat berarti membayar panggilan model untuk permintaan yang memang
    # tidak boleh dilayani.
    await guard_material(tenant, pipeline, payload.content_id)
    try:
        bank = await susun_bank_soal(
            store=pipeline._store,
            generator=pipeline._generator,
            content_id=payload.content_id,
            section_id=payload.section_id,
            class_name=payload.class_name,
            section_name=payload.section_name,
            jumlah_soal=payload.jumlah_soal,
            konsep=[
                KonsepMasukan(
                    concept_id=k.concept_id,
                    nama=k.nama,
                    deskripsi=k.deskripsi,
                    sumber=[s.model_dump() for s in k.sumber],
                )
                for k in payload.concepts
            ],
            tenant_id=tenant.tenant_id,
        )
    except Exception:
        logger.exception("Penyusunan bank soal gagal (tenant={})", tenant.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Penyusunan bank soal gagal",
        ) from None

    return ConceptQuestionsResponse(
        content_id=bank.content_id,
        section_id=bank.section_id,
        questions=[
            ConceptQuestionInfo(
                concept_id=s.concept_id,
                nama_konsep=s.nama_konsep,
                tipe=s.tipe,
                pertanyaan=s.pertanyaan,
                opsi=[QuestionOptionInfo(teks=o.teks, benar=o.benar) for o in s.opsi],
                pembahasan=s.pembahasan,
                urutan=s.urutan,
            )
            for s in bank.soal
        ],
        konsep_gagal=bank.konsep_gagal,
    )
