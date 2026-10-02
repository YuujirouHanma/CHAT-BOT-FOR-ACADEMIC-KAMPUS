"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from src.pipeline import RAGPipeline
from src.tenancy import TenantContext


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError(
            "RAGPipeline not initialized. Did lifespan startup run?"
        )
    return pipeline


async def guard_material(
    tenant: TenantContext, pipeline: RAGPipeline, content_id: str | None,
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

    # Tanpa materi tertentu, kueri di hilir mencakup SELURUH milik tenant —
    # lebih luas dari jatah kunci ini, jadi ditolak tanpa menanyakan apa pun ke
    # indeks. Melewatkan `None` ke `get_content_course` justru berbahaya: ia
    # memulangkan mata kuliah titik mana pun yang kebetulan terbaca lebih dulu,
    # sehingga kunci terbatas lolos atau ditolak secara untung-untungan.
    if not content_id:
        tenant.require_course(None)
        return

    course_id = await pipeline._store.get_content_course(
        content_id, tenant_id=tenant.tenant_id,
    )
    # `None` di sini berarti materinya tidak ada. Tetap ditolak: menjawab
    # "tidak ada" kepada kunci yang memang tidak berhak membedakan materi yang
    # belum diunggah dari materi fakultas lain.
    tenant.require_course(course_id)
