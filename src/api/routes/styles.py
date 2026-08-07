"""Daftar gaya belajar yang boleh dipilih mahasiswa.

Gaya belajar mengganti system prompt yang mengatur CARA LLM menjawab dan keluaran
tambahan yang ia hasilkan (diagram Mermaid, kode yang dapat diekspor ke notebook).
Klien memakai daftar ini untuk merender pilihannya, sehingga tidak perlu
menyalin nama gaya secara manual.
"""
from __future__ import annotations

from fastapi import APIRouter

from src import learning_styles
from src.api.schemas import LearningStyleInfo, LearningStyleListResponse

router = APIRouter(prefix="/learning-styles", tags=["learning-styles"])


@router.get("", response_model=LearningStyleListResponse)
async def list_learning_styles() -> LearningStyleListResponse:
    return LearningStyleListResponse(
        default=learning_styles.DEFAULT_STYLE,
        styles=[
            LearningStyleInfo(
                key=s.key,
                label=s.label,
                description=s.description,
                produces_notebook=s.produces_notebook,
            )
            for s in learning_styles.all_styles()
        ],
    )
