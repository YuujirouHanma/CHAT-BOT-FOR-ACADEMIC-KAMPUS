"""Riwayat percakapan: daftar, buka kembali, dan hapus.

Percakapan diidentifikasi oleh `session_id` yang sama dengan yang dipakai
`POST /chat/ask`. Membuka kembali sebuah percakapan cukup dengan mengirim
`session_id` itu lagi — server memuatnya dari disk bila sudah tidak di memori.
Endpoint di sini melengkapinya: menampilkan daftar, membaca isinya untuk
ditampilkan ulang, dan menghapus.

    GET    /conversations                 daftar ringkas, terbaru dulu
    GET    /conversations/{id}            isi lengkap untuk ditampilkan ulang
    DELETE /conversations/{id}            hapus permanen
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas import (
    ChatContext,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    TranscriptMessage,
)
from src.api.session import session_store
from src.storage import conversation_store

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _context(d: dict) -> ChatContext:
    weeks = list(d.get("weeks") or [])
    return ChatContext(
        course_id=d.get("course_id"),
        course_name=d.get("course_name"),
        weeks=weeks,
        week=weeks[0] if weeks else None,
        content_id=d.get("content_id"),
        source_file=d.get("source_file"),
        style=d.get("style"),
        topic=d.get("topic"),
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    student_id: str | None = Query(
        default=None,
        description="Saring hanya percakapan milik mahasiswa ini. "
                    "Tanpa ini, seluruh percakapan dikembalikan.",
    ),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationListResponse:
    ringkas = conversation_store.list_summaries(student_id=student_id, limit=limit)
    return ConversationListResponse(
        total=len(ringkas),
        conversations=[
            ConversationSummary(
                conversation_id=r["conversation_id"],
                title=r["title"],
                student_id=r.get("student_id"),
                created_at=r.get("created_at"),
                updated_at=r.get("updated_at"),
                message_count=r.get("message_count", 0),
                context=_context(r.get("context") or {}),
            )
            for r in ringkas
        ],
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    """Isi lengkap sebuah percakapan, siap ditampilkan ulang apa adanya.

    Transkrip memuat langkah navigasi juga (pilihan mata kuliah, minggu, materi),
    supaya tampilannya persis seperti saat percakapan berlangsung.
    """
    d = conversation_store.load(conversation_id)
    if d is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Percakapan '{conversation_id}' tidak ditemukan",
        )
    transcript = [
        TranscriptMessage(**m) for m in (d.get("transcript") or [])
    ]
    return ConversationDetail(
        conversation_id=d.get("conversation_id", conversation_id),
        title=d.get("title") or "Percakapan baru",
        student_id=d.get("student_id"),
        created_at=d.get("created_at"),
        updated_at=d.get("updated_at"),
        message_count=len(transcript),
        context=_context(d.get("context") or {}),
        transcript=transcript,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_model=None)
async def delete_conversation(conversation_id: str) -> None:
    """Hapus permanen. Session di memori ikut dibuang agar tidak tertulis ulang.

    Tanpa membuang salinan di memori, permintaan berikutnya dengan `session_id`
    yang sama akan menyimpannya kembali ke disk — dan percakapan yang sudah
    dihapus mahasiswa muncul lagi.
    """
    session_store.drop(conversation_id)
    if not conversation_store.delete(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Percakapan '{conversation_id}' tidak ditemukan",
        )
