"""Browse endpoints — navigate contents and files.

GET /browse/contents
GET /browse/contents/{content_id}/files
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import (
    ContentListResponse,
    FileInfo,
    FileListResponse,
)
from src.pipeline import RAGPipeline
from src.storage.content_store import list_contents, list_files

router = APIRouter(prefix="/browse", tags=["browse"])


@router.get("/contents", response_model=ContentListResponse)
async def get_contents() -> ContentListResponse:
    return ContentListResponse(contents=list_contents())


@router.get(
    "/contents/{content_id}/files",
    response_model=FileListResponse,
)
async def get_files(
    content_id: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> FileListResponse:
    """List ALL files for a content_id — documents, videos, audio, etc.
    Documents get an `indexed` flag from Qdrant."""
    fs_files = list_files(content_id)
    if not fs_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No files found for content '{content_id}'",
        )

    indexed_records = await pipeline._store.list_indexed_files(content_id=content_id)
    indexed_names = {r["source_file"] for r in indexed_records}

    files = [
        FileInfo(
            filename=f.filename,
            content_id=f.content_id,
            size_bytes=f.size_bytes,
            size_display=f.size_display,
            category=f.category.value,
            is_indexable=f.is_indexable,
            indexed=f.filename in indexed_names,
        )
        for f in fs_files
    ]

    total_indexable = sum(1 for f in files if f.is_indexable)
    total_indexed = sum(1 for f in files if f.indexed)

    return FileListResponse(
        content_id=content_id,
        files=files,
        total_files=len(files),
        total_indexable=total_indexable,
        total_indexed=total_indexed,
    )
