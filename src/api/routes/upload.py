"""Upload route: save a document to storage and trigger indexing."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.dependencies import get_pipeline
from src.api.schemas import IndexResponse
from src.config import settings
from src.ingestion.validators import FileValidationError
from src.pipeline import RAGPipeline
from src.storage.content_store import storage_root
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    content_id: str | None = Form(default=None),
    course_id: str | None = Form(default=None),
    course_name: str | None = Form(default=None),
    week: int | None = Form(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IndexResponse:
    """Upload a document, save to storage/{content_id}/, and index.

    If content_id given → saved under storage/{content_id}/{filename}.
    Otherwise → saved to data/uploads/ with UUID prefix.

    course_id / course_name / week are optional — they power the guided catalog
    (mata kuliah → minggu → materi). If omitted they are derived from content_id
    (e.g. "sbd-minggu-2" → course "sbd", week 2).
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required")

    target_path = _save_file(file, content_id=content_id)

    try:
        result = await pipeline.index_document(
            file_path=target_path,
            content_id=content_id,
            course_id=course_id,
            course_name=course_name,
            week=week,
        )
    except FileValidationError as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        logger.exception("Indexing failed for {}", file.filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to index: {exc}") from exc

    return IndexResponse(
        source_file=result.source_file,
        elements_parsed=result.elements_parsed,
        chunks_created=result.chunks_created,
        points_stored=result.points_stored,
        content_id=result.content_id,
    )


def _save_file(file: UploadFile, content_id: str | None) -> Path:
    original_name = Path(file.filename or "").name
    if content_id:
        dest_dir = storage_root() / content_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / original_name
    else:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        target = settings.upload_dir / f"{uuid.uuid4().hex}_{original_name}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    logger.info("Saved upload to {}", target)
    return target
