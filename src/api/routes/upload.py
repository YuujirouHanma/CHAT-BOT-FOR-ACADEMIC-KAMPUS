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
from src.storage.course_store import storage_root
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    course: str | None = Form(default=None),
    week: int | None = Form(default=None),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IndexResponse:
    """Upload a document, save to storage/{course}/minggu_{week}/, and index.

    If course+week given → saved under structured path.
    Otherwise → saved to data/uploads/ with UUID prefix.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename required")

    target_path = _save_file(file, course=course, week=week)

    try:
        result = await pipeline.index_document(
            file_path=target_path, course=course, week=week,
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
        course=result.course,
        week=result.week,
    )


def _save_file(file: UploadFile, course: str | None, week: int | None) -> Path:
    original_name = Path(file.filename or "").name
    if course and week is not None:
        dest_dir = storage_root() / course / f"minggu_{week}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / original_name
    else:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        target = settings.upload_dir / f"{uuid.uuid4().hex}_{original_name}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    logger.info("Saved upload to {}", target)
    return target