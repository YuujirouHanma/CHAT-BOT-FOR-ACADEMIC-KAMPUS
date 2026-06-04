"""Upload route: receive a document, save it, and trigger indexing."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.dependencies import get_pipeline
from src.api.schemas import IndexResponse
from src.config import settings
from src.ingestion.validators import FileValidationError
from src.pipeline import RAGPipeline
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IndexResponse:
    """Upload and index a document.

    Validation, parsing, and indexing happen inline. For very large files
    or batch uploads, prefer a background-task queue (Celery/RQ/Arq).
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    target_path = _safe_save(file)

    try:
        result = await pipeline.index_document(target_path)
    except FileValidationError as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        logger.exception(f"Indexing failed for {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {exc}",
        ) from exc

    return IndexResponse(
        source_file=result.source_file,
        elements_parsed=result.elements_parsed,
        chunks_created=result.chunks_created,
        points_stored=result.points_stored,
    )


def _safe_save(file: UploadFile) -> Path:
    """Save upload to disk with a sanitized name to avoid path traversal."""
    original_name = Path(file.filename or "").name
    safe_name = f"{uuid.uuid4().hex}_{original_name}"
    target = settings.upload_dir / safe_name
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    logger.info(f"Saved upload to {target}")
    return target