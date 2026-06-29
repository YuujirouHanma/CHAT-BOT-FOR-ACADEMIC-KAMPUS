"""Batch indexing — index all indexable files in a course/week.

POST /documents/index-batch
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import BatchFileResult, BatchIndexRequest, BatchIndexSummary
from src.ingestion.validators import FileValidationError
from src.pipeline import IndexResult, RAGPipeline
from src.storage.course_store import list_files, list_weeks
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/index-batch", response_model=BatchIndexSummary)
async def index_batch(
    body: BatchIndexRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> BatchIndexSummary:
    """Index all indexable documents in storage/{course}/minggu_{week}/.

    Non-indexable files (video, audio, images) are silently skipped.
    If week is omitted, indexes every week of the course.
    """
    weeks_to_index = (
        [body.week] if body.week is not None
        else list_weeks(body.course)
    )

    if not weeks_to_index:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No weeks found for course '{body.course}'",
        )

    total = indexed = skipped = 0
    errors: list[str] = []
    results: list[BatchFileResult] = []

    for week in weeks_to_index:
        files = list_files(body.course, week)
        for f in files:
            total += 1

            if not f.is_indexable:
                skipped += 1
                logger.debug("Skipping non-indexable: {} ({})", f.filename, f.category.value)
                continue

            try:
                result: IndexResult = await pipeline.index_document(
                    file_path=f.path,
                    course=body.course,
                    week=week,
                )
                indexed += 1
                results.append(BatchFileResult(
                    filename=f.filename,
                    course=body.course,
                    week=week,
                    chunks_created=result.chunks_created,
                    points_stored=result.points_stored,
                    status="ok",
                ))
            except FileValidationError as exc:
                skipped += 1
                errors.append(f"{f.filename}: {exc}")
                logger.warning("Skipped {}: {}", f.filename, exc)
            except Exception as exc:
                errors.append(f"{f.filename}: {exc}")
                logger.exception("Failed to index {}", f.filename)

    logger.info(
        "Batch done: {}/{} indexed, {} skipped, {} errors",
        indexed, total, skipped, len(errors),
    )
    return BatchIndexSummary(
        total_files=total,
        indexed=indexed,
        skipped=skipped,
        errors=errors,
        results=results,
    )