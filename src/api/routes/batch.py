"""Batch indexing — index all indexable files in a content folder.

POST /documents/index-batch
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.auth import assert_body_tenant, client_ip, request_id, tenant_content_write
from src.api.dependencies import get_pipeline
from src.api.schemas import BatchFileResult, BatchIndexRequest, BatchIndexSummary
from src.ingestion.validators import FileValidationError
from src.pipeline import IndexResult, RAGPipeline
from src.security import audit
from src.storage.content_store import list_contents, list_files
from src.tenancy import TenantContext
from src.utils.logger import logger

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/index-batch", response_model=BatchIndexSummary)
async def index_batch(
    request: Request,
    body: BatchIndexRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    tenant: TenantContext = Depends(tenant_content_write),
) -> BatchIndexSummary:
    """Index all indexable documents in storage/{tenant_id}/{content_id}/.

    Non-indexable files (video, audio, images) are silently skipped.
    """
    rid = request_id(request)
    assert_body_tenant(tenant, body.tenant_id, request_id_=rid)

    if body.content_id not in list_contents(tenant_id=tenant.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Materi tidak ditemukan",
        )

    files = list_files(body.content_id, tenant_id=tenant.tenant_id)
    total = indexed = skipped = 0
    errors: list[str] = []
    results: list[BatchFileResult] = []

    for f in files:
        total += 1

        if not f.is_indexable:
            skipped += 1
            logger.debug("Skipping non-indexable: {} ({})", f.filename, f.category.value)
            continue

        try:
            result: IndexResult = await pipeline.index_document(
                file_path=f.path,
                content_id=body.content_id,
                tenant_id=tenant.tenant_id,
            )
            indexed += 1
            results.append(BatchFileResult(
                filename=f.filename,
                content_id=body.content_id,
                chunks_created=result.chunks_created,
                points_stored=result.points_stored,
                status="ok",
            ))
        except FileValidationError as exc:
            skipped += 1
            errors.append(f"{f.filename}: {exc}")
            logger.warning("Skipped {}: {}", f.filename, exc)
        except Exception:
            # Pesan galat pustaka TIDAK ikut ke balasan. `errors` di sini terkirim
            # sebagai body 200 biasa, sehingga ia melewati penjagaan di
            # `src/api/errors.py` yang menahan rincian 5xx — kebocoran yang sama,
            # hanya lewat pintu yang berbeda. Rinciannya tetap ada di log server.
            errors.append(f"{f.filename}: gagal diindeks")
            logger.exception("Failed to index {}", f.filename)

    logger.info(
        "Batch done (tenant={}): {}/{} indexed, {} skipped, {} errors",
        tenant.tenant_id, indexed, total, skipped, len(errors),
    )
    audit.record(
        audit.CONTENT_INDEX, tenant_id=tenant.tenant_id, actor=tenant.key_id,
        request_id=rid, ip=client_ip(request),
        detail={"content_id": body.content_id, "indexed": indexed, "skipped": skipped},
    )
    return BatchIndexSummary(
        total_files=total,
        indexed=indexed,
        skipped=skipped,
        errors=errors,
        results=results,
    )
