"""Chat route: answer a user question against the indexed corpus."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import QueryRequest, QueryResponse, SourceInfo
from src.pipeline import RAGPipeline
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    try:
        result = await pipeline.query(
            question=body.question,
            source_filter=body.source_filter,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {exc}",
        ) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
    )