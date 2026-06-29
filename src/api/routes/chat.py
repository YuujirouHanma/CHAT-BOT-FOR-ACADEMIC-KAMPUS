"""Chat route: answer a user question against the indexed corpus."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_pipeline
from src.api.schemas import QueryRequest, QueryResponse, SourceInfo
from src.api.session import session_store
from src.pipeline import RAGPipeline
from src.utils.logger import logger

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    session = session_store.get_or_create(body.session_id)

    if body.course is not None or body.week is not None or body.source_filter is not None:
        session.set_context(
            course=body.course, week=body.week, source_filter=body.source_filter,
        )

    effective_course = body.course or session.course
    effective_week = body.week or session.week
    effective_source = body.source_filter or session.source_filter

    logger.info(
        "Chat | session={} course={} week={} source={}",
        session.session_id[:8], effective_course, effective_week, effective_source,
    )

    try:
        result = await pipeline.query(
            question=body.question,
            course=effective_course,
            week=effective_week,
            source_filter=effective_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Query failed: {exc}") from exc

    session.add_turn("user", body.question)
    session.add_turn("assistant", result.answer)

    return QueryResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
        recommendations=result.recommendations,
        session_id=session.session_id,
    )