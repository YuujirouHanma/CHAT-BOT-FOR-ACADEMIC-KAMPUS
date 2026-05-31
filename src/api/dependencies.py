"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Request

from src.pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError(
            "RAGPipeline not initialized. Did lifespan startup run?"
        )
    return pipeline
