"""FastAPI application entry point.

Run: uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import verify_api_key
from src.api.routes import batch, browse, chat, upload
from src.config import settings
from src.pipeline import RAGPipeline
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting RAGAcademic (env={}, llm={}/{})",
                settings.app_env, settings.generation_provider, settings.generation_model)
    pipeline = RAGPipeline()
    await pipeline.setup()
    app.state.pipeline = pipeline
    logger.info("Pipeline ready")
    try:
        yield
    finally:
        logger.info("Shutting down")


app = FastAPI(
    title="RAGAcademic",
    description="Multimodal RAG chatbot untuk materi kuliah",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_auth = [Depends(verify_api_key)]
app.include_router(upload.router, dependencies=_auth)
app.include_router(batch.router, dependencies=_auth)
app.include_router(browse.router, dependencies=_auth)
app.include_router(chat.router, dependencies=_auth)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "0.3.0",
        "llm": f"{settings.generation_provider}/{settings.generation_model}",
    }