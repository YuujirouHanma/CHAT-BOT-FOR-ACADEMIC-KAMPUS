"""FastAPI application entry point.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

The pipeline is constructed in `lifespan` so heavy models load once at startup
(not per request) and the Qdrant collection is created if missing.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import chat, upload
from src.config import settings
from src.pipeline import RAGPipeline
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Classroom RAG (env={})", settings.app_env)
    pipeline = RAGPipeline()
    await pipeline.setup()
    app.state.pipeline = pipeline
    logger.info("Pipeline ready")
    try:
        yield
    finally:
        logger.info("Shutting down")


app = FastAPI(
    title="Classroom RAG",
    description="Multimodal RAG chatbot for educational materials",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}