"""FastAPI application entry point.

Run: uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import verify_api_key
from src.api.routes import batch, browse, catalog, chat, models, upload
from src.config import settings
from src.ingestion.nltk_data import warn_if_missing
from src.pipeline import RAGPipeline
from src.utils.logger import logger


def _check_auth_config() -> None:
    """Cegah konfigurasi auth pengembangan ikut terbawa ke produksi.

    `RAGACADEMIC_API_KEY` kosong mematikan autentikasi — memang disengaja untuk
    pengembangan lokal, tetapi kalau terbawa ke produksi seluruh materi kuliah
    dapat diakses siapa pun tanpa kredensial. Di produksi ini dijadikan galat
    yang menggagalkan startup, bukan sekadar peringatan yang mudah terlewat.
    """
    if settings.ragacademic_api_key.get_secret_value():
        return
    if settings.app_env == "production":
        raise RuntimeError(
            "RAGACADEMIC_API_KEY kosong sementara APP_ENV=production — "
            "API akan terbuka tanpa autentikasi. Isi key tersebut di .env."
        )
    logger.warning(
        "AUTENTIKASI MATI (RAGACADEMIC_API_KEY kosong). Wajar untuk pengembangan "
        "lokal; WAJIB diisi sebelum deployment."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting RAGAcademic (env={}, llm={}/{})",
                settings.app_env, settings.generation_provider, settings.generation_model)
    warn_if_missing()  # NLTK data — missing data breaks pptx/docx parsing at runtime
    _check_auth_config()
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
app.include_router(catalog.router, dependencies=_auth)
app.include_router(models.router, dependencies=_auth)
app.include_router(chat.router, dependencies=_auth)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "0.3.0",
        "llm": f"{settings.generation_provider}/{settings.generation_model}",
    }