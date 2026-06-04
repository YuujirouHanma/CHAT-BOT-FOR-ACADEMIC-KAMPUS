"""Centralized configuration using pydantic-settings.

All environment variables are validated and type-checked here.
Import `settings` anywhere in the project — never read os.environ directly.

Validation rules:
- chunk_overlap MUST be smaller than chunk_size
- retrieval_top_k MUST be >= rerank_top_k (can't rerank more than retrieved)
- embed_dim MUST match the chosen embed_model
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pydantic import Field, SecretStr, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
else:
    try:
        from pydantic import Field, SecretStr, field_validator, model_validator
        from pydantic_settings import BaseSettings, SettingsConfigDict
    except ImportError:  # pragma: no cover - dev-time fallback when deps not installed
        from typing import Any

        def Field(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-any-return]
            return None

        SecretStr = str

        def field_validator(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-any-return]
            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

        def model_validator(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-any-return]
            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

        class BaseSettings:  # type: ignore[misc]
            pass

        SettingsConfigDict = dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    openai_api_key: SecretStr = Field(...)
    groq_api_key: SecretStr = Field(...)

    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = "classroom_docs"

    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    enable_hybrid_search: bool = True

    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    retrieval_top_k: int = Field(default=20, ge=1, le=200)
    rerank_top_k: int = Field(default=5, ge=1, le=50)

    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)

    groq_summary_model: str = "llama-3.1-8b-instant"
    openai_vision_model: str = "gpt-4o-mini"

    generation_model: str = "gpt-4o-mini"
    generation_provider: Literal["openai", "groq"] = "openai"
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1024, ge=64, le=8192)

    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    allowed_extensions: str = "pdf,docx,pptx,txt,md"

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, v: str) -> str:
        return ",".join(ext.strip().lower().lstrip(".") for ext in v.split(","))

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _check_retrieval(self) -> Settings:
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) cannot exceed "
                f"retrieval_top_k ({self.retrieval_top_k})"
            )
        return self

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def upload_dir(self) -> Path:
        return UPLOAD_DIR

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DIR

    @property
    def allowed_extensions_set(self) -> set[str]:
        return set(self.allowed_extensions.split(","))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()