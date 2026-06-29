"""Centralized configuration using pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict as SettingsConfigDict  # re-export keeps Pylance happy

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- API keys ---
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    hf_api_key: SecretStr = SecretStr("")

    # --- Generation LLM ---
    generation_provider: Literal["openai", "groq", "huggingface", "ollama"] = "huggingface"
    generation_model: str = "Qwen/Qwen3.5-9B"
    generation_base_url: str | None = None
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=1024, ge=64, le=16384)

    # --- Summarization ---
    groq_summary_model: str = "llama-3.1-8b-instant"
    openai_vision_model: str = "gpt-4o-mini"

    # --- Embedding ---
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    enable_hybrid_search: bool = True

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = "classroom_docs"

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=20, ge=1, le=200)
    rerank_top_k: int = Field(default=5, ge=1, le=50)

    # --- Chunking ---
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0)

    # --- Storage ---
    storage_root: str = "storage"
    max_upload_size_mb: int = Field(default=100, ge=1, le=2000)

    # --- File extensions ---
    indexable_extensions: str = "pdf,docx,pptx,txt,md,csv,xlsx,html,rtf,rst,epub,tsv"
    video_extensions: str = "mp4,avi,mov,mkv,webm,flv,wmv,m4v"
    audio_extensions: str = "mp3,wav,ogg,flac,aac,m4a,wma"
    image_extensions: str = "jpg,jpeg,png,gif,bmp,webp,svg,tiff"

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
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

    @model_validator(mode="after")
    def _resolve_generation_url(self) -> Settings:
        """Auto-set base_url if not explicitly provided."""
        if self.generation_base_url is None:
            urls = {
                "openai": "https://api.openai.com/v1",
                "groq": "https://api.groq.com/openai/v1",
                "huggingface": "https://router.huggingface.co/featherless-ai/v1",
                "ollama": "http://localhost:11434/v1",
            }
            self.generation_base_url = urls[self.generation_provider]
        return self

    @property
    def generation_api_key(self) -> str:
        """Return the right API key for the active provider."""
        if self.generation_provider == "ollama":
            return "ollama"
        if self.generation_provider == "openai":
            return self.openai_api_key.get_secret_value()
        if self.generation_provider == "groq":
            return self.groq_api_key.get_secret_value()
        return self.hf_api_key.get_secret_value()

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def storage_path(self) -> Path:
        return PROJECT_ROOT / self.storage_root

    @property
    def upload_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "uploads"

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    def _ext_set(self, raw: str) -> set[str]:
        return {e.strip().lower().lstrip(".") for e in raw.split(",") if e.strip()}

    @property
    def indexable_extensions_set(self) -> set[str]:
        return self._ext_set(self.indexable_extensions)

    @property
    def video_extensions_set(self) -> set[str]:
        return self._ext_set(self.video_extensions)

    @property
    def audio_extensions_set(self) -> set[str]:
        return self._ext_set(self.audio_extensions)

    @property
    def image_extensions_set(self) -> set[str]:
        return self._ext_set(self.image_extensions)

    @property
    def all_known_extensions(self) -> set[str]:
        return (
            self.indexable_extensions_set
            | self.video_extensions_set
            | self.audio_extensions_set
            | self.image_extensions_set
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()