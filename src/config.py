"""Centralized configuration using pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

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

    # Muat model embedding & reranker saat startup, bukan saat permintaan pertama.
    # Menambah waktu nyala server, tetapi menghilangkan lonjakan pada pertanyaan
    # pertama mahasiswa — yang tanpa ini bisa menunggu beberapa menit karena
    # model ±2 GB baru diunduh/dimuat saat itu juga.
    warmup_models: bool = True

    # --- Inbound API auth (for other services calling this API, e.g. tim BE) ---
    # Kunci global warisan versi satu-pelanggan. Dipertahankan HANYA agar
    # integrasi lama tidak putus seketika; di produksi ia ditolak (lihat
    # src/api/main.py) karena tidak membawa identitas tenant — satu kunci untuk
    # semua pelanggan berarti tidak ada isolasi sama sekali.
    ragacademic_api_key: SecretStr = SecretStr("")
    # Tenant yang menerima permintaan berkunci warisan tersebut, bila masih dipakai.
    legacy_tenant_id: str = "default"

    # --- Multi-tenant ---
    # Pepper sisi server untuk hash kunci API, MAC jejak audit, dan indeks buta.
    # WAJIB diisi di produksi dan TIDAK BOLEH berubah setelah ada kunci terbit:
    # menggantinya membuat seluruh kunci yang beredar tidak lagi cocok dan
    # memutus verifikasi rantai audit yang sudah tertulis.
    tenant_key_pepper: SecretStr = SecretStr("")
    # Di pengembangan, tanpa satu pun tenant terdaftar, permintaan tanpa kunci
    # dilayani sebagai tenant "dev". Ditolak keras di produksi.
    dev_anonymous_tenant: str = "dev"
    # Cocokkan `tenant_id` yang dikirim klien dengan yang diturunkan dari kunci.
    # Ketidakcocokan berarti salah konfigurasi di sisi pemanggil — atau percobaan
    # mengakses data tenant lain. Keduanya harus ditolak, bukan didiamkan.
    enforce_tenant_body_match: bool = True

    # --- Enkripsi data pribadi (student_id) ---
    encrypt_pii: bool = False
    pii_kek: SecretStr = SecretStr("")     # base64url 32 byte; dev saja, produksi pakai KMS

    # --- Pembatasan laju ---
    rate_limit_enabled: bool = True
    rate_limit_ip_per_minute: int = Field(default=120, ge=0, le=100_000)
    # Baca alamat asli dari X-Forwarded-For. Nyalakan HANYA bila ada proxy tepercaya
    # di depan yang menimpa header itu — kalau tidak, siapa pun dapat memalsukan
    # alamatnya dan lolos dari pembatasan laju per IP.
    trust_proxy_headers: bool = False

    # --- Pengerasan HTTP ---
    # Bintang hanya wajar saat pengembangan. Di produksi isi daftar asal yang
    # sungguhan: `allow_credentials` bersama origin bintang membuat peramban mana
    # pun dapat memanggil API ini dengan kredensial pengguna.
    cors_allow_origins: str = "*"
    trusted_hosts: str = "*"
    max_request_body_mb: int = Field(default=110, ge=1, le=2000)
    # HSTS hanya berarti bila TLS memang sudah dipasang di depan (proxy/ingress).
    enable_hsts: bool = True
    hsts_max_age: int = Field(default=31_536_000, ge=0)

    # --- API keys ---
    openai_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    hf_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")

    # --- Generation LLM ---
    generation_provider: Literal[
        "openai", "groq", "huggingface", "ollama", "openrouter", "gemini"
    ] = "huggingface"
    generation_model: str = "Qwen/Qwen3.5-9B"
    generation_base_url: str | None = None
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Batas untuk jawaban utama. Dinaikkan dari 1024 karena model reasoning
    # (mis. Qwen 3.7) menghabiskan token untuk penalaran sebelum menulis jawaban,
    # dan token itu ikut dihitung ke batas ini — jawaban level "detail" berisiko
    # terpotong di tengah. Batas atas, bukan target; sisanya tidak ditagih.
    generation_max_tokens: int = Field(default=3072, ge=64, le=16384)

    # --- Embedding ---
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    enable_hybrid_search: bool = True

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Transcription (video/audio → teks, agar bisa ditanya) ---
    # Butuh ffmpeg di sistem. "disabled" → file media tetap disimpan tapi tak diindex.
    transcription_provider: Literal["groq", "openai", "disabled"] = "groq"
    transcription_model: str = "whisper-large-v3"
    transcription_language: str = "id"
    # Audio dipecah agar tiap potongan aman di bawah batas ukuran API.
    transcription_segment_seconds: int = Field(default=600, ge=60, le=1800)

    # --- Qdrant ---
    # local  → embedded on-disk client (QdrantClient(path=...)); no server needed.
    #          Right for dev/tests and single-process runs.
    # server → connect to a running Qdrant server at qdrant_host:qdrant_port.
    #          Right for Docker/production (see docker-compose.yml qdrant service).
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: str = "./qdrant_storage"
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

    # OpenAI-compatible base URL per provider. All providers below expose the
    # OpenAI chat-completions API shape, so one client class serves them all.
    PROVIDER_BASE_URLS: ClassVar[dict[str, str]] = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "huggingface": "https://router.huggingface.co/featherless-ai/v1",
        "ollama": "http://localhost:11434/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }

    def base_url_for(self, provider: str) -> str:
        """OpenAI-compatible base URL for any supported provider."""
        return self.PROVIDER_BASE_URLS[provider]

    def api_key_for(self, provider: str) -> str:
        """API key for any supported provider (Ollama needs no real key)."""
        keys = {
            "ollama": "ollama",
            "openai": self.openai_api_key.get_secret_value(),
            "groq": self.groq_api_key.get_secret_value(),
            "openrouter": self.openrouter_api_key.get_secret_value(),
            "gemini": self.gemini_api_key.get_secret_value(),
            "huggingface": self.hf_api_key.get_secret_value(),
        }
        return keys.get(provider, "")

    @model_validator(mode="after")
    def _resolve_generation_url(self) -> Settings:
        """Auto-set base_url for the default provider if not explicitly provided."""
        if self.generation_base_url is None:
            self.generation_base_url = self.PROVIDER_BASE_URLS[self.generation_provider]
        return self

    @property
    def generation_api_key(self) -> str:
        """API key for the default/active provider."""
        return self.api_key_for(self.generation_provider)

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

    def _csv_list(self, raw: str) -> list[str]:
        return [v.strip() for v in (raw or "").split(",") if v.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return self._csv_list(self.cors_allow_origins) or ["*"]

    @property
    def trusted_hosts_list(self) -> list[str]:
        return self._csv_list(self.trusted_hosts) or ["*"]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_request_body_bytes(self) -> int:
        return self.max_request_body_mb * 1024 * 1024

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