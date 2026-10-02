"""Centralized configuration using pydantic-settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings

# re-export keeps Pylance happy
from pydantic_settings import SettingsConfigDict as SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Batas atas anggaran token keluaran, dipakai BERSAMA oleh `generation_max_tokens`
# di bawah dan oleh mekanisme percobaan-ulang di src/generation/llm.py. Ditaruh di
# sini supaya keduanya tidak pernah lepas sinkron: menaikkan batas config tanpa
# menaikkan plafon percobaan ulang membuat percobaan itu berhenti lebih awal
# daripada yang diizinkan config — diam-diam, tanpa pesan galat apa pun.
MAX_GENERATION_TOKENS = 16384

# Base URL kompatibel-OpenAI untuk tiap penyedia. Semua penyedia di bawah memakai
# bentuk API chat-completions yang sama, jadi satu kelas klien melayani semuanya.
GENERATION_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "huggingface": "https://router.huggingface.co/featherless-ai/v1",
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


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
    # pertama mahasiswa — yang tanpa ini menunggu beberapa menit karena model
    # +-2 GB baru diunduh/dimuat saat itu juga. Matikan bila EMBEDDING_BACKEND dan
    # RERANKER_BACKEND keduanya "openrouter": tidak ada model lokal untuk dimuat,
    # dan pemanasan hanya membakar satu panggilan API percuma di tiap pekerja.
    warmup_models: bool = True

    # --- Inbound API auth (for other services calling this API, e.g. tim BE) ---
    # Kunci global warisan versi satu-kampus. Hanya diterima di luar produksi,
    # dipetakan ke `legacy_tenant_id`; di produksi server menolak menyala bila
    # masih terisi — lihat src/api/main.py::_check_security_config.
    ragacademic_api_key: SecretStr = SecretStr("")

    # --- Multi-kampus (tenant) ---
    # Pepper sisi server untuk hash kunci API kampus, MAC jejak audit, dan indeks
    # buta. WAJIB di produksi dan TIDAK BOLEH berubah setelah ada kunci terbit:
    # menggantinya membuat seluruh kunci yang beredar tidak lagi cocok dan memutus
    # verifikasi rantai audit yang sudah tertulis.
    tenant_key_pepper: SecretStr = SecretStr("")
    # Direktori registri kampus & hash kuncinya, relatif terhadap PROJECT_ROOT.
    # Wajib dipersistenkan (volume Docker) — tanpa itu kunci hilang saat redeploy.
    tenants_dir: str = "data/tenants"
    legacy_tenant_id: str = "default"
    # Tenant sementara untuk pengembangan lokal: berlaku hanya bila belum ada satu
    # pun kampus terdaftar, tidak ada kunci warisan, dan bukan produksi.
    dev_anonymous_tenant: str = "dev"
    # Cocokkan `tenant_id` yang dikirim klien di body dengan yang diturunkan dari
    # kunci. Ketidakcocokan berarti salah konfigurasi di sisi pemanggil — atau
    # percobaan membaca data kampus lain. Keduanya ditolak, bukan didiamkan.
    enforce_tenant_body_match: bool = True

    # --- Admin API (kelola kampus lewat HTTP) ---
    # MATI secara bawaan, dan itu disengaja. Endpoint ini dapat menerbitkan kunci
    # untuk kampus MANA PUN — ia yang paling berbahaya di seluruh layanan.
    # Nyalakan hanya pada instance terpisah yang tidak terjangkau dari internet,
    # bukan pada instance yang sama dengan yang melayani mahasiswa.
    enable_admin_api: bool = False

    # --- Enkripsi data pribadi (student_id) ---
    # student_id melekat pada seluruh riwayat belajar seseorang: bocornya berkas
    # percakapan berarti bocornya "siapa mempelajari apa".
    encrypt_pii: bool = False
    pii_kek: SecretStr = SecretStr("")  # base64url 32 byte; dev saja, produksi KMS

    # --- Pembatasan laju ---
    # Kuota permintaan per kampus (menit & hari, lihat TenantQuota). Memakai Redis
    # bila SESSION_BACKEND=redis sehingga berlaku lintas pekerja uvicorn.
    rate_limit_enabled: bool = True
    # Lapis kedua, per alamat IP: kuota kampus tidak menolong bila satu kunci bocor
    # lalu dipakai ramai-ramai dari satu mesin. 0 = tanpa batas per IP.
    rate_limit_ip_per_minute: int = Field(default=120, ge=0, le=100_000)
    # Baca alamat asli dari X-Forwarded-For. Nyalakan HANYA bila ada proxy tepercaya
    # di depan yang MENIMPA header itu — kalau tidak, siapa pun dapat memalsukan
    # alamatnya dan lolos dari pembatasan laju per IP.
    trust_proxy_headers: bool = False

    # --- Pengerasan HTTP ---
    # Kosong = tidak ada peramban lintas-origin yang boleh memanggil API ini.
    # Backend dipanggil server-ke-server oleh portal PHP, jadi itu bawaan yang
    # benar; isi daftar asal (dipisah koma) hanya bila memang ada klien peramban.
    cors_allow_origins: str = ""
    # Tolak permintaan yang menyebut Host palsu. "*" hanya wajar saat pengembangan.
    trusted_hosts: str = "*"
    # Batas ukuran body permintaan, diperiksa SEBELUM body dibaca. Multipart
    # diurai FastAPI sebelum dependency auth berjalan, jadi tanpa batas ini
    # permintaan tanpa kunci pun dapat memenuhi disk sementara server.
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
    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = None

    # --- Generation LLM ---
    generation_provider: Literal[
        "openai", "groq", "huggingface", "ollama", "openrouter", "gemini"
    ] = "huggingface"
    generation_model: str = "Qwen/Qwen3.5-9B"
    generation_base_url: str | None = None
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    # Batas untuk jawaban utama. Model reasoning (mis. Qwen 3.7) membakar token
    # untuk penalaran sebelum menulis jawaban, dan token itu ikut dihitung ke batas
    # ini — jawaban level "detail" berisiko terpotong di tengah. Ini batas atas,
    # bukan target: sisa jatah tidak ditagih.
    generation_max_tokens: int = Field(default=3072, ge=64, le=MAX_GENERATION_TOKENS)

    # --- Sesi obrolan ---
    # "memory" hanya benar bila backend berjalan sebagai SATU proses. Begitu
    # uvicorn dijalankan dengan beberapa pekerja, tiap pekerja punya dict
    # sendiri dan mahasiswa kehilangan konteks percakapan secara acak.
    session_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://redis:6379/0"
    # Plafon token "berpikir" tersembunyi. Wajib lebih kecil dari
    # generation_max_tokens karena keduanya berbagi jatah yang sama — lihat
    # llm.py::_reasoning_body.
    reasoning_max_tokens: int = Field(default=512, ge=64, le=8192)

    # --- Embedding ---
    embedding_backend: Literal["local", "openrouter"] = "local"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = Field(default=1024, gt=0)
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    enable_hybrid_search: bool = True

    # --- Reranker ---
    reranker_backend: Literal["local", "openrouter"] = "local"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Batas token untuk PASANGAN kueri+dokumen, bukan untuk dokumen saja. Lewat
    # batas ini cross-encoder lokal membuang ekornya diam-diam. Hanya berlaku pada
    # jalur "local"; jalur "openrouter" memakai batas milik penyedia. Dapat
    # dinaikkan lewat .env setelah membaca peringatan pemotongan di log — dengan
    # ongkos: biaya rerank tumbuh bersama panjangnya, dan di CPU sudah +-1,96
    # detik per pasangan.
    reranker_max_length: int = Field(default=512, ge=128, le=8192)

    # --- Transkripsi (video/audio -> teks, agar bisa ditanya) ---
    # Butuh ffmpeg di sistem. "disabled" -> berkas media tetap disimpan tapi tidak
    # diindeks, sehingga tidak pernah muncul sebagai sumber jawaban.
    transcription_provider: Literal["groq", "openai", "disabled"] = "groq"
    transcription_model: str = "whisper-large-v3"
    transcription_language: str = "id"
    # Audio dipecah agar tiap potongan aman di bawah batas ukuran unggah API.
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

    @model_validator(mode="after")
    def _check_generation_budget(self) -> Settings:
        # Token penalaran dan token jawaban berbagi jatah yang sama. Bila plafon
        # penalaran menyamai atau melampaui anggaran total, model dapat kehabisan
        # jatah sebelum menulis satu karakter jawaban pun — gejalanya balasan
        # kosong tanpa galat, yang mahal sekali untuk dilacak saat rilis.
        if self.reasoning_max_tokens >= self.generation_max_tokens:
            raise ValueError(
                f"reasoning_max_tokens ({self.reasoning_max_tokens}) harus < "
                f"generation_max_tokens ({self.generation_max_tokens})"
            )
        return self

    @model_validator(mode="after")
    def _check_tenant_ids(self) -> Settings:
        # Diperiksa saat konfigurasi dimuat: nilai yang salah ketik di .env baru
        # akan meledak sebagai 500 di setiap permintaan tanpa kunci.
        from src.tenancy import is_valid_tenant_id

        for nama in ("legacy_tenant_id", "dev_anonymous_tenant"):
            if not is_valid_tenant_id(getattr(self, nama)):
                raise ValueError(
                    f"{nama.upper()} harus huruf kecil/angka/'-'/'_' (2-63 karakter)"
                )
        return self

    @model_validator(mode="after")
    def _resolve_generation_url(self) -> Settings:
        """Auto-set base_url if not explicitly provided."""
        if self.generation_base_url is None:
            self.generation_base_url = GENERATION_PROVIDER_URLS[self.generation_provider]
        return self

    def base_url_for(self, provider: str) -> str:
        """Base URL kompatibel-OpenAI untuk penyedia mana pun yang didukung.

        GENERATION_BASE_URL di .env sengaja hanya menimpa penyedia AKTIF: itulah
        satu-satunya yang dimaksud operator saat mengarahkan ke gateway sendiri
        atau instans Ollama lokal. Penyedia lain — yang dipakai pemilih model per
        percakapan — tetap memakai alamat resminya.
        """
        if provider == self.generation_provider and self.generation_base_url:
            return self.generation_base_url
        return GENERATION_PROVIDER_URLS[provider]

    def api_key_for(self, provider: str) -> str:
        """Kunci API untuk penyedia mana pun (Ollama tidak butuh kunci sungguhan).

        Penyedia yang tidak dikenal mengembalikan string kosong, bukan kunci milik
        penyedia lain: daftar model yang ditawarkan ke mahasiswa disaring dengan
        fungsi ini, dan nilai balik yang keliru membuat model yang tak dapat
        dipanggil ikut muncul di sana.
        """
        keys = {
            "ollama": "ollama",
            "openai": self.openai_api_key.get_secret_value(),
            "groq": self.groq_api_key.get_secret_value(),
            "openrouter": self.openrouter_api_key.get_secret_value(),
            "gemini": self.gemini_api_key.get_secret_value(),
            "huggingface": self.hf_api_key.get_secret_value(),
        }
        return keys.get(provider, "")

    @property
    def generation_api_key(self) -> str:
        """Return the right API key for the active provider."""
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

    @property
    def tenants_path(self) -> Path:
        return PROJECT_ROOT / self.tenants_dir

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def _csv_list(self, raw: str) -> list[str]:
        return [v.strip() for v in (raw or "").split(",") if v.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        """Daftar kosong berarti CORS tidak dipasang sama sekali — lihat main.py."""
        return self._csv_list(self.cors_allow_origins)

    @property
    def trusted_hosts_list(self) -> list[str]:
        """Kosong diperlakukan '*': salah ketik di .env tidak boleh menolak semua."""
        return self._csv_list(self.trusted_hosts) or ["*"]

    @property
    def max_request_body_bytes(self) -> int:
        return self.max_request_body_mb * 1024 * 1024

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
