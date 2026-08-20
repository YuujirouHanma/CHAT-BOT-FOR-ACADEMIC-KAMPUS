"""FastAPI application entry point.

Run: uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from src.api import errors
from src.api.auth import require_tenant
from src.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from src.api.routes import (
    batch,
    browse,
    catalog,
    chat,
    conversations,
    models,
    styles,
    upload,
)
from src.config import settings
from src.ingestion.nltk_data import warn_if_missing
from src.pipeline import RAGPipeline
from src.security import crypto
from src.storage import tenant_store
from src.utils.logger import logger


def _check_security_config() -> None:
    """Cegah konfigurasi pengembangan ikut terbawa ke produksi.

    Semua pemeriksaan di sini MENGGAGALKAN STARTUP di produksi, bukan sekadar
    memperingatkan. Peringatan pada log startup terbukti mudah terlewat, dan
    setiap butir di bawah ini berarti data satu pelanggan dapat terbaca
    pelanggan lain — kegagalan yang berisik jauh lebih murah daripada kebocoran
    yang senyap.
    """
    masalah: list[str] = []
    peringatan: list[str] = []

    ada_pepper = bool(settings.tenant_key_pepper.get_secret_value())
    jumlah_tenant = tenant_store.count()
    kunci_warisan = bool(settings.ragacademic_api_key.get_secret_value())

    if not ada_pepper:
        pesan = (
            "TENANT_KEY_PEPPER kosong — hash kunci API, MAC jejak audit, dan "
            "indeks buta berjalan tanpa rahasia sisi server."
        )
        (masalah if settings.is_production else peringatan).append(pesan)

    if jumlah_tenant == 0:
        pesan = (
            "Belum ada tenant terdaftar. Buat lewat: "
            "python -m scripts.tenantctl create <tenant_id>"
        )
        (masalah if settings.is_production else peringatan).append(pesan)

    if kunci_warisan:
        pesan = (
            "RAGACADEMIC_API_KEY (kunci global warisan) masih terisi. Kunci ini "
            "tidak membawa identitas tenant, sehingga isolasi antar pelanggan "
            "tidak dapat ditegakkan untuk permintaan yang memakainya."
        )
        (masalah if settings.is_production else peringatan).append(pesan)

    if settings.is_production:
        if "*" in settings.cors_origins_list:
            masalah.append(
                "CORS_ALLOW_ORIGINS masih '*' sementara kredensial diizinkan — "
                "isi daftar asal yang sungguhan."
            )
        if "*" in settings.trusted_hosts_list:
            peringatan.append(
                "TRUSTED_HOSTS masih '*'; isi nama host layanan agar "
                "permintaan dengan Host palsu ditolak."
            )
        if settings.encrypt_pii and not crypto.is_available():
            masalah.append(
                "ENCRYPT_PII=true tetapi penyandian tidak dapat dijalankan "
                "(paket 'cryptography' belum terpasang atau PII_KEK kosong)."
            )
        if settings.encrypt_pii and not getattr(
            crypto.key_provider(), "production_safe", False
        ):
            peringatan.append(
                "Kunci induk PII berasal dari variabel lingkungan. Di produksi "
                "pakai KMS (AWS/GCP) atau Vault — lihat docs/SECURITY.md."
            )
        if not settings.encrypt_pii:
            peringatan.append(
                "ENCRYPT_PII=false — student_id tersimpan apa adanya di disk."
            )

    for p in peringatan:
        logger.warning("KONFIGURASI: {}", p)
    if masalah:
        raise RuntimeError(
            "Konfigurasi tidak layak produksi:\n  - " + "\n  - ".join(masalah)
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting RAGAcademic (env={}, llm={}/{})",
                settings.app_env, settings.generation_provider, settings.generation_model)
    warn_if_missing()  # NLTK data — missing data breaks pptx/docx parsing at runtime
    _check_security_config()
    logger.info("Tenant terdaftar: {}", tenant_store.count())
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
    description="Multimodal RAG chatbot untuk materi kuliah (multi-tenant)",
    version="0.4.0",
    lifespan=lifespan,
    # Dokumentasi interaktif memerikan seluruh permukaan API. Di produksi ia
    # dimatikan: nilainya bagi tim integrasi tidak sebanding dengan peta gratis
    # yang diberikannya kepada siapa pun yang menemukan alamat layanan ini.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Urutan middleware penting: yang ditambahkan TERAKHIR berjalan PERTAMA.
# Konteks permintaan harus lebih dulu agar request_id sudah tersedia saat
# lapisan lain — termasuk balasan galat — membutuhkannya.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts_list,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    # Kredensial hanya diizinkan bila daftar asal sudah spesifik. Bintang plus
    # kredensial adalah kombinasi yang membuat halaman mana pun di internet
    # dapat memanggil API ini memakai kredensial pengunjungnya.
    allow_credentials="*" not in settings.cors_origins_list,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    max_age=600,
)

errors.install(app)

# Setiap router menuntut tenant. Rute yang butuh hak lebih spesifik
# (`content:write`, `conversation:delete`, …) memasangnya sendiri di berkasnya —
# dependency di sini adalah lantai, bukan langit-langit.
_auth = [Depends(require_tenant)]
app.include_router(upload.router, dependencies=_auth)
app.include_router(batch.router, dependencies=_auth)
app.include_router(browse.router, dependencies=_auth)
app.include_router(catalog.router, dependencies=_auth)
app.include_router(models.router, dependencies=_auth)
app.include_router(styles.router, dependencies=_auth)
app.include_router(conversations.router, dependencies=_auth)
app.include_router(chat.router, dependencies=_auth)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Pemeriksaan kesehatan — tanpa autentikasi, jadi isinya sengaja minim.

    Versi LLM dan nama model dihapus dari balasan: endpoint ini kerap terbuka ke
    pemantau luar, dan menyebut versi komponen mempermudah pencarian kerentanan
    yang sudah diketahui.
    """
    return {"status": "ok"}
