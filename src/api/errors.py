"""Balasan galat yang seragam dan tidak membocorkan isi dalam sistem.

Pesan galat bawaan sebuah kerangka kerja adalah sumber kebocoran yang mudah
terlewat: jejak tumpukan menyebutkan path berkas di server, versi pustaka, dan
kadang potongan kueri beserta datanya. Semua itu mempermudah penyerang memetakan
sistem sebelum menyerangnya.

Aturan di sini:

- Galat 5xx SELALU dibalas dengan pesan umum. Rinciannya hanya masuk log server.
- Setiap balasan galat membawa `request_id`, sehingga pengguna dapat melaporkan
  satu kode pendek dan operator menemukan barisnya di log tanpa perlu rincian
  apa pun ikut terkirim ke luar.
- Galat validasi menyebut FIELD yang salah, tanpa memantulkan kembali nilai yang
  dikirim — nilai itu bisa saja berisi kredensial yang salah tempat.
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.tenancy import TenantPermissionError, TenantScopeError
from src.utils.logger import logger

# Kode stabil untuk klien. Sengaja kasar: cukup untuk memutuskan tindakan
# (coba lagi, perbaiki permintaan, hubungi admin) tanpa memerikan isi sistem.
CODE_VALIDATION = "permintaan_tidak_valid"
CODE_UNAUTHORIZED = "tidak_terautentikasi"
CODE_FORBIDDEN = "akses_ditolak"
CODE_NOT_FOUND = "tidak_ditemukan"
CODE_RATE_LIMITED = "terlalu_banyak_permintaan"
CODE_TOO_LARGE = "muatan_terlalu_besar"
CODE_INTERNAL = "galat_internal"

_STATUS_CODES: dict[int, str] = {
    400: CODE_VALIDATION,
    401: CODE_UNAUTHORIZED,
    403: CODE_FORBIDDEN,
    404: CODE_NOT_FOUND,
    413: CODE_TOO_LARGE,
    422: CODE_VALIDATION,
    429: CODE_RATE_LIMITED,
}


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _body(
    code: str, message: str, request: Request, extra: dict | None = None,
) -> dict:
    payload = {"error": code, "detail": message, "request_id": _rid(request)}
    if extra:
        payload.update(extra)
    return payload


def install(app: FastAPI) -> None:
    """Pasang seluruh penangan galat pada aplikasi."""

    @app.exception_handler(TenantScopeError)
    async def _tenant_scope(request: Request, exc: TenantScopeError) -> JSONResponse:
        # Sampai di sini berarti ada jalur kode yang memanggil penyimpanan tanpa
        # tenant — cacat pemrograman, bukan kesalahan klien. Dicatat sebagai
        # galat agar terlihat, dan dibalas 500 tanpa menyebut sebabnya.
        logger.error("Operasi tanpa tenant ditolak: {}", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body(CODE_INTERNAL, "Terjadi galat internal", request),
        )

    @app.exception_handler(TenantPermissionError)
    async def _tenant_permission(
        request: Request, exc: TenantPermissionError,
    ) -> JSONResponse:
        logger.warning("Akses ditolak: {}", exc)
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=_body(CODE_FORBIDDEN, "Akses ditolak", request),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        # Hanya lokasi dan jenis kesalahannya yang dikembalikan. `msg` bawaan
        # Pydantic kerap memuat kembali nilai yang dikirim — tidak dipakai.
        masalah = [
            {
                "field": ".".join(str(x) for x in e.get("loc", ())[1:]) or "body",
                "type": e.get("type", "invalid"),
            }
            for e in exc.errors()[:20]
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body(
                CODE_VALIDATION, "Permintaan tidak valid", request,
                {"fields": masalah},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        kode = _STATUS_CODES.get(exc.status_code, CODE_INTERNAL)
        # 5xx yang dilempar sebagai HTTPException pun tidak boleh membawa
        # rinciannya keluar — pesannya sering berisi galat pustaka apa adanya.
        pesan = (
            "Terjadi galat internal"
            if exc.status_code >= 500
            else str(exc.detail)
        )
        if exc.status_code >= 500:
            logger.error("HTTP {} pada {}: {}", exc.status_code, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(kode, pesan, request),
            headers=getattr(exc, "headers", None) or None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # `logger.exception` menulis jejak tumpukan ke LOG SERVER; balasannya
        # tetap tidak memuat apa pun tentang isi dalam sistem.
        logger.exception(
            "Galat tak tertangani pada {} {}", request.method, request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body(CODE_INTERNAL, "Terjadi galat internal", request),
        )
