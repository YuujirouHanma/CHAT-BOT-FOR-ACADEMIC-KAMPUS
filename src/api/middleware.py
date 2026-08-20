"""Middleware HTTP: penanda permintaan, batas ukuran, dan header pengerasan.

Ketiganya berlaku untuk SETIAP permintaan, termasuk rute yang belum ada saat
berkas ini ditulis. Menaruhnya di middleware — bukan mengulanginya di tiap rute —
membuat rute baru ikut terlindungi tanpa harus diingat.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings
from src.utils.logger import logger

_Next = Callable[[Request], Awaitable[Response]]

# Kebijakan konten yang ketat. API ini hanya membalas JSON, jadi tidak ada satu
# pun sumber daya yang perlu diizinkan — kalau ada balasan HTML yang lolos
# (halaman galat proxy, misalnya), skrip di dalamnya tetap tidak akan berjalan.
_CSP = (
    "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)

# Dokumentasi interaktif memuat aset dari CDN dan butuh kebijakan lebih longgar.
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
_CSP_DOCS = (
    "default-src 'self'; img-src 'self' data: https://fastapi.tiangolo.com; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "frame-ancestors 'none'; base-uri 'none'"
)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Beri setiap permintaan satu id, lalu catat hasilnya.

    Id itu ikut di setiap balasan galat dan di jejak audit, sehingga satu laporan
    pengguna ("request_id abc123 gagal") dapat ditelusuri tanpa perlu rincian
    galat ikut terkirim keluar.
    """

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
        # Hanya karakter aman: nilai ini dipantulkan kembali sebagai header.
        rid = "".join(c for c in rid if c.isalnum() or c in "-_") or uuid.uuid4().hex[:16]
        request.state.request_id = rid

        mulai = time.monotonic()
        response = await call_next(request)
        lama = (time.monotonic() - mulai) * 1000

        response.headers["X-Request-ID"] = rid
        tenant = getattr(request.state, "tenant", None)
        logger.info(
            "{} {} → {} ({:.0f}ms) rid={} tenant={}",
            request.method, request.url.path, response.status_code, lama, rid,
            getattr(tenant, "tenant_id", "-"),
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Tolak muatan yang terlalu besar sedini mungkin.

    Tanpa batas ini, satu permintaan besar cukup untuk menghabiskan memori proses
    — dan menaikkan biaya penyimpanan tanpa batas. Header `Content-Length`
    diperiksa lebih dulu supaya penolakan terjadi sebelum satu byte pun dibaca;
    permintaan tanpa header itu tetap dibatasi oleh server ASGI di depannya.
    """

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        panjang = request.headers.get("content-length")
        if panjang and panjang.isdigit():
            if int(panjang) > settings.max_request_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "muatan_terlalu_besar",
                        "detail": f"Ukuran maksimum {settings.max_request_body_mb} MB",
                        "request_id": getattr(request.state, "request_id", ""),
                    },
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Header pengerasan pada setiap balasan."""

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        response = await call_next(request)
        h = response.headers

        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        h.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        h.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        h.setdefault(
            "Content-Security-Policy",
            _CSP_DOCS if request.url.path in _DOCS_PATHS else _CSP,
        )
        # Balasan API memuat data satu tenant — tidak boleh disinggahi cache
        # bersama di jaringan perantara.
        h.setdefault("Cache-Control", "no-store")

        # HSTS hanya bermakna di atas TLS. Mengirimkannya pada koneksi polos
        # tidak berbahaya tetapi juga tidak berguna, jadi dibatasi ke produksi
        # tempat TLS memang dipasang di depan.
        if settings.enable_hsts and settings.is_production:
            h.setdefault(
                "Strict-Transport-Security",
                f"max-age={settings.hsts_max_age}; includeSubDomains",
            )
        return response
