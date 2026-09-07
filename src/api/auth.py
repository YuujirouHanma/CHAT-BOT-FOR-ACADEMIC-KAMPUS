"""Autentikasi dan otorisasi per tenant.

Satu aturan yang menopang seluruh isolasi data:

    `tenant_id` HANYA lahir dari kunci API. Tidak pernah dari body permintaan.

Klien boleh ikut menyertakan `tenant_id` di body; nilainya dicocokkan dengan yang
diturunkan dari kunci dan ketidakcocokan ditolak (403). Yang dicocokkan itu
bukan sumber kebenaran — ia hanya alat mendeteksi salah pasang kunci di sisi
pemanggil, sekaligus menangkap percobaan membaca data tenant lain.

Urutan pemeriksaan disusun dari yang paling murah ke yang paling mahal, sehingga
lalu lintas sampah tersaring sebelum menyentuh penyimpanan:

    1. batas laju per IP        → menahan banjir permintaan tanpa kredensial
    2. perlambatan progresif    → menahan penebakan kunci secara sistematis
    3. verifikasi kunci         → HMAC waktu-tetap terhadap hash tersimpan
    4. batas laju kunci/tenant  → kuota komersial
    5. pemeriksaan hak (RBAC)   → apakah kunci ini boleh melakukannya
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings
from src.security import audit
from src.security.ratelimit import auth_throttle, limiter
from src.storage import tenant_store
from src.tenancy import (
    ALL_SCOPES,
    SCOPE_ADMIN,
    SCOPE_CATALOG_READ,
    SCOPE_CHAT,
    SCOPE_CONTENT_READ,
    SCOPE_CONTENT_WRITE,
    SCOPE_CONVERSATION_DELETE,
    SCOPE_CONVERSATION_READ,
    SCOPE_VALIDATE,
    TenantContext,
    TenantQuota,
)
from src.utils.logger import logger

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Pesan galat sengaja seragam dan tidak informatif. Membedakan "kunci tidak
# ditemukan" dari "kunci salah" memberi tahu penyerang bahwa sebuah key_id
# memang ada — cukup untuk mempersempit pencariannya.
_UNAUTHORIZED = "Kredensial tidak sah"


def client_ip(request: Request) -> str:
    """Alamat asal permintaan.

    `X-Forwarded-For` hanya dibaca bila `trust_proxy_headers` dinyalakan. Tanpa
    proxy tepercaya di depan, header itu sepenuhnya dikendalikan pemanggil —
    mempercayainya membuat pembatasan laju per IP dapat dilewati hanya dengan
    mengarang nilai baru setiap permintaan.
    """
    if settings.trust_proxy_headers:
        maju = request.headers.get("x-forwarded-for", "")
        if maju:
            return maju.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or uuid.uuid4().hex[:16]


def _fail(detail: str, code: int, headers: dict[str, str] | None = None) -> HTTPException:
    return HTTPException(status_code=code, detail=detail, headers=headers or {})


def _dev_tenant() -> TenantContext:
    """Tenant sementara untuk pengembangan lokal, saat belum ada tenant terdaftar.

    Menjaga pengalaman `uvicorn ... --reload` tanpa penyiapan apa pun. Di produksi
    kondisi ini menggagalkan startup (lihat `src/api/main.py`), jadi ia tidak
    mungkin terbawa tanpa sengaja.
    """
    return TenantContext(
        tenant_id=settings.dev_anonymous_tenant,
        name="Pengembangan lokal",
        # Seluruh hak KECUALI pengelolaan tenant. Tenant dadakan ini lahir tanpa
        # kredensial apa pun; memberinya hak menerbitkan kunci berarti siapa pun
        # yang menjangkau port ini dapat membuat kunci untuk tenant mana pun —
        # tepat pada pemasangan yang paling mungkin lupa dikonfigurasi.
        scopes=frozenset(ALL_SCOPES - {SCOPE_ADMIN}),
        quota=TenantQuota(),
        key_id="dev",
    )


def _legacy_tenant() -> TenantContext:
    """Pemetaan kunci global warisan ke satu tenant tetap.

    Ada semata agar integrasi tim BE yang sudah berjalan tidak putus di hari
    peralihan. Tidak tersedia di produksi — kunci yang sama untuk semua pemanggil
    tidak dapat membedakan siapa pun, sehingga isolasi tidak mungkin ditegakkan.
    """
    return TenantContext(
        tenant_id=settings.legacy_tenant_id,
        name="Kunci warisan",
        scopes=frozenset(ALL_SCOPES),
        quota=TenantQuota(),
        key_id="legacy",
    )


async def require_tenant(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> TenantContext:
    """Selesaikan identitas tenant untuk satu permintaan, atau tolak."""
    ip = client_ip(request)
    rid = request_id(request)

    # 1 — batas laju per IP, sebelum menyentuh penyimpanan apa pun.
    if settings.rate_limit_enabled:
        putusan = limiter.check(
            ip=ip, key_id=None, tenant_id=None,
            per_minute=0, per_day=0,
            ip_per_minute=settings.rate_limit_ip_per_minute,
        )
        if not putusan.allowed:
            audit.record(
                audit.RATE_LIMITED, outcome="ditolak", request_id=rid, ip=ip,
                detail={"scope": putusan.scope},
            )
            raise _fail(
                "Terlalu banyak permintaan", status.HTTP_429_TOO_MANY_REQUESTS,
                {"Retry-After": str(putusan.retry_after)},
            )

    # 2 — perlambatan progresif terhadap penebakan kunci.
    #
    # Hukumannya dihitung SEKARANG tetapi diterapkan SETELAH kunci diverifikasi,
    # bukan sebelumnya. Menolak lebih dulu terlihat lebih aman, tetapi membuat
    # kunci yang SAH pun ikut ditolak — dan karena penolakan itu terjadi sebelum
    # verifikasi, tidak ada lagi jalan mencatat keberhasilan yang menghapus
    # hukumannya. Alamat itu terkunci penuh 15 menit. Di balik NAT atau egress
    # bersama — persis keadaan kampus — satu klien yang salah pasang kunci akan
    # mematikan seluruh klien sah di gedung yang sama.
    #
    # Menunda penerapannya tidak melemahkan pertahanan: penebak tidak pernah
    # memegang kunci sah, jadi ia tetap terhukum, sementara laju percobaannya
    # sudah dibatasi lapis per-IP di atas.
    terkunci, tunggu = auth_throttle.penalty(ip)

    tenant = _resolve(
        api_key, ip=ip, rid=rid, terkunci=terkunci, tunggu=tunggu,
    )

    # 4 — kuota kunci & tenant.
    if settings.rate_limit_enabled:
        putusan = limiter.check(
            ip=None, key_id=tenant.key_id, tenant_id=tenant.tenant_id,
            per_minute=tenant.quota.requests_per_minute,
            per_day=tenant.quota.requests_per_day,
            ip_per_minute=0,
        )
        if not putusan.allowed:
            audit.record(
                audit.RATE_LIMITED, tenant_id=tenant.tenant_id, actor=tenant.key_id,
                outcome="ditolak", request_id=rid, ip=ip,
                detail={"scope": putusan.scope},
            )
            raise _fail(
                "Kuota pemakaian terlampaui", status.HTTP_429_TOO_MANY_REQUESTS,
                {"Retry-After": str(putusan.retry_after)},
            )

    request.state.tenant = tenant
    return tenant


def _resolve(
    api_key: str | None,
    *,
    ip: str,
    rid: str,
    terkunci: bool = False,
    tunggu: int = 0,
) -> TenantContext:
    """Ubah header kunci menjadi tenant, atau lempar 401/429.

    `terkunci`/`tunggu` adalah hukuman perlambatan yang sudah dihitung pemanggil.
    Keduanya hanya berlaku bagi kunci yang GAGAL — kunci sah selalu diloloskan
    dan sekaligus menghapus catatan kegagalan alamat itu.
    """
    def _tolak() -> HTTPException:
        """Kegagalan autentikasi: catat, lalu tolak dengan hukuman bila ada."""
        auth_throttle.record_failure(ip)
        if terkunci or tunggu:
            audit.record(
                audit.AUTH_FAILURE, outcome="dilambatkan", request_id=rid, ip=ip,
                detail={"locked": terkunci, "retry_after": tunggu},
            )
            return _fail(
                _UNAUTHORIZED, status.HTTP_429_TOO_MANY_REQUESTS,
                {"Retry-After": str(tunggu or 1)},
            )
        return _fail(_UNAUTHORIZED, status.HTTP_401_UNAUTHORIZED)

    # Jalur warisan: satu kunci global tanpa identitas tenant.
    warisan = settings.ragacademic_api_key.get_secret_value()
    if warisan and api_key == warisan and not settings.is_production:
        auth_throttle.record_success(ip)
        return _legacy_tenant()

    if api_key:
        tenant = tenant_store.authenticate(api_key)
        if tenant is not None:
            auth_throttle.record_success(ip)
            audit.record(
                audit.AUTH_SUCCESS, tenant_id=tenant.tenant_id, actor=tenant.key_id,
                request_id=rid, ip=ip,
            )
            return tenant
        audit.record(audit.AUTH_FAILURE, outcome="kunci_salah", request_id=rid, ip=ip)
        raise _tolak()

    # Tanpa kunci sama sekali: hanya boleh di pengembangan, dan hanya selama
    # belum ada satu pun tenant terdaftar.
    if not settings.is_production and tenant_store.count() == 0 and not warisan:
        return _dev_tenant()

    audit.record(audit.AUTH_FAILURE, outcome="tanpa_kunci", request_id=rid, ip=ip)
    raise _tolak()


def assert_body_tenant(
    tenant: TenantContext, body_tenant_id: str | None, *, request_id_: str = "",
) -> None:
    """Tolak bila `tenant_id` di body berbeda dari yang diturunkan dari kunci.

    Nilai di body TIDAK PERNAH dipakai untuk menentukan cakupan data — hanya
    dicocokkan. Ketidakcocokan hampir selalu berarti kunci tertukar di sisi
    pemanggil, dan sisanya berarti percobaan membaca data tenant lain; keduanya
    harus berhenti di sini.
    """
    if not body_tenant_id or not settings.enforce_tenant_body_match:
        return
    if body_tenant_id != tenant.tenant_id:
        audit.record(
            audit.TENANT_MISMATCH, tenant_id=tenant.tenant_id, actor=tenant.key_id,
            outcome="ditolak", request_id=request_id_,
            detail={"diminta": body_tenant_id},
        )
        logger.warning(
            "tenant_id di body ({}) tidak cocok dengan kunci ({})",
            body_tenant_id, tenant.tenant_id,
        )
        raise _fail(
            "tenant_id pada body tidak cocok dengan kredensial",
            status.HTTP_403_FORBIDDEN,
        )


def require_scope(scope: str) -> Callable[..., Awaitable[TenantContext]]:
    """Dependency yang menuntut satu hak tertentu.

    Pemeriksaan hak melekat pada tiap rute, bukan hanya pada gerbang API. Rute
    baru yang lupa memasangnya akan gagal karena tidak punya tenant sama sekali,
    bukan diam-diam terbuka untuk semua kunci.
    """

    async def dependency(
        request: Request,
        tenant: TenantContext = Depends(require_tenant),
    ) -> TenantContext:
        if not tenant.has(scope):
            audit.record(
                audit.AUTH_FORBIDDEN, tenant_id=tenant.tenant_id, actor=tenant.key_id,
                outcome="ditolak", request_id=request_id(request),
                ip=client_ip(request), detail={"scope": scope},
            )
            raise _fail("Akses ditolak", status.HTTP_403_FORBIDDEN)
        return tenant

    return dependency


# Dependency siap pakai per jenis operasi.
tenant_chat = require_scope(SCOPE_CHAT)
tenant_catalog = require_scope(SCOPE_CATALOG_READ)
tenant_content_read = require_scope(SCOPE_CONTENT_READ)
tenant_content_write = require_scope(SCOPE_CONTENT_WRITE)
tenant_conversation_read = require_scope(SCOPE_CONVERSATION_READ)
tenant_conversation_delete = require_scope(SCOPE_CONVERSATION_DELETE)
tenant_validate = require_scope(SCOPE_VALIDATE)
tenant_admin = require_scope(SCOPE_ADMIN)
