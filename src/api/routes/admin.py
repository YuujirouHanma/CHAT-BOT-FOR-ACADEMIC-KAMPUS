"""Pengelolaan tenant lewat HTTP — endpoint paling berbahaya di layanan ini.

    GET    /admin/tenants                      daftar tenant
    POST   /admin/tenants                      buat tenant baru
    GET    /admin/tenants/{id}                 rincian satu tenant
    POST   /admin/tenants/{id}/keys            terbitkan kunci baru
    DELETE /admin/tenants/{id}/keys/{key_id}   cabut kunci
    POST   /admin/tenants/{id}/status          bekukan / aktifkan
    DELETE /admin/tenants/{id}/data            hapus SELURUH data tenant

Tiga pengaman, dan ketiganya perlu:

1. **Mati secara bawaan.** `ENABLE_ADMIN_API=false` sampai dinyalakan sengaja.
   Endpoint ini dapat menerbitkan kunci untuk tenant MANA PUN — kebocorannya
   bukan berarti satu pelanggan terbuka, melainkan semuanya sekaligus.
2. **Hak `admin:tenants`**, yang tidak pernah ada di kunci aplikasi mahasiswa
   maupun kunci dosen.
3. **Setiap tindakan masuk jejak audit**, termasuk yang gagal.

Anjuran penempatan: jalankan pada instance TERPISAH yang tidak terjangkau dari
internet, bukan pada instance yang sama dengan yang melayani mahasiswa. Selama
pelanggan masih sedikit, `scripts/tenantctl.py` lebih aman daripada ini —
permukaan serangnya nol.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from src.api.auth import client_ip, request_id, tenant_admin
from src.api.dependencies import get_pipeline
from src.api.schemas import (
    AdminIssuedKeyResponse,
    AdminIssueKeyRequest,
    AdminPurgeResponse,
    AdminStatusRequest,
    AdminTenantCreateRequest,
    AdminTenantDetail,
    AdminTenantKeyInfo,
    AdminTenantListResponse,
    AdminTenantSummary,
)
from src.pipeline import RAGPipeline
from src.security import audit
from src.storage import conversation_store, tenant_store
from src.tenancy import TenantContext, TenantQuota, is_valid_tenant_id

router = APIRouter(prefix="/admin", tags=["admin"])

_TENANT_PATH = Path(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")


def _audit(
    event: str, request: Request, actor: TenantContext,
    target: str, outcome: str = "ok", **detail,
) -> None:
    audit.record(
        event, tenant_id=actor.tenant_id, actor=actor.key_id, outcome=outcome,
        request_id=request_id(request), ip=client_ip(request),
        detail={"target_tenant": target, **detail},
    )


def _summary(t: tenant_store.Tenant) -> AdminTenantSummary:
    return AdminTenantSummary(
        tenant_id=t.tenant_id,
        name=t.name,
        status=t.status,
        created_at=t.created_at,
        active_keys=sum(1 for k in t.api_keys if k.is_active),
    )


def _require_tenant(tenant_id: str) -> tenant_store.Tenant:
    t = tenant_store.get(tenant_id)
    if t is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' tidak ditemukan",
        )
    return t


@router.get("/tenants", response_model=AdminTenantListResponse)
async def list_tenants(
    admin: TenantContext = Depends(tenant_admin),
) -> AdminTenantListResponse:
    daftar = tenant_store.list_tenants()
    return AdminTenantListResponse(
        total=len(daftar), tenants=[_summary(t) for t in daftar],
    )


@router.get("/tenants/{tenant_id}", response_model=AdminTenantDetail)
async def get_tenant(
    tenant_id: str = _TENANT_PATH,
    admin: TenantContext = Depends(tenant_admin),
) -> AdminTenantDetail:
    """Rincian tenant beserta daftar kuncinya.

    Yang ditampilkan hanya `key_id` dan metadatanya. Rahasia kuncinya tidak
    pernah tersimpan dalam bentuk yang dapat dikembalikan — hanya hash-nya.
    """
    t = _require_tenant(tenant_id)
    return AdminTenantDetail(
        **_summary(t).model_dump(),
        quota=t.quota.as_dict(),
        keys=[
            AdminTenantKeyInfo(
                key_id=k.key_id, label=k.label, scopes=k.scopes,
                allowed_courses=k.allowed_courses, created_at=k.created_at,
                revoked_at=k.revoked_at, last_used_at=k.last_used_at,
            )
            for k in t.api_keys
        ],
    )


@router.post(
    "/tenants", response_model=AdminTenantDetail,
    status_code=status.HTTP_201_CREATED,
)
async def create_tenant(
    body: AdminTenantCreateRequest,
    request: Request,
    admin: TenantContext = Depends(tenant_admin),
) -> AdminTenantDetail:
    if not is_valid_tenant_id(body.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id hanya boleh huruf kecil, angka, strip, garis bawah",
        )
    try:
        t = tenant_store.create_tenant(
            body.tenant_id, body.name or body.tenant_id,
            quota=TenantQuota(**body.quota) if body.quota else None,
        )
    except (ValueError, TypeError) as exc:
        _audit("admin.tenant_create", request, admin, body.tenant_id, "gagal")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    _audit("admin.tenant_create", request, admin, t.tenant_id)
    return AdminTenantDetail(
        **_summary(t).model_dump(), quota=t.quota.as_dict(), keys=[],
    )


@router.post(
    "/tenants/{tenant_id}/keys", response_model=AdminIssuedKeyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def issue_key(
    body: AdminIssueKeyRequest,
    request: Request,
    tenant_id: str = _TENANT_PATH,
    admin: TenantContext = Depends(tenant_admin),
) -> AdminIssuedKeyResponse:
    """Terbitkan kunci baru.

    Nilai mentah kunci HANYA muncul di balasan ini dan tidak pernah tersimpan.
    Hilang berarti harus diterbitkan yang baru dan yang lama dicabut.
    """
    _require_tenant(tenant_id)
    try:
        diterbitkan = tenant_store.issue_key(
            tenant_id, label=body.label or "",
            scopes=set(body.scopes) if body.scopes else None,
            allowed_courses=body.allowed_courses,
        )
    except ValueError as exc:
        _audit("admin.key_issue", request, admin, tenant_id, "gagal")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc

    _audit("admin.key_issue", request, admin, tenant_id, key_issued=diterbitkan.key_id)
    return AdminIssuedKeyResponse(
        tenant_id=tenant_id,
        key_id=diterbitkan.key_id,
        api_key=diterbitkan.raw,
        peringatan=(
            "Salin sekarang — kunci ini tidak akan ditampilkan lagi. "
            "Yang tersimpan di server hanya hash-nya."
        ),
    )


@router.delete(
    "/tenants/{tenant_id}/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT, response_model=None,
)
async def revoke_key(
    request: Request,
    tenant_id: str = _TENANT_PATH,
    key_id: str = Path(min_length=4, max_length=64, pattern=r"^[0-9a-f]+$"),
    admin: TenantContext = Depends(tenant_admin),
) -> None:
    """Cabut satu kunci. Catatannya disimpan, tidak dihapus, agar audit utuh."""
    _require_tenant(tenant_id)
    if not tenant_store.revoke_key(tenant_id, key_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kunci tidak ditemukan atau sudah dicabut",
        )
    _audit("admin.key_revoke", request, admin, tenant_id, key_revoked=key_id)


@router.post("/tenants/{tenant_id}/status", response_model=AdminTenantSummary)
async def set_status(
    body: AdminStatusRequest,
    request: Request,
    tenant_id: str = _TENANT_PATH,
    admin: TenantContext = Depends(tenant_admin),
) -> AdminTenantSummary:
    """Bekukan atau aktifkan seluruh akses satu tenant sekaligus.

    Pembekuan berlaku seketika untuk SEMUA kuncinya — dipakai saat langganan
    berakhir, tanpa perlu mencabut kunci satu per satu.
    """
    _require_tenant(tenant_id)
    try:
        tenant_store.set_status(tenant_id, body.status)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    _audit("admin.tenant_status", request, admin, tenant_id, status=body.status)
    return _summary(_require_tenant(tenant_id))


@router.delete("/tenants/{tenant_id}/data", response_model=AdminPurgeResponse)
async def purge_data(
    request: Request,
    tenant_id: str = _TENANT_PATH,
    admin: TenantContext = Depends(tenant_admin),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AdminPurgeResponse:
    """Hapus SELURUH data terindeks dan percakapan milik satu tenant.

    Tidak dapat dibatalkan. Ada karena hak untuk dihapus bukan sesuatu yang bisa
    dikerjakan belakangan dengan skrip manual saat pelanggan memintanya.

    Catatan tenant dan jejak auditnya sengaja TIDAK ikut terhapus: keduanya
    adalah bukti bahwa penghapusan itu benar dilakukan, oleh siapa, dan kapan.
    """
    _require_tenant(tenant_id)
    await pipeline._store.delete_tenant_data(tenant_id=tenant_id)
    percakapan = conversation_store.delete_all_for_tenant(tenant_id=tenant_id)
    _audit(
        "admin.tenant_purge", request, admin, tenant_id,
        conversations_deleted=percakapan,
    )
    return AdminPurgeResponse(
        tenant_id=tenant_id,
        conversations_deleted=percakapan,
        catatan="Data index dan percakapan dihapus. Catatan tenant dan jejak "
                "audit sengaja dipertahankan sebagai bukti penghapusan.",
    )
