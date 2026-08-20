"""Fixture bersama untuk seluruh test suite.

Setelah sistem menjadi multi-tenant, hampir setiap uji butuh dua hal yang sama:
sebuah `TenantContext` untuk dipakai memanggil lapisan penyimpanan, dan jaminan
bahwa uji itu tidak menyentuh data sungguhan di dalam repo.

Tiga kelompok fixture di sini, semuanya autouse:

1. **Singleton keamanan** (`limiter`, `auth_throttle`) — keduanya global dan
   menyimpan hitungan lintas permintaan. Tanpa direset, sebuah modul uji yang
   menembakkan banyak permintaan lewat TestClient akan menabrak batas 120/menit
   per IP, dan uji SESUDAHNYA yang gagal — bukan uji yang menyebabkannya.
2. **Direktori data** — jejak audit, percakapan, log HITL, dan cache generasi
   semuanya menulis ke `data/` di dalam repo. Dialihkan ke `tmp_path` supaya
   menjalankan pytest tidak meninggalkan berkas dan tidak mencemari uji lain.
3. **Direktori tenant** — `tenant_store` menyimpan tenant di memori dengan
   penanda mtime direktori, jadi selain mengalihkan direktorinya, cache-nya
   harus ikut dibatalkan agar tenant sisa uji sebelumnya tidak terbawa.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from src import gen_cache
from src.config import settings
from src.hitl import logger as hitl_logger
from src.security import audit
from src.security.ratelimit import auth_throttle, limiter
from src.storage import conversation_store, tenant_store
from src.tenancy import ALL_SCOPES, TenantContext, TenantQuota

# tenant_id harus cocok dengan pola ketat di src/tenancy.py: huruf kecil, angka,
# strip, garis bawah — tanpa titik dan garis miring.
TEST_TENANT_ID = "tenant-uji"


def make_tenant(
    tenant_id: str = TEST_TENANT_ID,
    *,
    scopes: set[str] | frozenset[str] | None = None,
    allowed_courses: list[str] | None = None,
    name: str = "Tenant Uji",
    key_id: str = "kunci-uji",
) -> TenantContext:
    """`TenantContext` siap pakai; semua hak diberikan kecuali diminta lain."""
    return TenantContext(
        tenant_id=tenant_id,
        name=name,
        scopes=frozenset(scopes) if scopes is not None else frozenset(ALL_SCOPES),
        quota=TenantQuota(),
        allowed_courses=frozenset(allowed_courses) if allowed_courses else None,
        key_id=key_id,
    )


@pytest.fixture
def tenant() -> TenantContext:
    return make_tenant()


@pytest.fixture(autouse=True)
def reset_security_singletons() -> Iterator[None]:
    """Kosongkan pembatas laju dan perlambatan auth di antara uji.

    Keduanya hidup di tingkat modul dan tidak dibuat ulang per uji, sehingga
    sisa hitungan satu uji akan menolak permintaan uji berikutnya.
    """
    limiter.reset()
    auth_throttle.reset()
    yield
    limiter.reset()
    auth_throttle.reset()


@pytest.fixture(autouse=True)
def isolated_data_dirs(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Alihkan seluruh penulisan ke disk ke direktori sementara."""
    monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(conversation_store, "CONVERSATION_DIR", tmp_path / "conversations")
    monkeypatch.setattr(hitl_logger, "HITL_DIR", tmp_path / "hitl_logs")
    monkeypatch.setattr(gen_cache, "_CACHE_DIR", tmp_path / "gen_cache")
    # `storage_path` = PROJECT_ROOT / storage_root; path absolut menang di
    # pathlib, jadi mengisinya dengan tmp_path memindahkan seluruh unggahan.
    monkeypatch.setattr(settings, "storage_root", str(tmp_path / "storage"))

    # Rantai MAC audit menyimpan mata rantai terakhir per cakupan di memori.
    # Tanpa dikosongkan, catatan pertama di direktori baru akan menunjuk ke
    # hash milik berkas uji sebelumnya yang sudah tidak ada.
    audit._tail_cache.clear()
    yield
    audit._tail_cache.clear()


@pytest.fixture(autouse=True)
def isolated_tenant_dir(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Direktori pendaftaran tenant sendiri, dengan cache memori dibatalkan."""
    monkeypatch.setattr(tenant_store, "TENANT_DIR", tmp_path / "tenants")
    tenant_store.invalidate_cache()
    yield
    tenant_store.invalidate_cache()


@pytest.fixture
def disable_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matikan pembatasan laju untuk modul uji yang menembak banyak permintaan.

    Dipakai uji yang memang tidak sedang menguji pembatasan laju itu sendiri.
    """
    monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)


# Seluruh dependency yang menghasilkan tenant. Dependency per-hak dibuat oleh
# `require_scope(...)` saat impor, jadi yang harus ditimpa adalah objek yang
# persis itu — bukan hasil pemanggilan `require_scope` yang baru.
def _auth_dependencies() -> tuple:
    from src.api import auth

    return (
        auth.require_tenant,
        auth.tenant_chat,
        auth.tenant_catalog,
        auth.tenant_content_read,
        auth.tenant_content_write,
        auth.tenant_conversation_read,
        auth.tenant_conversation_delete,
        auth.tenant_admin,
    )


@pytest.fixture
def override_tenant(
    tenant: TenantContext, disable_rate_limit: None,
) -> Iterator[TenantContext]:
    """Lewati autentikasi: setiap rute menerima `tenant` yang sama.

    Menimpa SELURUH dependency hak akses, bukan hanya `require_tenant`, supaya
    uji tidak diam-diam bergantung pada rantai dependency di dalam
    `require_scope` — dan supaya rute yang memasang hak lebih spesifik tetap
    terlayani.
    """
    from src.api.main import app

    for dep in _auth_dependencies():
        app.dependency_overrides[dep] = lambda t=tenant: t
    yield tenant
    for dep in _auth_dependencies():
        app.dependency_overrides.pop(dep, None)
