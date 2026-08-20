"""Pendaftaran tenant dan kunci API-nya.

Satu berkas JSON per tenant di `data/tenants/`, mengikuti pola yang sudah dipakai
`conversation_store`: mengubah satu tenant tidak perlu menulis ulang seluruh
daftar, sehingga dua perubahan bersamaan tidak saling menimpa.

Yang disimpan hanya HASH kunci, tidak pernah kuncinya sendiri. Bocornya seluruh
isi direktori ini tidak memberi penyerang satu pun kunci yang bisa dipakai.

Skala: pencarian kunci memakai indeks di memori yang dibangun dari berkas-berkas
ini dan disegarkan saat direktori berubah. Cukup untuk ratusan tenant. Di atas
itu — atau begitu ada beberapa proses server — pindahkan ke tabel basis data:
antarmuka modul ini (`find_by_key_id`, `get`, `save`) sengaja dibuat sempit agar
penggantinya cukup mengganti isi fungsi, bukan seluruh pemanggilnya.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.security import keys as key_utils
from src.tenancy import (
    ALL_SCOPES,
    READER_SCOPES,
    TenantContext,
    TenantQuota,
    is_valid_tenant_id,
    require_tenant_id,
)
from src.utils.logger import logger

TENANT_DIR = PROJECT_ROOT / "data" / "tenants"

STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class ApiKeyRecord:
    """Satu kunci API milik sebuah tenant."""
    key_id: str
    secret_hash: str
    label: str = ""
    scopes: list[str] = field(default_factory=lambda: sorted(READER_SCOPES))
    # ABAC: kosong/None berarti seluruh mata kuliah milik tenant.
    allowed_courses: list[str] | None = None
    created_at: str = field(default_factory=_now)
    revoked_at: str | None = None
    last_used_at: str | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "secret_hash": self.secret_hash,
            "label": self.label,
            "scopes": self.scopes,
            "allowed_courses": self.allowed_courses,
            "created_at": self.created_at,
            "revoked_at": self.revoked_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ApiKeyRecord:
        return cls(
            key_id=d["key_id"],
            secret_hash=d["secret_hash"],
            label=d.get("label", ""),
            scopes=list(d.get("scopes") or sorted(READER_SCOPES)),
            allowed_courses=(
                list(d["allowed_courses"]) if d.get("allowed_courses") else None
            ),
            created_at=d.get("created_at") or _now(),
            revoked_at=d.get("revoked_at"),
            last_used_at=d.get("last_used_at"),
        )


@dataclass
class Tenant:
    tenant_id: str
    name: str = ""
    status: str = STATUS_ACTIVE
    quota: TenantQuota = field(default_factory=TenantQuota)
    api_keys: list[ApiKeyRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    @property
    def is_active(self) -> bool:
        return self.status == STATUS_ACTIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status,
            "quota": self.quota.as_dict(),
            "api_keys": [k.as_dict() for k in self.api_keys],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tenant:
        kuota = d.get("quota") or {}
        return cls(
            tenant_id=d["tenant_id"],
            name=d.get("name", ""),
            status=d.get("status", STATUS_ACTIVE),
            quota=TenantQuota(
                requests_per_minute=int(kuota.get("requests_per_minute", 60)),
                requests_per_day=int(kuota.get("requests_per_day", 5_000)),
                max_upload_mb=int(kuota.get("max_upload_mb", 100)),
                max_storage_mb=int(kuota.get("max_storage_mb", 20_000)),
            ),
            api_keys=[ApiKeyRecord.from_dict(k) for k in (d.get("api_keys") or [])],
            created_at=d.get("created_at") or _now(),
        )


# --- indeks di memori -------------------------------------------------------
# Dibangun dari berkas dan disegarkan saat direktori berubah. Menghindari
# membaca seluruh berkas tenant pada setiap permintaan API.
_cache: dict[str, Tenant] = {}
_key_index: dict[str, tuple[str, ApiKeyRecord]] = {}
_cache_stamp: float = -1.0


def _dir_stamp() -> float:
    """Penanda perubahan direktori tenant; -1 bila direktori belum ada."""
    if not TENANT_DIR.exists():
        return -1.0
    try:
        return max(
            [TENANT_DIR.stat().st_mtime]
            + [p.stat().st_mtime for p in TENANT_DIR.glob("*.json")]
        )
    except OSError:
        return -1.0


def _refresh(force: bool = False) -> None:
    global _cache_stamp
    stamp = _dir_stamp()
    if not force and stamp == _cache_stamp and _cache_stamp != -1.0:
        return

    _cache.clear()
    _key_index.clear()
    if TENANT_DIR.exists():
        for path in sorted(TENANT_DIR.glob("*.json")):
            try:
                tenant = Tenant.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                # Satu berkas rusak tidak boleh mematikan autentikasi seluruh
                # tenant lain — dilewati dengan catatan, bukan dilempar.
                logger.error("Berkas tenant {} tidak terbaca: {}", path.name, exc)
                continue
            if not is_valid_tenant_id(tenant.tenant_id):
                logger.error("tenant_id tidak sah di {}, dilewati", path.name)
                continue
            _cache[tenant.tenant_id] = tenant
            for k in tenant.api_keys:
                if k.is_active:
                    _key_index[k.key_id] = (tenant.tenant_id, k)
    _cache_stamp = stamp


def invalidate_cache() -> None:
    """Paksa muat ulang pada akses berikutnya (dipakai tes dan CLI)."""
    global _cache_stamp
    _cache_stamp = -1.0


def _path_for(tenant_id: str) -> Path:
    return TENANT_DIR / f"{require_tenant_id(tenant_id, operation='tenant_store')}.json"


def save(tenant: Tenant) -> None:
    """Tulis satu tenant secara atomik."""
    require_tenant_id(tenant.tenant_id, operation="tenant_store.save")
    TENANT_DIR.mkdir(parents=True, exist_ok=True)
    path = _path_for(tenant.tenant_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(tenant.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8",
    )
    tmp.replace(path)
    invalidate_cache()


def get(tenant_id: str) -> Tenant | None:
    if not is_valid_tenant_id(tenant_id):
        return None
    _refresh()
    return _cache.get(tenant_id)


def list_tenants() -> list[Tenant]:
    _refresh()
    return sorted(_cache.values(), key=lambda t: t.tenant_id)


def count() -> int:
    _refresh()
    return len(_cache)


def find_by_key_id(key_id: str) -> tuple[Tenant, ApiKeyRecord] | None:
    """Cari tenant + catatan kunci dari `key_id`; None bila tidak ada/dicabut."""
    _refresh()
    hit = _key_index.get(key_id)
    if hit is None:
        return None
    tenant_id, record = hit
    tenant = _cache.get(tenant_id)
    if tenant is None or not record.is_active:
        return None
    return tenant, record


def authenticate(raw_key: str | None) -> TenantContext | None:
    """Ubah kunci API mentah menjadi `TenantContext`; None bila tidak sah.

    Inilah satu-satunya tempat `tenant_id` lahir. Tidak ada jalur lain yang boleh
    membuatnya dari masukan klien — kalau ada, seluruh isolasi runtuh.
    """
    terurai = key_utils.parse_key(raw_key)
    if terurai is None:
        return None
    key_id, secret = terurai

    hit = find_by_key_id(key_id)
    if hit is None:
        # Tetap jalankan satu perhitungan hash walau kunci tidak ditemukan,
        # supaya lama proses untuk key_id yang ada dan yang tidak ada serupa dan
        # tidak bisa dipakai memetakan kunci mana yang valid.
        key_utils.hash_secret(secret)
        return None

    tenant, record = hit
    if not key_utils.verify_secret(secret, record.secret_hash):
        return None
    if not tenant.is_active:
        return None

    return TenantContext(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        scopes=frozenset(record.scopes),
        quota=tenant.quota,
        allowed_courses=(
            frozenset(record.allowed_courses) if record.allowed_courses else None
        ),
        key_id=record.key_id,
    )


# --- penyediaan tenant (dipakai CLI `scripts/tenantctl.py`) ------------------

def create_tenant(
    tenant_id: str,
    name: str = "",
    quota: TenantQuota | None = None,
) -> Tenant:
    require_tenant_id(tenant_id, operation="create_tenant")
    if get(tenant_id) is not None:
        raise ValueError(f"Tenant '{tenant_id}' sudah ada")
    tenant = Tenant(
        tenant_id=tenant_id, name=name or tenant_id, quota=quota or TenantQuota(),
    )
    save(tenant)
    logger.info("Tenant dibuat: {}", tenant_id)
    return tenant


def issue_key(
    tenant_id: str,
    label: str = "",
    scopes: set[str] | None = None,
    allowed_courses: list[str] | None = None,
) -> key_utils.IssuedKey:
    """Terbitkan kunci baru untuk sebuah tenant.

    Nilai mentah kunci HANYA dikembalikan di sini dan tidak pernah disimpan —
    kalau hilang, terbitkan kunci baru dan cabut yang lama.
    """
    tenant = get(tenant_id)
    if tenant is None:
        raise ValueError(f"Tenant '{tenant_id}' tidak ditemukan")

    diminta = set(scopes) if scopes else set(READER_SCOPES)
    tak_dikenal = diminta - ALL_SCOPES
    if tak_dikenal:
        raise ValueError(f"Hak tidak dikenal: {sorted(tak_dikenal)}")

    diterbitkan = key_utils.generate_key()
    tenant.api_keys.append(
        ApiKeyRecord(
            key_id=diterbitkan.key_id,
            secret_hash=diterbitkan.secret_hash,
            label=label,
            scopes=sorted(diminta),
            allowed_courses=list(allowed_courses) if allowed_courses else None,
        )
    )
    save(tenant)
    logger.info("Kunci diterbitkan untuk tenant {} (key_id={})", tenant_id, diterbitkan.key_id)
    return diterbitkan


def revoke_key(tenant_id: str, key_id: str) -> bool:
    """Cabut satu kunci. Catatannya disimpan, tidak dihapus, agar jejak audit utuh."""
    tenant = get(tenant_id)
    if tenant is None:
        return False
    for k in tenant.api_keys:
        if k.key_id == key_id and k.is_active:
            k.revoked_at = _now()
            save(tenant)
            logger.info("Kunci dicabut: tenant={} key_id={}", tenant_id, key_id)
            return True
    return False


def set_status(tenant_id: str, status: str) -> bool:
    """Aktifkan atau bekukan seluruh akses satu tenant sekaligus."""
    if status not in (STATUS_ACTIVE, STATUS_SUSPENDED):
        raise ValueError(f"Status tidak dikenal: {status}")
    tenant = get(tenant_id)
    if tenant is None:
        return False
    tenant.status = status
    save(tenant)
    logger.info("Status tenant {} → {}", tenant_id, status)
    return True
