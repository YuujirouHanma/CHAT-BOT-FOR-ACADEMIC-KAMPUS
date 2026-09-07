"""Identitas tenant — fondasi isolasi data antar pelanggan.

Sebelumnya sistem ini menganggap hanya ada satu pemakai selamanya: satu koleksi
Qdrant, satu folder penyimpanan, satu API key. Begitu dijual ke lebih dari satu
kampus, asumsi itu menjadi kebocoran data.

Modul ini menyediakan satu konsep saja — `TenantContext` — dan aturannya:

    Tenant TIDAK PERNAH berasal dari body permintaan.

`tenant_id` selalu diturunkan dari kredensial (API key) di `src/security/keys.py`.
Klien boleh ikut mengirim `tenant_id` di body, tetapi itu hanya dicocokkan untuk
mendeteksi salah konfigurasi — bukan dipercaya. Kalau body dipercaya, tenant A
cukup menulis `tenant_id` milik tenant B untuk membaca seluruh datanya (kelas
kerentanan IDOR/BOLA), dan tidak ada lapisan lain yang akan menahannya.

Namespacing dipisah menurut jenis penyimpanan, bukan diseragamkan:

- Qdrant   → `tenant_id` sebagai field payload + filter WAJIB di setiap query.
             Lebih baik daripada menempelkannya ke `course_id`, karena nama
             mata kuliah tetap terbaca apa adanya di UI dan filternya bisa
             diindeks Qdrant sebagai kunci tenant (lihat QdrantStore).
- Berkas   → dipisah secara FISIK: `storage/{tenant_id}/{content_id}/`.
             Direktori terpisah membuat dua tenant yang kebetulan memakai
             `content_id` sama tidak mungkin saling menimpa berkas.
- Cache    → `tenant_id` ikut masuk kunci hash.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

# tenant_id ikut menjadi nama direktori, jadi polanya sengaja ketat: huruf kecil,
# angka, strip, garis bawah. Tidak ada titik dan garis miring, sehingga mustahil
# dipakai keluar dari direktori induk (path traversal) sekalipun nilainya berasal
# dari sumber yang salah.
_TENANT_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")

# Cakupan hak akses (RBAC). Dipakai di lapisan domain — bukan hanya di gateway —
# supaya endpoint baru tidak otomatis terbuka hanya karena lupa didaftarkan.
SCOPE_CHAT: Final = "chat:ask"
SCOPE_CATALOG_READ: Final = "catalog:read"
SCOPE_CONTENT_READ: Final = "content:read"
SCOPE_CONTENT_WRITE: Final = "content:write"
SCOPE_CONVERSATION_READ: Final = "conversation:read"
SCOPE_CONVERSATION_DELETE: Final = "conversation:delete"
SCOPE_VALIDATE: Final = "answer:validate"   # dosen menilai jawaban AI
SCOPE_ADMIN: Final = "admin:tenants"

ALL_SCOPES: Final = frozenset({
    SCOPE_CHAT,
    SCOPE_CATALOG_READ,
    SCOPE_CONTENT_READ,
    SCOPE_CONTENT_WRITE,
    SCOPE_CONVERSATION_READ,
    SCOPE_CONVERSATION_DELETE,
    SCOPE_VALIDATE,
    SCOPE_ADMIN,
})

# Hak untuk kunci "aplikasi mahasiswa": boleh bertanya dan membaca katalog,
# tidak boleh mengunggah materi apalagi mengelola tenant.
READER_SCOPES: Final = frozenset({
    SCOPE_CHAT,
    SCOPE_CATALOG_READ,
    SCOPE_CONTENT_READ,
    SCOPE_CONVERSATION_READ,
    SCOPE_CONVERSATION_DELETE,
})


class TenantScopeError(RuntimeError):
    """Operasi penyimpanan dipanggil tanpa tenant yang sah.

    Sengaja galat keras, bukan peringatan: satu pemanggilan tanpa tenant berarti
    kueri berjalan ke seluruh index lintas pelanggan. Lebih baik permintaan itu
    gagal terang-terangan daripada diam-diam mengembalikan data tenant lain.
    """


class TenantPermissionError(RuntimeError):
    """Tenant sah, tetapi tidak berhak atas operasi atau data yang diminta."""


def is_valid_tenant_id(tenant_id: str | None) -> bool:
    return bool(tenant_id and _TENANT_ID_RE.match(tenant_id))


def require_tenant_id(tenant_id: str | None, *, operation: str = "") -> str:
    """Pastikan `tenant_id` ada dan berbentuk sah; kalau tidak, gagalkan.

    Dipanggil di ambang setiap operasi penyimpanan. Ini jaring pengaman lapis
    kedua — lapis pertama adalah `tenant_id` sebagai argumen wajib (keyword-only
    tanpa nilai bawaan), sehingga lupa mengirimnya sudah menjadi TypeError di
    Python sebelum sampai ke sini.
    """
    if not is_valid_tenant_id(tenant_id):
        konteks = f" pada operasi '{operation}'" if operation else ""
        raise TenantScopeError(
            f"tenant_id wajib dan harus sah{konteks}; diterima: {tenant_id!r}. "
            "Tanpa ini kueri akan mencakup data seluruh tenant."
        )
    assert tenant_id is not None
    return tenant_id


@dataclass(frozen=True)
class TenantQuota:
    """Batas pemakaian per tenant.

    Ada karena setiap pertanyaan memanggil LLM berbayar: tanpa batas, satu tenant
    yang salah membuat perulangan bisa menghabiskan tagihan seluruh layanan.
    """
    requests_per_minute: int = 60
    requests_per_day: int = 5_000
    max_upload_mb: int = 100
    max_storage_mb: int = 20_000

    def as_dict(self) -> dict[str, int]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_day": self.requests_per_day,
            "max_upload_mb": self.max_upload_mb,
            "max_storage_mb": self.max_storage_mb,
        }


@dataclass(frozen=True)
class TenantContext:
    """Identitas pemanggil untuk satu permintaan.

    Dibuat HANYA oleh lapisan autentikasi setelah memverifikasi API key, lalu
    diteruskan secara eksplisit ke setiap lapisan di bawahnya. Sengaja tidak
    memakai variabel global / contextvar: kewenangan yang mengambang di
    lingkungan (*ambient authority*) persis yang membuat satu jalur kode terlupa
    lalu membaca data seluruh tenant tanpa ada yang menyadarinya.
    """
    tenant_id: str
    name: str = ""
    scopes: frozenset[str] = field(default_factory=lambda: READER_SCOPES)
    quota: TenantQuota = field(default_factory=TenantQuota)
    # ABAC: bila diisi, kunci ini hanya boleh menyentuh mata kuliah tertentu.
    # Berguna untuk kampus yang ingin memberi satu kunci per fakultas.
    allowed_courses: frozenset[str] | None = None
    # Dipakai di log audit untuk menelusuri kunci mana yang dipakai, tanpa
    # pernah menuliskan kuncinya sendiri.
    key_id: str = ""

    def __post_init__(self) -> None:
        require_tenant_id(self.tenant_id, operation="TenantContext")

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def require(self, scope: str) -> None:
        if not self.has(scope):
            raise TenantPermissionError(
                f"Kunci ini tidak memiliki hak '{scope}'"
            )

    def may_access_course(self, course_id: str | None) -> bool:
        """ABAC: apakah kunci ini boleh menyentuh mata kuliah tersebut.

        `allowed_courses` kosong berarti seluruh mata kuliah milik tenant.
        """
        if self.allowed_courses is None:
            return True
        if course_id is None:
            # Tanpa mata kuliah tertentu, kueri akan mencakup semua milik tenant —
            # lebih luas dari yang diizinkan kunci ini.
            return False
        return course_id in self.allowed_courses

    def require_course(self, course_id: str | None) -> None:
        if not self.may_access_course(course_id):
            raise TenantPermissionError(
                "Kunci ini dibatasi pada sebagian mata kuliah saja"
            )

    def redacted(self) -> dict[str, str]:
        """Bentuk yang aman untuk log — tanpa kunci, tanpa data pribadi."""
        return {"tenant_id": self.tenant_id, "key_id": self.key_id}


def storage_prefix(tenant_id: str) -> str:
    """Segmen direktori milik satu tenant.

    Divalidasi ulang di sini karena nilainya langsung menjadi bagian dari path
    di disk — pemeriksaan ganda jauh lebih murah daripada satu kebocoran.
    """
    return require_tenant_id(tenant_id, operation="storage_prefix")


def cache_key(tenant_id: str, *parts: str) -> str:
    """Kunci cache yang selalu diawali tenant, agar cache tidak dipakai bersama.

    Tanpa ini, kuis dan pertanyaan pembuka yang dihasilkan dari materi tenant A
    akan tersaji kepada tenant B yang kebetulan memakai `content_id` sama.
    """
    return "\x00".join([require_tenant_id(tenant_id, operation="cache_key"), *parts])
