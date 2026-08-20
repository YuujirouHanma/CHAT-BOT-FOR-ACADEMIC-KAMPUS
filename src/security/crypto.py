"""Enkripsi tingkat field untuk data pribadi, dengan pola *envelope encryption*.

Data pribadi di sistem ini pada dasarnya satu: `student_id` — identitas mahasiswa
yang dikirim aplikasi kampus. Ia melekat pada seluruh riwayat belajar seseorang,
jadi bocornya berkas percakapan berarti bocornya "siapa mempelajari apa".

Dua kebutuhan yang saling bertentangan diselesaikan dengan dua turunan berbeda
dari nilai yang sama:

- **Kerahasiaan** → `encrypt_field()`: AES-256-GCM, nonce acak setiap kali.
  Dua penyimpanan nilai yang sama menghasilkan sandi yang berbeda, sehingga
  penyerang tidak bisa menyimpulkan apa pun dari pola pengulangan.
- **Pencarian** → `blind_index()`: HMAC-SHA256, deterministik. Dipakai untuk
  menyaring "percakapan milik mahasiswa ini" tanpa pernah mendekripsi apa pun.

Enkripsi deterministik untuk keduanya sekaligus akan tampak lebih sederhana,
tetapi membocorkan kesamaan nilai (dua baris dengan sandi sama = mahasiswa sama)
dan membuka analisis frekuensi. Karena itu keduanya sengaja dipisah.

*Envelope encryption*: setiap nilai disandikan dengan kunci data (DEK) acak; DEK
itu sendiri disandikan dengan kunci induk (KEK) milik KMS. Dampaknya, memutar
(rotate) kunci induk tidak menuntut penyandian ulang seluruh basis data, dan
kunci induk tidak pernah hadir di dalam kode maupun berkas konfigurasi.

INFRASTRUKTUR YANG DIBUTUHKAN DI PRODUKSI:
- KEK di AWS KMS / Google Cloud KMS / HashiCorp Vault (bukan variabel lingkungan).
- Paket `cryptography` terpasang. Tanpa itu penyandian tidak tersedia, dan
  `src/api/main.py` menolak start di produksi bila `encrypt_pii=true`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Final, Protocol

from src.config import settings
from src.utils.logger import logger

_VERSION: Final = "v1"           # awalan sandi; memungkinkan ganti algoritma nanti
_NONCE_BYTES: Final = 12         # 96 bit — ukuran yang dianjurkan untuk GCM
_DEK_BYTES: Final = 32           # AES-256

try:  # pragma: no cover - bergantung lingkungan pemasangan
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _AESGCM_AVAILABLE = True
except ImportError:  # pragma: no cover
    AESGCM = None  # type: ignore[assignment,misc]
    _AESGCM_AVAILABLE = False


class CryptoUnavailableError(RuntimeError):
    """Penyandian diminta tetapi tidak dapat dijalankan pada pemasangan ini."""


class KeyProvider(Protocol):
    """Sumber kunci induk (KEK).

    Antarmuka sengaja hanya dua metode agar penggantinya — AWS KMS
    (`Encrypt`/`Decrypt`), Google Cloud KMS, atau Vault Transit — cukup
    mengimplementasikan keduanya tanpa menyentuh pemanggil mana pun.
    """

    def wrap(self, dek: bytes) -> bytes: ...
    def unwrap(self, wrapped: bytes) -> bytes: ...


class EnvKeyProvider:
    """KEK dari variabel lingkungan — HANYA untuk pengembangan dan pengujian.

    Tidak layak produksi: kunci induk berakhir di berkas `.env`, log proses, dan
    citra kontainer. Di produksi pakai KMS sungguhan; `main.py` menolak start
    bila penyandian aktif sementara penyedianya masih ini.
    """

    production_safe = False

    def __init__(self, key: bytes | None = None) -> None:
        if key is None:
            mentah = settings.pii_kek.get_secret_value()
            key = base64.urlsafe_b64decode(mentah) if mentah else b""
        if key and len(key) != 32:
            raise CryptoUnavailableError(
                "PII_KEK harus 32 byte (base64url) untuk AES-256"
            )
        self._key = key

    def _cipher(self):
        if not _AESGCM_AVAILABLE:
            raise CryptoUnavailableError(
                "Paket 'cryptography' belum terpasang; penyandian PII tidak tersedia"
            )
        if not self._key:
            raise CryptoUnavailableError("PII_KEK belum diisi")
        return AESGCM(self._key)

    def wrap(self, dek: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        return nonce + self._cipher().encrypt(nonce, dek, b"dek")

    def unwrap(self, wrapped: bytes) -> bytes:
        nonce, sandi = wrapped[:_NONCE_BYTES], wrapped[_NONCE_BYTES:]
        return self._cipher().decrypt(nonce, sandi, b"dek")


_provider: KeyProvider | None = None


def set_key_provider(provider: KeyProvider | None) -> None:
    """Pasang penyedia kunci (dipakai saat startup dan oleh pengujian)."""
    global _provider
    _provider = provider


def key_provider() -> KeyProvider:
    global _provider
    if _provider is None:
        _provider = EnvKeyProvider()
    return _provider


def is_available() -> bool:
    """Apakah penyandian benar-benar dapat dijalankan sekarang."""
    if not _AESGCM_AVAILABLE:
        return False
    try:
        key_provider().wrap(os.urandom(_DEK_BYTES))
        return True
    except Exception:
        return False


def encrypt_field(plaintext: str | None, *, aad: str = "") -> str | None:
    """Sandikan satu nilai. None dan string kosong dibiarkan apa adanya.

    `aad` (*additional authenticated data*) mengikat sandi pada konteksnya —
    diisi `tenant_id`. Sandi milik tenant A yang dipindahkan ke berkas tenant B
    akan gagal didekripsi, bukan diam-diam terbaca.
    """
    if not plaintext:
        return plaintext
    if not settings.encrypt_pii:
        return plaintext
    if not _AESGCM_AVAILABLE:
        raise CryptoUnavailableError(
            "ENCRYPT_PII=true tetapi paket 'cryptography' belum terpasang"
        )

    dek = os.urandom(_DEK_BYTES)
    nonce = os.urandom(_NONCE_BYTES)
    sandi = AESGCM(dek).encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
    terbungkus = key_provider().wrap(dek)

    def b64(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")

    return f"{_VERSION}:{b64(terbungkus)}:{b64(nonce)}:{b64(sandi)}"


def decrypt_field(stored: str | None, *, aad: str = "") -> str | None:
    """Kembalikan nilai asli. Nilai yang belum tersandi dikembalikan apa adanya.

    Toleransi terhadap nilai lama yang belum tersandi disengaja: penyandian bisa
    dinyalakan pada data yang sudah ada tanpa migrasi serentak, dan baris lama
    tetap terbaca sampai tersentuh penulisan berikutnya.
    """
    if not stored or not stored.startswith(f"{_VERSION}:"):
        return stored

    try:
        _, terbungkus_b64, nonce_b64, sandi_b64 = stored.split(":", 3)

        def unb64(s: str) -> bytes:
            return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

        dek = key_provider().unwrap(unb64(terbungkus_b64))
        return AESGCM(dek).decrypt(
            unb64(nonce_b64), unb64(sandi_b64), aad.encode("utf-8"),
        ).decode("utf-8")
    except Exception as exc:
        # Nilai yang tidak terbaca tidak boleh menggagalkan seluruh daftar
        # riwayat; dicatat lalu diperlakukan sebagai tidak diketahui.
        logger.error("Gagal mendekripsi field PII: {}", type(exc).__name__)
        return None


def blind_index(value: str | None, *, tenant_id: str = "") -> str | None:
    """Indeks buta: penyaringan berdasarkan nilai tanpa menyimpan nilainya.

    Deterministik, sehingga bisa dicocokkan; satu arah, sehingga tidak bisa
    dikembalikan menjadi identitas aslinya. `tenant_id` ikut masuk ke masukan
    HMAC agar `student_id` yang sama di dua kampus menghasilkan indeks berbeda —
    tanpa itu, dua tenant dapat saling menyimpulkan bahwa seseorang terdaftar di
    keduanya hanya dengan membandingkan indeks.
    """
    if not value:
        return None
    kunci = settings.tenant_key_pepper.get_secret_value().encode("utf-8") or b"dev-index"
    pesan = f"{tenant_id}\x00{value}".encode()
    return hmac.new(kunci, pesan, hashlib.sha256).hexdigest()[:32]
