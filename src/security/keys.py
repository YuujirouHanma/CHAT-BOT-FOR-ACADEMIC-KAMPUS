"""Pembuatan dan verifikasi kunci API per tenant.

Bentuk kunci:

    ragk_<key_id>.<secret>
         └─ 16 hex   └─ 43 karakter base64url (256 bit acak)

`key_id` bukan rahasia. Ia dipakai untuk mencari catatan kunci secara langsung
(O(1)) alih-alih mencocokkan ke seluruh tenant satu per satu — pencarian
menyeluruh selain lambat juga membocorkan informasi lewat selisih waktu. `key_id`
juga aman ditulis ke log dan dipakai menelusuri kunci mana yang dipakai atau
harus dicabut, tanpa pernah menuliskan rahasianya.

Sengaja TIDAK memakai nama tenant di dalam kunci: kunci sering tidak sengaja
tersalin ke tiket dukungan, log CI, atau repositori publik, dan nama pelanggan
tidak perlu ikut bocor bersamanya.

---

Kenapa HMAC-SHA256, bukan Argon2id, untuk kunci API?

Argon2id dirancang untuk rahasia beruntropi RENDAH — kata sandi buatan manusia,
yang bisa ditebak dari kamus. Biaya komputasinya yang tinggi itulah pertahanannya.

Kunci di sini 256 bit acak dari `secrets.token_urlsafe`. Menebaknya secara paksa
mustahil terlepas dari fungsi hash yang dipakai, sementara Argon2id akan
menambah ±100 ms pada SETIAP permintaan API — beban yang nyata dan sia-sia.
Karena itu dipakai HMAC-SHA256 dengan *pepper* sisi server, sebagaimana praktik
penyedia API besar. *Pepper* disimpan di luar basis data, jadi bocornya salinan
basis data saja belum cukup untuk memakai kunci-kunci di dalamnya.

Untuk kata sandi manusia — kalau nanti ada login dosen/admin — pakai
`hash_password()` di bawah, yang memang memakai Argon2id.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Final

from src.config import settings

_PREFIX: Final = "ragk"
_KEY_ID_BYTES: Final = 8       # → 16 karakter hex
_SECRET_BYTES: Final = 32      # → 256 bit
_SEPARATOR: Final = "."

# Kata sandi (kalau nanti ada login manusia). OWASP: PBKDF2-HMAC-SHA256 minimal
# 600.000 iterasi bila Argon2id tidak tersedia.
_PBKDF2_ITERATIONS: Final = 600_000
_PBKDF2_SALT_BYTES: Final = 16


@dataclass(frozen=True)
class IssuedKey:
    """Kunci yang baru diterbitkan.

    `raw` hanya ada di memori sekali seumur hidup kunci: ditampilkan kepada
    admin saat pembuatan lalu dibuang. Yang tersimpan hanya `secret_hash`,
    sehingga bocornya berkas tenant tidak memberi penyerang kunci yang bisa
    dipakai.
    """
    key_id: str
    raw: str
    secret_hash: str


def _pepper() -> bytes:
    """Pepper sisi server untuk hashing kunci.

    Kosong hanya diizinkan di luar produksi; `src/api/main.py` menolak start di
    produksi bila belum diisi, agar konfigurasi pengembangan tidak terbawa.
    """
    return settings.tenant_key_pepper.get_secret_value().encode("utf-8")


def hash_secret(secret: str) -> str:
    """HMAC-SHA256 dari bagian rahasia sebuah kunci, dalam hex."""
    return hmac.new(_pepper(), secret.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_key() -> IssuedKey:
    """Terbitkan kunci API baru. Nilai mentahnya tidak pernah disimpan."""
    key_id = secrets.token_hex(_KEY_ID_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    return IssuedKey(
        key_id=key_id,
        raw=f"{_PREFIX}_{key_id}{_SEPARATOR}{secret}",
        secret_hash=hash_secret(secret),
    )


def parse_key(raw: str | None) -> tuple[str, str] | None:
    """Pisahkan kunci mentah menjadi (key_id, secret); None bila bentuknya salah.

    Bentuk yang salah ditolak di sini, sebelum menyentuh penyimpanan — permintaan
    dengan header sampah tidak perlu membebani pencarian tenant.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    awalan = f"{_PREFIX}_"
    if not raw.startswith(awalan):
        return None
    sisa = raw[len(awalan):]
    key_id, _, secret = sisa.partition(_SEPARATOR)
    if not secret or len(key_id) != _KEY_ID_BYTES * 2:
        return None
    if not all(c in "0123456789abcdef" for c in key_id):
        return None
    return key_id, secret


def verify_secret(secret: str, expected_hash: str) -> bool:
    """Cocokkan rahasia dengan hash tersimpan, dalam waktu tetap.

    `compare_digest` dipakai agar lama pembandingan tidak bergantung pada berapa
    karakter awal yang sudah benar — pembandingan biasa membocorkan itu lewat
    selisih waktu dan bisa dipakai menebak hash karakter demi karakter.
    """
    if not secret or not expected_hash:
        return False
    return hmac.compare_digest(hash_secret(secret), expected_hash)


# --- Kata sandi manusia (belum dipakai; disiapkan bila nanti ada login) ---

def hash_password(password: str) -> str:
    """Hash kata sandi. Argon2id bila tersedia, selain itu PBKDF2-HMAC-SHA256.

    Argon2id lebih disukai karena tahan terhadap serangan berbasis GPU/ASIC —
    ia menuntut memori, bukan hanya waktu. PBKDF2 dipakai sebagai cadangan agar
    modul ini tidak memaksa dependensi baru pada pemasangan yang belum
    membutuhkan login manusia.
    """
    try:
        from argon2 import PasswordHasher  # type: ignore[import-not-found]
    except ImportError:
        garam = secrets.token_bytes(_PBKDF2_SALT_BYTES)
        turunan = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), garam, _PBKDF2_ITERATIONS,
        )
        return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${garam.hex()}${turunan.hex()}"

    # Parameter bawaan argon2-cffi mengikuti anjuran RFC 9106.
    return PasswordHasher().hash(password)


def verify_password(password: str, stored: str) -> bool:
    """Verifikasi kata sandi terhadap hash tersimpan, apa pun algoritmanya."""
    if not stored:
        return False

    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterasi_s, garam_hex, harapan_hex = stored.split("$", 3)
            turunan = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"),
                bytes.fromhex(garam_hex), int(iterasi_s),
            )
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(turunan.hex(), harapan_hex)

    try:
        from argon2 import PasswordHasher  # type: ignore[import-not-found]
        from argon2.exceptions import VerifyMismatchError  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return PasswordHasher().verify(stored, password)
    except (VerifyMismatchError, Exception):
        return False
