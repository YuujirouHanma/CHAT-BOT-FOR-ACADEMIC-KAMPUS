"""Penyuntingan data sensitif sebelum ditulis ke log.

Log adalah tempat rahasia paling sering bocor: ia disalin ke tiket dukungan,
dikirim ke layanan agregasi pihak ketiga, dan disimpan jauh lebih lama daripada
data aslinya. Karena itu penyuntingan dilakukan di sisi PENULIS — bukan
diserahkan kepada disiplin tiap pemanggil untuk "jangan mencatat yang sensitif".

Yang disunting:
- kunci API dan token (pola `ragk_…`, `Bearer …`, `sk-…`)
- nilai pada field bernama sensitif (password, secret, token, api_key, …)
- identitas mahasiswa (`student_id`) dan alamat surel — data pribadi
"""
from __future__ import annotations

import re
from typing import Any

MASK = "[disunting]"

# Nama field yang isinya tidak pernah boleh muncul utuh di log.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey", "x-api-key",
    "authorization", "auth", "cookie", "set-cookie", "session_token",
    "secret_hash", "private_key", "refresh_token", "access_token",
    "tenant_key_pepper", "pepper", "raw", "credential", "credentials",
})

# Field yang berisi data pribadi: disamarkan sebagian, tidak dihapus, supaya
# masih bisa dipakai menelusuri satu kasus tanpa mengungkap identitasnya.
_PII_KEYS: frozenset[str] = frozenset({
    "student_id", "email", "nim", "phone", "nama", "full_name", "name_student",
})

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ragk_[0-9a-f]{16}\.[A-Za-z0-9_-]+"), f"ragk_{MASK}"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE), f"Bearer {MASK}"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}"), f"sk-{MASK}"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{16,}"), f"gsk_{MASK}"),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}"), f"hf_{MASK}"),
    (
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        lambda m: pseudonymize(m.group(0)),  # type: ignore[arg-type]
    ),
)

_MAX_DEPTH = 12


def pseudonymize(value: str) -> str:
    """Samarkan nilai pribadi tetapi tetap bisa dibedakan satu sama lain.

    Contoh: `2021001234` → `20…34`. Cukup untuk mencocokkan dua baris log yang
    membicarakan mahasiswa yang sama saat menelusuri masalah, tetapi tidak cukup
    untuk mengetahui siapa orangnya dari log saja.
    """
    if not value:
        return ""
    s = str(value)
    if len(s) <= 4:
        return "…"
    return f"{s[:2]}…{s[-2:]}"


def scrub_text(text: str) -> str:
    """Buang rahasia yang terlanjur menempel di dalam sebuah string."""
    if not text:
        return text
    hasil = text
    for pola, pengganti in _PATTERNS:
        hasil = pola.sub(pengganti, hasil)  # type: ignore[arg-type]
    return hasil


def scrub(obj: Any, _depth: int = 0) -> Any:
    """Salin struktur data dengan bagian sensitifnya sudah disunting.

    Bekerja rekursif pada dict/list. Kedalaman dibatasi supaya struktur bersiklus
    atau sangat dalam tidak membuat proses pencatatan log macet.
    """
    if _depth > _MAX_DEPTH:
        return "[terlalu dalam]"

    if obj is None or isinstance(obj, int | float | bool):
        return obj
    if isinstance(obj, str):
        return scrub_text(obj)
    if isinstance(obj, dict):
        hasil: dict[Any, Any] = {}
        for k, v in obj.items():
            kunci = str(k).lower()
            if kunci in _SENSITIVE_KEYS:
                hasil[k] = MASK
            elif kunci in _PII_KEYS:
                hasil[k] = pseudonymize(str(v)) if v is not None else None
            else:
                hasil[k] = scrub(v, _depth + 1)
        return hasil
    if isinstance(obj, list | tuple | set):
        return [scrub(x, _depth + 1) for x in obj]
    return scrub_text(str(obj))
