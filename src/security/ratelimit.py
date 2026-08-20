"""Pembatasan laju berlapis dan pencegahan penebakan kunci.

Tiga lapis, dievaluasi berurutan dari yang paling murah:

1. **Per alamat IP** — menahan penyalahgunaan sebelum kredensial diperiksa,
   termasuk lalu lintas yang tidak membawa kunci sama sekali.
2. **Per kunci API (`key_id`)** — satu integrasi yang salah membuat perulangan
   tidak menghabiskan jatah seluruh tenant.
3. **Per tenant** — batas komersial sesungguhnya; inilah yang menjaga tagihan
   LLM tetap terkendali dan menjadi dasar penjualan berjenjang.

Algoritma: *token bucket*. Dipilih daripada jendela tetap karena membolehkan
lonjakan wajar (mahasiswa membuka riwayat lalu langsung bertanya) tanpa
melonggarkan laju rata-rata, dan tidak punya masalah "reset serentak di detik
ke-0" seperti jendela tetap.

BATASAN PENTING — penyimpanan di memori proses. Dengan lebih dari satu worker
uvicorn atau lebih dari satu replika, tiap proses punya hitungannya sendiri,
sehingga batas efektif menjadi berlipat jumlah proses. Untuk produksi, ganti
`_MemoryBackend` dengan backend Redis (skrip Lua `INCR`+`EXPIRE` atau
`CL.THROTTLE`); antarmuka `_Backend` sengaja dibuat sempit agar penggantinya
tidak menyentuh pemanggil. Lihat docs/SECURITY.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

# Batas jumlah ember yang disimpan. Tanpa ini, penyerang yang mengganti IP atau
# key_id tiap permintaan akan menumbuhkan memori proses tanpa henti — pembatas
# laju itu sendiri berubah menjadi jalur serangan kehabisan memori.
_MAX_BUCKETS = 50_000
_EVICT_TO = 40_000


@dataclass
class _Bucket:
    tokens: float
    updated: float


class _Backend(Protocol):
    def take(self, key: str, capacity: int, refill_per_sec: float, cost: float) -> float:
        """Ambil `cost` token. Mengembalikan 0.0 bila boleh, atau detik tunggu."""
        ...


class _MemoryBackend:
    """Backend token bucket di memori proses (cocok untuk satu proses/dev)."""

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}

    def _evict_if_needed(self, now: float) -> None:
        if len(self._buckets) <= _MAX_BUCKETS:
            return
        # Buang yang paling lama tidak dipakai. Ember yang sudah penuh kembali
        # tidak membawa informasi apa pun, jadi membuangnya tidak melonggarkan
        # batas siapa pun yang sedang aktif.
        urut = sorted(self._buckets.items(), key=lambda kv: kv[1].updated)
        for key, _ in urut[: len(self._buckets) - _EVICT_TO]:
            self._buckets.pop(key, None)

    def take(self, key: str, capacity: int, refill_per_sec: float, cost: float = 1.0) -> float:
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            self._evict_if_needed(now)
            b = _Bucket(tokens=float(capacity), updated=now)
            self._buckets[key] = b
        else:
            b.tokens = min(float(capacity), b.tokens + (now - b.updated) * refill_per_sec)
            b.updated = now

        if b.tokens >= cost:
            b.tokens -= cost
            return 0.0
        kurang = cost - b.tokens
        return kurang / refill_per_sec if refill_per_sec > 0 else 3600.0

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._buckets.clear()
        else:
            self._buckets.pop(key, None)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: int = 0
    scope: str = ""      # lapis mana yang menolak — dipakai log & header

    @property
    def reason(self) -> str:
        return f"batas laju {self.scope}" if self.scope else "batas laju"


@dataclass
class RateLimiter:
    """Pembatas laju berlapis."""
    backend: _MemoryBackend = field(default_factory=_MemoryBackend)

    def check(
        self,
        *,
        ip: str | None,
        key_id: str | None,
        tenant_id: str | None,
        per_minute: int,
        per_day: int,
        ip_per_minute: int,
        cost: float = 1.0,
    ) -> Decision:
        """Periksa ketiga lapis. Lapis pertama yang menolak menghentikan proses."""
        if ip and ip_per_minute > 0:
            tunggu = self.backend.take(
                f"ip:{ip}", ip_per_minute, ip_per_minute / 60.0, cost,
            )
            if tunggu > 0:
                return Decision(False, _ceil(tunggu), "ip")

        if key_id and per_minute > 0:
            tunggu = self.backend.take(
                f"key:{key_id}", per_minute, per_minute / 60.0, cost,
            )
            if tunggu > 0:
                return Decision(False, _ceil(tunggu), "kunci")

        if tenant_id:
            if per_minute > 0:
                tunggu = self.backend.take(
                    f"tenant:m:{tenant_id}", per_minute, per_minute / 60.0, cost,
                )
                if tunggu > 0:
                    return Decision(False, _ceil(tunggu), "tenant/menit")
            if per_day > 0:
                tunggu = self.backend.take(
                    f"tenant:d:{tenant_id}", per_day, per_day / 86_400.0, cost,
                )
                if tunggu > 0:
                    return Decision(False, _ceil(tunggu), "tenant/hari")

        return Decision(True)

    def reset(self, key: str | None = None) -> None:
        self.backend.reset(key)


class AuthThrottle:
    """Perlambatan progresif untuk kunci API yang salah.

    Menahan penebakan kunci secara sistematis. Perlambatan dihitung dari alamat
    IP, bukan dari `key_id`, karena penebak justru mencoba `key_id` yang
    berbeda-beda — membatasi per `key_id` tidak akan pernah menyentuhnya.

    Penundaan naik berlipat (1s, 2s, 4s, …) hingga batas atas, lalu ditolak
    langsung. Kegagalan dilupakan setelah jendela tenang berlalu, sehingga
    kesalahan konfigurasi sesaat tidak mengunci integrasi yang sah selamanya.
    """

    def __init__(
        self,
        threshold: int = 5,
        max_delay: float = 60.0,
        window_seconds: float = 900.0,
        lockout_after: int = 20,
    ) -> None:
        self._threshold = threshold
        self._max_delay = max_delay
        self._window = window_seconds
        self._lockout_after = lockout_after
        self._failures: dict[str, tuple[int, float]] = {}

    def _current(self, ident: str, now: float) -> int:
        jumlah, terakhir = self._failures.get(ident, (0, 0.0))
        if jumlah and (now - terakhir) > self._window:
            self._failures.pop(ident, None)
            return 0
        return jumlah

    def record_failure(self, ident: str) -> None:
        now = time.monotonic()
        jumlah = self._current(ident, now)
        if len(self._failures) > _MAX_BUCKETS:
            self._failures.clear()   # backstop memori; lihat catatan di atas
        self._failures[ident] = (jumlah + 1, now)

    def record_success(self, ident: str) -> None:
        self._failures.pop(ident, None)

    def penalty(self, ident: str) -> tuple[bool, int]:
        """(terkunci, detik_tunggu) untuk sebuah identitas."""
        jumlah = self._current(ident, time.monotonic())
        if jumlah >= self._lockout_after:
            return True, int(self._max_delay)
        if jumlah < self._threshold:
            return False, 0
        tunda = min(self._max_delay, 2.0 ** (jumlah - self._threshold))
        return False, _ceil(tunda)

    def reset(self) -> None:
        self._failures.clear()


def _ceil(x: float) -> int:
    return max(1, int(x) + (1 if x > int(x) else 0))


limiter = RateLimiter()
auth_throttle = AuthThrottle()
