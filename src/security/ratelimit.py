"""Kuota permintaan berlapis dan pencegahan penebakan kunci.

Tiga lapis, dievaluasi berurutan dari yang paling murah:

1. **Per alamat IP** (`check_ip`) — menahan penyalahgunaan SEBELUM kredensial
   diperiksa, termasuk lalu lintas yang tidak membawa kunci sama sekali.
2. **Perlambatan auth** (`AuthThrottle`) — menahan penebakan kunci sistematis.
3. **Per kampus** (`check`) — kuota komersial: setiap pertanyaan memanggil LLM
   berbayar, jadi tanpa batas, satu kampus yang ramai — atau satu skrip yang
   salah membuat perulangan — menghabiskan kapasitas dan tagihan milik semua.

Algoritma jendela tetap per menit dan per hari. Dipilih karena di Redis cukup
INCR + EXPIRE yang atomik, tanpa skrip Lua. Lonjakan di batas dua jendela
(maksimal dua kali lipat sesaat) dapat diterima untuk tujuan ini.

Backend penyimpan mengikuti `SESSION_BACKEND`, sama seperti sesi dan job unggah:
backend produksi berjalan dengan beberapa pekerja uvicorn, dan hitungan di memori
tiap pekerja membuat batas efektif berlipat jumlah pekerja. Memori hanya benar
untuk satu proses (pengembangan & uji).

Bila Redis gagal, permintaan DILOLOSKAN (fail open) dengan peringatan di log:
kuota adalah kendali kewajaran pemakaian, bukan pengaman akses — gangguan Redis
sesaat tidak boleh menutup layanan untuk semua kampus.

`AuthThrottle` sengaja TIDAK ikut ke Redis. Ia hanya perlu memperlambat, dan
perlambatan per proses sudah cukup menghabisi laju penebakan; menaruhnya di Redis
berarti satu permintaan tambahan ke jaringan pada jalur autentikasi setiap
permintaan, termasuk yang sah.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from src.config import settings
from src.utils.logger import logger

_PREFIX = "ragacademic:rl:"
_MINUTE_TTL = 70          # sedikit lebih lama dari jendelanya, lalu hilang sendiri
_DAY_TTL = 90_000         # 25 jam
_MAX_KEYS = 50_000        # batas cache memori agar tidak tumbuh tanpa henti


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: int = 0
    scope: str = ""


class _Backend(Protocol):
    async def hit(self, key: str, ttl: int) -> int: ...


class MemoryBackend:
    """Hitungan di memori proses. Hanya benar bila backend berjalan satu proses."""

    def __init__(self) -> None:
        self._counts: dict[str, tuple[int, float]] = {}

    def reset(self) -> None:
        """Kosongkan seluruh hitungan.

        Dipakai di antara uji: tanpa ini satu uji yang menghabiskan jatah akan
        menjatuhkan uji berikutnya, dan kegagalannya tampak acak karena
        bergantung pada urutan jalannya.
        """
        self._counts.clear()

    async def hit(self, key: str, ttl: int) -> int:
        now = time.monotonic()
        count, expires = self._counts.get(key, (0, now + ttl))
        if expires <= now:
            count, expires = 0, now + ttl
        count += 1
        self._counts[key] = (count, expires)
        if len(self._counts) > _MAX_KEYS:
            self._counts = {k: v for k, v in self._counts.items() if v[1] > now}
        return count


class RedisBackend:
    """Hitungan bersama antar pekerja, kedaluwarsa ditangani Redis."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, decode_responses=True)

    async def hit(self, key: str, ttl: int) -> int:
        pipe = self._redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, ttl)
        count, _ = await pipe.execute()
        return int(count)


class TenantRateLimiter:
    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    def reset(self) -> None:
        """Kosongkan hitungan backend bila ia mendukungnya.

        Backend Redis sengaja tidak menyediakannya: mengosongkan hitungan di
        Redis berarti menghapus jatah SELURUH kampus yang sedang berjalan, dan
        itu bukan sesuatu yang boleh dilakukan dari dalam proses layanan.
        """
        bersihkan = getattr(self._backend, "reset", None)
        if callable(bersihkan):
            bersihkan()

    async def check_ip(
        self, ip: str | None, *, per_minute: int, now: float | None = None,
    ) -> Decision:
        """Lapis termurah: batasi per alamat asal, sebelum kunci diperiksa.

        Ada karena lapis per kampus baru dapat dihitung SETELAH kunci diverifikasi
        — jadi ia tidak menahan apa pun dari pemanggil yang tidak membawa kunci
        sama sekali. Alamat asal dibaca lewat `src.api.auth.client_ip`, yang hanya
        mempercayai `X-Forwarded-For` bila memang ada proxy tepercaya di depan.
        """
        if not ip or per_minute <= 0:
            return Decision(True)
        now = time.time() if now is None else now
        try:
            jendela = int(now // 60)
            n = await self._backend.hit(f"{_PREFIX}ip:{ip}:{jendela}", _MINUTE_TTL)
            if n > per_minute:
                return Decision(False, max(1, 60 - int(now % 60)), "ip")
        except Exception as exc:  # noqa: BLE001 - lihat catatan fail open di atas
            logger.warning("Kuota IP tidak dapat diperiksa: {}", exc)
        return Decision(True)

    async def check(
        self,
        tenant_id: str,
        *,
        per_minute: int,
        per_day: int,
        now: float | None = None,
    ) -> Decision:
        """Hitung satu permintaan kampus ini; tolak bila melewati kuotanya.

        Kuota 0 berarti tanpa batas untuk jendela itu.
        """
        now = time.time() if now is None else now
        try:
            if per_minute > 0:
                jendela = int(now // 60)
                n = await self._backend.hit(f"{_PREFIX}{tenant_id}:m:{jendela}", _MINUTE_TTL)
                if n > per_minute:
                    return Decision(False, max(1, 60 - int(now % 60)), "minute")
            if per_day > 0:
                hari = int(now // 86_400)
                n = await self._backend.hit(f"{_PREFIX}{tenant_id}:d:{hari}", _DAY_TTL)
                if n > per_day:
                    return Decision(False, max(1, 86_400 - int(now % 86_400)), "day")
        except Exception as exc:  # noqa: BLE001 - lihat catatan fail open di atas
            logger.warning("Kuota kampus {} tidak dapat diperiksa: {}", tenant_id, exc)
        return Decision(True)


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
        if len(self._failures) > _MAX_KEYS:
            # Backstop memori: penyerang yang mengganti alamat tiap percobaan
            # tidak boleh menumbuhkan proses ini tanpa henti.
            self._failures.clear()
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


def _build_limiter() -> TenantRateLimiter:
    if settings.session_backend == "redis":
        return TenantRateLimiter(RedisBackend(settings.redis_url))
    return TenantRateLimiter(MemoryBackend())


limiter = _build_limiter()
auth_throttle = AuthThrottle()
