"""Penyimpan status job pengindeksan berkas di latar belakang.

Sama persis polanya dengan ``session.py`` — dua backend, dipilih lewat
``settings.session_backend`` (dipakai ulang, bukan pengaturan baru): job yang
dibuat oleh satu pekerja uvicorn harus terbaca oleh pekerja LAIN yang kebetulan
menerima permintaan cek status, dan dict dalam proses tidak terlihat lintas
proses. Lihat komentar di puncak ``session.py`` untuk alasan lengkapnya —
kelasnya identik, hanya isinya beda (status upload, bukan riwayat obrolan).

Job terikat pada kampus pengunggahnya: `job_id` milik kampus lain diperlakukan
sama dengan `job_id` yang tidak pernah ada (404), bukan dibacakan statusnya.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import settings
from src.tenancy import require_tenant_id
from src.utils.logger import logger

TTL_SECONDS = 3600

_REDIS_PREFIX = "ragacademic:upload_job:"


@dataclass
class UploadJob:
    job_id: str
    tenant_id: str
    status: str = "pending"  # pending | running | done | error
    result: dict[str, Any] | None = None
    error: str | None = None
    updated_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.updated_at) > TTL_SECONDS

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str | bytes) -> UploadJob | None:
        try:
            data: dict[str, Any] = json.loads(raw)
            return UploadJob(**data)
        except (ValueError, TypeError) as exc:
            logger.warning("Job upload tidak bisa dibaca: {}", exc)
            return None


class MemoryJobStore:
    """Dict dalam proses. Hanya benar bila proses backend cuma satu."""

    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], UploadJob] = {}

    async def create(self, *, tenant_id: str) -> UploadJob:
        tenant_id = require_tenant_id(tenant_id, operation="job.create")
        job = UploadJob(job_id=str(uuid.uuid4()), tenant_id=tenant_id)
        self._jobs[(tenant_id, job.job_id)] = job
        return job

    async def get(self, job_id: str, *, tenant_id: str) -> UploadJob | None:
        kunci = (require_tenant_id(tenant_id, operation="job.get"), job_id)
        job = self._jobs.get(kunci)
        if job is None:
            return None
        if job.is_expired():
            del self._jobs[kunci]
            return None
        return job

    async def save(self, job: UploadJob) -> None:
        job.updated_at = time.time()
        self._jobs[(job.tenant_id, job.job_id)] = job


class RedisJobStore:
    """Job bersama antar pekerja, dengan kedaluwarsa ditangani Redis."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(tenant_id: str, job_id: str) -> str:
        return f"{_REDIS_PREFIX}{require_tenant_id(tenant_id, operation='job')}:{job_id}"

    async def create(self, *, tenant_id: str) -> UploadJob:
        job = UploadJob(job_id=str(uuid.uuid4()), tenant_id=tenant_id)
        await self.save(job)
        return job

    async def get(self, job_id: str, *, tenant_id: str) -> UploadJob | None:
        raw = await self._redis.get(self._key(tenant_id, job_id))
        if raw is None:
            return None
        job = UploadJob.from_json(raw)
        return job if job is not None and job.tenant_id == tenant_id else None

    async def save(self, job: UploadJob) -> None:
        job.updated_at = time.time()
        await self._redis.set(
            self._key(job.tenant_id, job.job_id), job.to_json(), ex=TTL_SECONDS,
        )


def _build_store() -> MemoryJobStore | RedisJobStore:
    if settings.session_backend == "redis":
        logger.info("Job store (upload): REDIS di {}", settings.redis_url)
        return RedisJobStore(settings.redis_url)

    logger.info("Job store (upload): MEMORI (hanya benar untuk satu proses backend)")
    return MemoryJobStore()


job_store = _build_store()
