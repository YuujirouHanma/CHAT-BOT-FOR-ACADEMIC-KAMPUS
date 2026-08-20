"""Qdrant vector store wrapper — multi-tenant.

Collection layout — designed for bge-m3 hybrid search:
- One named DENSE vector ("dense") with cosine distance and dim from settings.
- One named SPARSE vector ("sparse"), enabled when settings.enable_hybrid_search.
- Payload mirrors the Chunk schema except the embeddings themselves.

ISOLASI ANTAR TENANT — bagian terpenting berkas ini
---------------------------------------------------
Satu koleksi dipakai bersama seluruh pelanggan, dibedakan oleh field payload
`tenant_id`. Agar pola itu aman, dua hal ditegakkan secara struktural, bukan
lewat kedisiplinan pemanggil:

1. `tenant_id` adalah argumen WAJIB berjenis keyword-only tanpa nilai bawaan
   pada setiap metode publik. Lupa mengirimnya menjadi TypeError saat pemanggilan
   — bukan pencarian diam-diam ke seluruh korpus lintas pelanggan.
2. `_build_filter()` SELALU mengembalikan filter berisi kondisi `tenant_id`, dan
   tidak pernah `None`. Sebelumnya fungsi ini mengembalikan `None` ketika tak ada
   penyaring, yang berarti "cari di semua" — persis jalur yang membocorkan data
   begitu ada pelanggan kedua. Jalur itu kini tidak ada lagi.

Mengapa field payload, bukan satu koleksi per tenant: koleksi Qdrant membawa
biaya tetap (segmen, indeks HNSW, memori) yang tidak masuk akal dikalikan jumlah
pelanggan, dan menambah pelanggan baru jadi menuntut operasi administratif.
Qdrant sendiri menganjurkan pola payload + indeks bertanda tenant, yang juga
mengelompokkan data satu tenant berdekatan di disk sehingga kuerinya lebih cepat.

Operations are async. Real Qdrant calls use `asyncio.to_thread` because the
official client is sync (AsyncQdrantClient exists but has fewer features
and a less stable API).

Hybrid search uses Qdrant's built-in `Prefetch` + `FusionQuery` (RRF) as of
qdrant-client >= 1.10, which fuses dense and sparse results server-side.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import settings
from src.schemas import Chunk
from src.tenancy import TenantScopeError, require_tenant_id
from src.utils.logger import logger

_DENSE_VEC = "dense"
_SPARSE_VEC = "sparse"
_UPSERT_BATCH = 64

# Field payload yang membawa pemilik data. Namanya dipakai juga sebagai kunci
# indeks bertanda tenant di Qdrant.
TENANT_FIELD = "tenant_id"


class QdrantStore:
    """Thin async wrapper around qdrant-client for our chunk schema."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        enable_sparse: bool | None = None,
    ) -> None:
        self._client = client or self._build_client()
        self._collection = collection or settings.qdrant_collection
        self._enable_sparse = (
            enable_sparse if enable_sparse is not None else settings.enable_hybrid_search
        )

    @staticmethod
    def _build_client() -> QdrantClient:
        """Build the Qdrant client from settings.

        local  → embedded on-disk client, no server required (dev/tests).
        server → connect to the Qdrant server (Docker/production).
        """
        if settings.qdrant_mode == "server":
            logger.info(
                "Qdrant client: SERVER mode at {}:{}",
                settings.qdrant_host,
                settings.qdrant_port,
            )
            return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

        logger.info("Qdrant client: LOCAL mode (path={})", settings.qdrant_path)
        return QdrantClient(path=settings.qdrant_path)

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, then ensure indexes. Idempotent."""
        exists = await asyncio.to_thread(
            self._client.collection_exists, self._collection
        )
        if exists:
            logger.info("Qdrant collection '{}' already exists", self._collection)
            await self._ensure_tenant_index()
            return

        vectors_config = {
            _DENSE_VEC: qm.VectorParams(
                size=settings.embed_dim,
                distance=qm.Distance.COSINE,
            )
        }
        sparse_vectors_config = (
            {_SPARSE_VEC: qm.SparseVectorParams(modifier=qm.Modifier.IDF)}
            if self._enable_sparse
            else None
        )

        await asyncio.to_thread(
            self._client.create_collection,
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        logger.info(
            "Created Qdrant collection '{}' (dim={}, sparse={})",
            self._collection,
            settings.embed_dim,
            self._enable_sparse,
        )
        await self._ensure_tenant_index()

    async def _ensure_tenant_index(self) -> None:
        """Indeks payload untuk `tenant_id`, ditandai sebagai kunci tenant.

        Tanpa indeks ini setiap kueri memindai seluruh koleksi lalu membuang
        milik tenant lain — benar tetapi semakin lambat seiring bertambahnya
        pelanggan. Dengan `is_tenant=True`, Qdrant menata data satu tenant
        berdekatan di disk sehingga kuerinya menyentuh jauh lebih sedikit segmen.

        Aman diulang. Kegagalan tidak menghentikan startup: indeks ini soal
        kinerja, sedangkan kebenaran isolasi dijamin oleh filternya.
        """
        try:
            try:
                skema: Any = qm.KeywordIndexParams(
                    type=qm.KeywordIndexType.KEYWORD, is_tenant=True,
                )
            except (AttributeError, TypeError):
                # Qdrant/klien lama belum mengenal `is_tenant`; indeks biasa
                # tetap memberi manfaat penyaringan, hanya tanpa penataan disk.
                skema = qm.PayloadSchemaType.KEYWORD

            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=self._collection,
                field_name=TENANT_FIELD,
                field_schema=skema,
            )
            logger.info("Indeks payload '{}' siap", TENANT_FIELD)
        except Exception as exc:
            logger.warning(
                "Indeks tenant belum terpasang ({}). Isolasi tetap aman, "
                "tetapi kueri akan melambat seiring bertambahnya tenant.", exc,
            )

    async def upsert_chunks(
        self, chunks: Sequence[Chunk], *, tenant_id: str,
    ) -> int:
        """Upsert chunks in batches. Returns number of points written.

        Setiap potongan dicap `tenant_id` di sini. Potongan yang sudah membawa
        `tenant_id` BERBEDA ditolak — kalau sampai terjadi, ada percampuran data
        di lapisan atas, dan menulisnya berarti menanam kebocoran permanen di
        dalam index yang baru ketahuan jauh belakangan.
        """
        tenant_id = require_tenant_id(tenant_id, operation="upsert_chunks")
        if not chunks:
            return 0

        asing = [
            c.chunk_id for c in chunks
            if c.tenant_id is not None and c.tenant_id != tenant_id
        ]
        if asing:
            raise TenantScopeError(
                f"{len(asing)} potongan membawa tenant_id berbeda dari '{tenant_id}' "
                f"(mis. {asing[0]}); penulisan dibatalkan."
            )

        points = [self._chunk_to_point(c, tenant_id) for c in chunks]
        total = 0
        for batch_start in range(0, len(points), _UPSERT_BATCH):
            batch = points[batch_start : batch_start + _UPSERT_BATCH]
            await asyncio.to_thread(
                self._client.upsert,
                collection_name=self._collection,
                points=batch,
                wait=True,
            )
            total += len(batch)
        logger.info(
            "Upserted {} points to '{}' (tenant={})", total, self._collection, tenant_id,
        )
        return total

    async def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        top_k: int | None = None,
        source_filter: str | None = None,
        content_id: str | None = None,
        course_id: str | None = None,
        weeks: list[int] | None = None,
        *,
        tenant_id: str,
    ) -> list[dict]:
        """Hybrid search if sparse vector is given and enabled, else dense-only.

        Args:
            dense_vector: 1024-dim dense embedding of the query.
            sparse_vector: token_id → weight map; required for hybrid.
            top_k: Number of results; defaults to settings.retrieval_top_k.
            source_filter: Optional file name to restrict results.
            content_id: Optional content identifier to restrict results.
            course_id: Optional course to restrict results.
            weeks: Optional list of weeks; hasil dibatasi ke minggu-minggu itu.
            tenant_id: WAJIB. Pemilik data yang boleh dicari.

        Returns:
            List of dicts: {"chunk_id", "score", "payload"}.
        """
        tenant_id = require_tenant_id(tenant_id, operation="search")
        top_k = top_k or settings.retrieval_top_k
        qfilter = self._build_filter(
            tenant_id=tenant_id, source_filter=source_filter, content_id=content_id,
            course_id=course_id, weeks=weeks,
        )

        use_hybrid = self._enable_sparse and sparse_vector is not None
        result = None
        if use_hybrid:
            assert sparse_vector is not None
            try:
                result = await asyncio.to_thread(
                    self._client.query_points,
                    collection_name=self._collection,
                    prefetch=[
                        qm.Prefetch(
                            query=dense_vector,
                            using=_DENSE_VEC,
                            limit=top_k * 2,
                            filter=qfilter,
                        ),
                        qm.Prefetch(
                            query=qm.SparseVector(
                                indices=list(sparse_vector.keys()),
                                values=list(sparse_vector.values()),
                            ),
                            using=_SPARSE_VEC,
                            limit=top_k * 2,
                            filter=qfilter,
                        ),
                    ],
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
            except KeyError as exc:
                # Qdrant local mode raises KeyError('sparse') from _rescore_idf
                # when the IDF-modified sparse index has no entries yet — e.g. a
                # collection with zero sparse-indexed points. Degrade gracefully
                # to dense-only instead of surfacing a 500 to the caller.
                logger.warning(
                    "Hybrid search unavailable ({}); falling back to dense-only", exc
                )

        if result is None:
            result = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self._collection,
                query=dense_vector,
                using=_DENSE_VEC,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )

        return [
            {
                "chunk_id": str(p.id),
                "score": float(p.score),
                "payload": dict(p.payload or {}),
            }
            for p in result.points
        ]

    async def list_indexed_files(
        self,
        content_id: str | None = None,
        *,
        tenant_id: str,
    ) -> list[dict]:
        """Return unique source files indexed for the given content_id.

        Scrolls Qdrant with an optional content_id filter and deduplicates
        by source_file. Returns list of {"source_file", "content_id"}.
        """
        tenant_id = require_tenant_id(tenant_id, operation="list_indexed_files")
        qfilter = self._build_filter(tenant_id=tenant_id, content_id=content_id)

        seen: set[str] = set()
        results: list[dict] = []
        offset: Any = None

        while True:
            batch, next_offset = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=100,
                offset=offset,
                with_payload=["source_file", "content_id"],
                with_vectors=False,
            )
            for point in batch:
                sf = (point.payload or {}).get("source_file")
                if sf and sf not in seen:
                    seen.add(sf)
                    results.append({
                        "source_file": sf,
                        "content_id": (point.payload or {}).get("content_id"),
                    })
            if next_offset is None:
                break
            offset = next_offset

        return results

    async def _scroll_payloads(
        self, fields: list[str], qfilter: qm.Filter,
    ) -> list[dict]:
        """Scroll the collection, returning payloads projected to `fields`.

        `qfilter` bukan opsional: setiap pemanggil membangunnya lewat
        `_build_filter`, yang selalu menyertakan `tenant_id`. Menjadikannya wajib
        di sini menutup kemungkinan seseorang menambah pemanggil baru yang lupa
        menyaring, lalu memindai data seluruh pelanggan.
        """
        payloads: list[dict] = []
        offset: Any = None
        while True:
            batch, next_offset = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=256,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            payloads.extend(dict(p.payload or {}) for p in batch)
            if next_offset is None:
                break
            offset = next_offset
        return payloads

    async def list_courses(self, *, tenant_id: str) -> list[dict]:
        """Distinct courses that have indexed content. [{course_id, course_name}].

        When a course has both a custom display name (sent explicitly at upload)
        and an auto-derived one, the custom name wins so the nice name isn't lost
        just because one week was uploaded without course_name.
        """
        from src.catalog import humanize_course

        tenant_id = require_tenant_id(tenant_id, operation="list_courses")
        payloads = await self._scroll_payloads(
            ["course_id", "course_name"], self._build_filter(tenant_id=tenant_id),
        )
        courses: dict[str, str] = {}
        for p in payloads:
            cid = p.get("course_id")
            if not cid:
                continue
            name = p.get("course_name") or cid
            existing = courses.get(cid)
            auto = humanize_course(cid)
            if existing is None or (existing == auto and name != auto):
                courses[cid] = name
        return [
            {"course_id": cid, "course_name": name}
            for cid, name in sorted(courses.items())
        ]

    async def list_weeks(self, course_id: str, *, tenant_id: str) -> list[int]:
        """Distinct weeks with indexed content for a course, ascending."""
        tenant_id = require_tenant_id(tenant_id, operation="list_weeks")
        qfilter = self._build_filter(tenant_id=tenant_id, course_id=course_id)
        payloads = await self._scroll_payloads(["week"], qfilter)
        weeks = {p["week"] for p in payloads if p.get("week") is not None}
        return sorted(weeks)

    async def list_materials(
        self, course_id: str, week: int, *, tenant_id: str,
    ) -> list[dict]:
        """Distinct materials (source files) for a course+week.

        Returns [{source_file, content_id}] — source_file is what the student
        picks; content_id is what /chat/ask needs.
        """
        tenant_id = require_tenant_id(tenant_id, operation="list_materials")
        qfilter = self._build_filter(
            tenant_id=tenant_id, course_id=course_id, weeks=[week],
        )
        payloads = await self._scroll_payloads(["source_file", "content_id"], qfilter)
        seen: dict[str, str | None] = {}
        for p in payloads:
            sf = p.get("source_file")
            if sf and sf not in seen:
                seen[sf] = p.get("content_id")
        return [
            {"source_file": sf, "content_id": cid}
            for sf, cid in sorted(seen.items())
        ]

    async def list_materials_for_weeks(
        self, course_id: str, weeks: list[int], *, tenant_id: str,
    ) -> list[dict]:
        """Materi untuk beberapa minggu sekaligus, terurut minggu lalu nama berkas.

        Setiap entri membawa `week`-nya sendiri supaya klien dapat menampilkan
        "Minggu 3 — bab3.pdf" ketika mahasiswa memilih lebih dari satu minggu dan
        nama berkas saja menjadi ambigu.
        """
        tenant_id = require_tenant_id(tenant_id, operation="list_materials_for_weeks")
        if not weeks:
            return []
        qfilter = self._build_filter(
            tenant_id=tenant_id, course_id=course_id, weeks=weeks,
        )
        payloads = await self._scroll_payloads(
            ["source_file", "content_id", "week"], qfilter
        )
        seen: dict[str, dict] = {}
        for p in payloads:
            sf = p.get("source_file")
            if sf and sf not in seen:
                seen[sf] = {
                    "source_file": sf,
                    "content_id": p.get("content_id"),
                    "week": p.get("week"),
                }
        return sorted(
            seen.values(), key=lambda m: (m["week"] if m["week"] is not None else 0,
                                          m["source_file"])
        )

    async def get_week_text(
        self, course_id: str, weeks: list[int], max_chars: int = 6000,
        *, tenant_id: str,
    ) -> str:
        """Gabungan teks materi pada minggu-minggu tertentu.

        Dipakai untuk menyimpulkan topik apa yang dibahas minggu itu, sehingga
        chatbot dapat menyebut isinya alih-alih hanya menampilkan nama berkas.
        """
        tenant_id = require_tenant_id(tenant_id, operation="get_week_text")
        if not weeks:
            return ""
        qfilter = self._build_filter(
            tenant_id=tenant_id, course_id=course_id, weeks=weeks,
        )
        payloads = await self._scroll_payloads(["text", "source_file"], qfilter)
        potongan: list[str] = []
        total = 0
        for p in payloads:
            teks = (p.get("text") or "").strip()
            if not teks:
                continue
            potongan.append(teks)
            total += len(teks)
            if total >= max_chars:
                break
        return "\n\n".join(potongan)[:max_chars]

    async def get_material_text(
        self, content_id: str, source_file: str, max_chars: int = 4000,
        *, tenant_id: str,
    ) -> str:
        """Concatenate the indexed text of one material, capped at max_chars.

        Used to auto-generate starter questions from the material content.
        """
        tenant_id = require_tenant_id(tenant_id, operation="get_material_text")
        qfilter = self._build_filter(
            tenant_id=tenant_id, content_id=content_id, source_filter=source_file,
        )
        payloads = await self._scroll_payloads(["text", "chunk_index"], qfilter)
        payloads.sort(key=lambda p: p.get("chunk_index") or 0)
        parts: list[str] = []
        total = 0
        for p in payloads:
            t = (p.get("text") or "").strip()
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(parts)[:max_chars]

    async def get_content_course(
        self, content_id: str, *, tenant_id: str,
    ) -> str | None:
        """Mata kuliah pemilik sebuah materi; None bila materinya tidak ada.

        Dibaca dari payload yang benar-benar terindeks, bukan diturunkan dari
        pola nama `content_id`. Keduanya biasanya sama, tetapi `course_id` boleh
        dikirim eksplisit saat unggah — dan bila keduanya berbeda, menebak dari
        nama akan memberi jawaban yang salah tepat pada pemeriksaan hak akses.
        """
        tenant_id = require_tenant_id(tenant_id, operation="get_content_course")
        payloads = await self._scroll_payloads(
            ["course_id"],
            self._build_filter(tenant_id=tenant_id, content_id=content_id),
        )
        for p in payloads:
            if p.get("course_id"):
                return str(p["course_id"])
        return None

    async def delete_by_source(self, source_file: str, *, tenant_id: str) -> None:
        """Delete all chunks belonging to one source file, milik tenant ini saja.

        Filter tenant di sini bukan sekadar kerapian: tanpa itu, dua pelanggan
        yang kebetulan mengunggah berkas bernama sama ("bab1.pdf") akan saling
        menghapus materi — kehilangan data permanen tanpa jejak yang jelas.
        """
        tenant_id = require_tenant_id(tenant_id, operation="delete_by_source")
        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=self._build_filter(
                    tenant_id=tenant_id, source_filter=source_file,
                )
            ),
        )
        logger.info(
            "Deleted chunks for source_file='{}' (tenant={})", source_file, tenant_id,
        )

    async def delete_tenant_data(self, *, tenant_id: str) -> None:
        """Hapus SELURUH data satu tenant dari index.

        Dibutuhkan saat pelanggan berhenti berlangganan: hak untuk dihapus bukan
        hal yang bisa dikerjakan belakangan dengan skrip manual.
        """
        tenant_id = require_tenant_id(tenant_id, operation="delete_tenant_data")
        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=self._build_filter(tenant_id=tenant_id)
            ),
        )
        logger.warning("Seluruh data index tenant '{}' dihapus", tenant_id)

    @staticmethod
    def _build_filter(
        *,
        tenant_id: str,
        source_filter: str | None = None,
        content_id: str | None = None,
        course_id: str | None = None,
        weeks: list[int] | None = None,
    ) -> qm.Filter:
        """Filter payload untuk membatasi cakupan pencarian.

        SELALU memuat `tenant_id`, dan tidak pernah mengembalikan `None`. Versi
        sebelumnya mengembalikan `None` bila tak ada penyaring — yang di Qdrant
        berarti "cari di seluruh koleksi". Dengan satu koleksi dipakai bersama,
        jalur itulah yang membocorkan materi antar kampus, dan ia sengaja
        dihilangkan sepenuhnya di sini.

        `weeks` berisi SATU ATAU LEBIH minggu — mahasiswa yang sedang menyiapkan
        ujian sering perlu membaca beberapa minggu sekaligus ("minggu 3 dan 4"),
        jadi dipakai MatchAny, bukan satu nilai.
        """
        tenant_id = require_tenant_id(tenant_id, operation="_build_filter")
        conditions: list[Any] = [
            qm.FieldCondition(key=TENANT_FIELD, match=qm.MatchValue(value=tenant_id)),
        ]
        if source_filter:
            conditions.append(
                qm.FieldCondition(key="source_file", match=qm.MatchValue(value=source_filter))
            )
        if content_id:
            conditions.append(
                qm.FieldCondition(key="content_id", match=qm.MatchValue(value=content_id))
            )
        if course_id:
            conditions.append(
                qm.FieldCondition(key="course_id", match=qm.MatchValue(value=course_id))
            )
        if weeks:
            conditions.append(
                qm.FieldCondition(key="week", match=qm.MatchAny(any=sorted(set(weeks))))
            )
        return qm.Filter(must=conditions)

    def _chunk_to_point(self, chunk: Chunk, tenant_id: str) -> qm.PointStruct:
        if chunk.dense_embedding is None:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no dense embedding; "
                "run Embedder.embed_chunks() first."
            )

        vector: dict[str, Any] = {
            _DENSE_VEC: chunk.dense_embedding
        }
        if self._enable_sparse and chunk.sparse_embedding:
            vector[_SPARSE_VEC] = qm.SparseVector(
                indices=list(chunk.sparse_embedding.keys()),
                values=list(chunk.sparse_embedding.values()),
            )

        payload = {
            TENANT_FIELD: tenant_id,
            "text": chunk.text,
            "parent_element_id": chunk.parent_element_id,
            "element_type": chunk.element_type.value,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "content_id": chunk.content_id,
            "course_id": chunk.course_id,
            "course_name": chunk.course_name,
            "week": chunk.week,
            "raw_html": chunk.raw_html,
            "image_base64": chunk.image_base64,
        }

        return qm.PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload=payload,
        )
