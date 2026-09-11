"""Uji regresi: mengindeks ulang sebuah berkas harus MENGGANTI potongannya.

Cacat aslinya: `chunk_id` dibuat acak oleh chunker dan `index_document` tidak
pernah menghapus titik lama, sehingga setiap indeks ulang menggandakan seluruh
potongan berkas itu. Terukur di index nyata: 44 titik, hanya 21 teks unik,
dengan 8 kelompok kembar di cakupan yang persis sama.

Uji di sini memakai Qdrant sungguhan (mode memori), bukan tiruan — yang dikunci
adalah perilaku penyaring Qdrant itu sendiri, termasuk kasus `content_id` kosong.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from src.schemas import Chunk, ElementType
from src.storage.qdrant_store import QdrantStore

A = "kampus-a"
B = "kampus-b"


def _chunk(label: str, *, source: str, content_id: str | None, teks: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_OID, label)),
        text=teks,
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file=source,
        page_number=1,
        content_id=content_id,
        dense_embedding=[0.1] * 1024,
    )


async def _teks(store: QdrantStore, tenant_id: str) -> list[str]:
    hasil = await store.search([0.1] * 1024, top_k=100, tenant_id=tenant_id)
    return sorted(r["payload"]["text"] for r in hasil)


@pytest.fixture
async def store() -> QdrantStore:
    s = QdrantStore(
        client=QdrantClient(":memory:"), collection="uji_indeks_ulang",
        enable_sparse=False,
    )
    await s.ensure_collection()
    return s


class TestIndeksUlang:
    async def test_potongan_lama_hilang_potongan_baru_tetap(
        self, store: QdrantStore,
    ) -> None:
        lama = [_chunk("lama1", source="bab1.pdf", content_id="sbd-minggu-1", teks="versi lama")]
        baru = [_chunk("baru1", source="bab1.pdf", content_id="sbd-minggu-1", teks="versi baru")]
        await store.upsert_chunks(lama, tenant_id=A)
        await store.upsert_chunks(baru, tenant_id=A)

        await store.delete_stale_chunks(
            "bab1.pdf", "sbd-minggu-1", [c.chunk_id for c in baru], tenant_id=A,
        )

        assert await _teks(store, A) == ["versi baru"]

    async def test_salinan_sah_di_content_id_lain_tidak_tersapu(
        self, store: QdrantStore,
    ) -> None:
        """Deck yang sama dipakai di dua minggu harus tetap ada di minggu lainnya."""
        await store.upsert_chunks(
            [_chunk("m2", source="mit.pdf", content_id="dasprog-minggu-2",
                    teks="salinan minggu 2")],
            tenant_id=A,
        )
        baru = [_chunk("m7", source="mit.pdf", content_id="dasprog-minggu-7",
                       teks="salinan minggu 7")]
        await store.upsert_chunks(baru, tenant_id=A)

        await store.delete_stale_chunks(
            "mit.pdf", "dasprog-minggu-7", [c.chunk_id for c in baru], tenant_id=A,
        )

        assert await _teks(store, A) == ["salinan minggu 2", "salinan minggu 7"]

    async def test_content_id_kosong_tidak_menyapu_cakupan_bernama(
        self, store: QdrantStore,
    ) -> None:
        """Unggahan tanpa content_id hanya membersihkan sesamanya yang juga tanpa content_id."""
        await store.upsert_chunks(
            [_chunk("bernama", source="bab1.pdf", content_id="sbd-minggu-1", teks="bernama")],
            tenant_id=A,
        )
        await store.upsert_chunks(
            [_chunk("kosong-lama", source="bab1.pdf", content_id=None, teks="kosong lama")],
            tenant_id=A,
        )
        baru = [_chunk("kosong-baru", source="bab1.pdf", content_id=None, teks="kosong baru")]
        await store.upsert_chunks(baru, tenant_id=A)

        await store.delete_stale_chunks(
            "bab1.pdf", None, [c.chunk_id for c in baru], tenant_id=A,
        )

        assert await _teks(store, A) == ["bernama", "kosong baru"]

    async def test_tenant_lain_tidak_tersentuh(self, store: QdrantStore) -> None:
        await store.upsert_chunks(
            [_chunk("milik-b", source="bab1.pdf", content_id="sbd-minggu-1", teks="milik B")],
            tenant_id=B,
        )
        baru = [_chunk("milik-a", source="bab1.pdf", content_id="sbd-minggu-1", teks="milik A")]
        await store.upsert_chunks(baru, tenant_id=A)

        await store.delete_stale_chunks(
            "bab1.pdf", "sbd-minggu-1", [c.chunk_id for c in baru], tenant_id=A,
        )

        assert await _teks(store, B) == ["milik B"]

    async def test_daftar_kosong_ditolak(self, store: QdrantStore) -> None:
        """keep_ids kosong berarti menghapus seluruh cakupan — bukan pembersihan."""
        await store.upsert_chunks(
            [_chunk("x", source="bab1.pdf", content_id="sbd-minggu-1", teks="jangan hilang")],
            tenant_id=A,
        )
        with pytest.raises(ValueError, match="keep_ids kosong"):
            await store.delete_stale_chunks("bab1.pdf", "sbd-minggu-1", [], tenant_id=A)
        assert await _teks(store, A) == ["jangan hilang"]
