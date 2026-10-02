"""`scroll_all` dan `delete_by_content_id` di atas Qdrant sungguhan.

Dua fungsi ini dipindahkan dari layanan tim back-end, dan pemanggilnya memakai
kata `section_ids` sementara payload di sini bernama `week`. Perbedaan nama
seperti itu TIDAK akan tertangkap oleh penyimpanan tiruan: tiruan menerima
argumen apa pun dan mengembalikan apa pun yang disetel. Karena itu berkas ini
memakai Qdrant in-memory yang benar-benar menyaring.
"""
from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from src.schemas import Chunk, ElementType
from src.storage.qdrant_store import QdrantStore

KAMPUS_A = "kampus-a"
KAMPUS_B = "kampus-b"


def _chunk(label: str, *, content_id: str, week: int, teks: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid5(uuid.NAMESPACE_OID, label)),
        text=teks,
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file=f"{content_id}.pdf",
        content_id=content_id,
        course_id=content_id.split("-")[0],
        week=week,
        dense_embedding=[0.1] * 1024,
    )


@pytest.fixture
async def store() -> QdrantStore:
    """Dua kampus, dua materi, tiga minggu — cukup untuk menguji tiap saringan."""
    s = QdrantStore(
        client=QdrantClient(":memory:"), collection="uji_scroll", enable_sparse=False,
    )
    await s.ensure_collection()
    await s.upsert_chunks(
        [
            _chunk("a1", content_id="sbd-minggu-1", week=1, teks="Normalisasi"),
            _chunk("a2", content_id="sbd-minggu-2", week=2, teks="Relasi"),
            _chunk("a3", content_id="sbd-minggu-3", week=3, teks="Indeks"),
        ],
        tenant_id=KAMPUS_A,
    )
    await s.upsert_chunks(
        [_chunk("b1", content_id="sbd-minggu-1", week=1, teks="Milik kampus lain")],
        tenant_id=KAMPUS_B,
    )
    return s


class TestScrollAll:
    async def test_tanpa_saringan_mengambil_semua_milik_kampus(self, store) -> None:
        hasil = await store.scroll_all(tenant_id=KAMPUS_A)
        assert len(hasil) == 3

    async def test_section_ids_menyaring_menurut_minggu(self, store) -> None:
        """Inti pemetaannya: `section_ids` di luar, medan `week` di dalam.

        Kalau pemetaan itu salah, saringan tidak cocok dengan apa pun dan
        fungsi ini mengembalikan daftar kosong tanpa galat — kegagalan paling
        sulit disadari, karena tampak seperti materi yang memang belum ada.
        """
        hasil = await store.scroll_all(section_ids=[2, 3], tenant_id=KAMPUS_A)
        assert len(hasil) == 2
        teks = {h["payload"]["text"] for h in hasil}
        assert teks == {"Relasi", "Indeks"}

    async def test_satu_minggu_saja(self, store) -> None:
        hasil = await store.scroll_all(section_ids=[1], tenant_id=KAMPUS_A)
        assert [h["payload"]["text"] for h in hasil] == ["Normalisasi"]

    async def test_content_id_menyaring_satu_materi(self, store) -> None:
        hasil = await store.scroll_all(content_id="sbd-minggu-2", tenant_id=KAMPUS_A)
        assert [h["payload"]["text"] for h in hasil] == ["Relasi"]

    async def test_tidak_pernah_melewati_batas_kampus(self, store) -> None:
        """content_id yang sama ada di dua kampus; yang terbaca hanya milik sendiri."""
        a = await store.scroll_all(content_id="sbd-minggu-1", tenant_id=KAMPUS_A)
        b = await store.scroll_all(content_id="sbd-minggu-1", tenant_id=KAMPUS_B)
        assert [x["payload"]["text"] for x in a] == ["Normalisasi"]
        assert [x["payload"]["text"] for x in b] == ["Milik kampus lain"]

    async def test_payload_fields_membatasi_medan_yang_terbawa(self, store) -> None:
        """Mengambil seluruh payload saat yang dibutuhkan satu medan itu boros.

        Teks potongan adalah bagian terbesar payload, dan pemanggil seperti
        daftar berkas terindeks tidak memerlukannya sama sekali.
        """
        hasil = await store.scroll_all(
            payload_fields=["source_file"], tenant_id=KAMPUS_A,
        )
        assert hasil
        assert set(hasil[0]["payload"]) == {"source_file"}

    async def test_minggu_yang_tidak_ada_mengembalikan_kosong(self, store) -> None:
        assert await store.scroll_all(section_ids=[99], tenant_id=KAMPUS_A) == []


class TestDeleteByContentId:
    async def test_menghapus_satu_materi_saja(self, store) -> None:
        await store.delete_by_content_id("sbd-minggu-2", tenant_id=KAMPUS_A)
        sisa = {h["payload"]["content_id"] for h in await store.scroll_all(tenant_id=KAMPUS_A)}
        assert sisa == {"sbd-minggu-1", "sbd-minggu-3"}

    async def test_tidak_menyentuh_kampus_lain(self, store) -> None:
        """Kampus lain hampir pasti punya content_id yang sama.

        Tanpa saringan tenant, indeks ulang satu kampus akan menghapus materi
        kampus lain — dan tidak ada yang tahu sampai mahasiswanya bertanya.
        """
        await store.delete_by_content_id("sbd-minggu-1", tenant_id=KAMPUS_A)
        b = await store.scroll_all(content_id="sbd-minggu-1", tenant_id=KAMPUS_B)
        assert [x["payload"]["text"] for x in b] == ["Milik kampus lain"]

    async def test_content_id_kosong_ditolak(self, store) -> None:
        """Tanpa penjagaan ini, saringannya hanya tenant — seluruh materi hilang."""
        with pytest.raises(ValueError):
            await store.delete_by_content_id("", tenant_id=KAMPUS_A)
