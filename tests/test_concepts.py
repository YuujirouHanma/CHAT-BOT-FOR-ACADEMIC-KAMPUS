"""Uji penyusunan peta konsep tingkat minggu.

Fokusnya pada bagian yang tidak bisa dinilai dengan melihat keluaran model:
pemilihan potongan saat materi melebihi anggaran, penguraian JSON yang tidak
rapi, dan pemetaan nomor potongan kembali ke berkas/halaman. Mutu konsepnya
sendiri dinilai manusia lewat scripts/lihat_konsep.py, bukan di sini.
"""
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_pipeline
from src.api.main import app
from src.indexing.concepts import (
    MAKS_KONSEP,
    PetaKonsep,
    _rakit_konsep,
    _urai_json,
    ambil_chunk_minggu,
    pilih_merata,
    susun_peta_konsep,
)

T = "kampus-uji"


def _chunk(nomor: int, berkas: str = "a.pdf", halaman: int | None = 1) -> dict:
    return {
        "chunk_id": f"c{nomor}",
        "text": f"isi {nomor}",
        "source_file": berkas,
        "page_number": halaman,
        "chunk_index": nomor,
        "element_type": "text",
    }


class TestPilihMerata:
    def test_di_bawah_anggaran_dikembalikan_utuh(self) -> None:
        chunks = [_chunk(i) for i in range(5)]
        assert pilih_merata(chunks, maks=10) == chunks

    def test_dipotong_tepat_sebanyak_anggaran(self) -> None:
        chunks = [_chunk(i) for i in range(100)]
        assert len(pilih_merata(chunks, maks=10)) == 10

    def test_menyebar_bukan_mengambil_yang_pertama(self) -> None:
        """Inti perbaikannya: cakupan harus menjangkau ujung materi.

        Mengambil N pertama persis kesalahan yang sedang diperbaiki — konsep di
        paruh akhir materi tidak akan pernah terlihat model.
        """
        chunks = [_chunk(i) for i in range(100)]
        terpilih = pilih_merata(chunks, maks=10)
        indeks = [int(c["chunk_id"][1:]) for c in terpilih]

        assert indeks[0] == 0
        assert indeks[-1] >= 85, f"potongan terakhir terlalu awal: {indeks}"
        assert indeks == sorted(indeks)

    def test_tidak_pernah_melewati_batas_daftar(self) -> None:
        chunks = [_chunk(i) for i in range(7)]
        assert len(pilih_merata(chunks, maks=3)) == 3


class TestUraiJson:
    def test_json_polos(self) -> None:
        assert _urai_json('[{"nama": "A"}]') == [{"nama": "A"}]

    def test_dibungkus_pagar_kode(self) -> None:
        assert _urai_json('```json\n[{"nama": "A"}]\n```') == [{"nama": "A"}]

    def test_diapit_kalimat_tambahan(self) -> None:
        mentah = 'Berikut hasilnya:\n[{"nama": "A"}]\nSemoga membantu.'
        assert _urai_json(mentah) == [{"nama": "A"}]

    def test_bukan_daftar_ditolak(self) -> None:
        assert _urai_json('{"nama": "A"}') == []

    def test_sampah_tidak_melempar(self) -> None:
        assert _urai_json("maaf, saya tidak bisa") == []

    def test_elemen_bukan_objek_disaring(self) -> None:
        assert _urai_json('[{"nama": "A"}, "bukan objek", 5]') == [{"nama": "A"}]


class TestRakitKonsep:
    def test_memetakan_nomor_ke_berkas_dan_halaman(self) -> None:
        chunks = [
            _chunk(0, "a.pdf", 3),
            _chunk(1, "a.pdf", 9),
            _chunk(2, "b.pdf", 2),
        ]
        hasil = _rakit_konsep(
            [{"nama": "K", "deskripsi": "d", "potongan": [1, 2, 3]}], chunks
        )

        assert len(hasil) == 1
        sumber = {s.source_file: s for s in hasil[0].sumber}
        assert sumber["a.pdf"].halaman_awal == 3
        assert sumber["a.pdf"].halaman_akhir == 9
        assert sumber["b.pdf"].halaman_awal == 2
        assert hasil[0].chunk_ids == ["c0", "c1", "c2"]

    def test_nomor_di_luar_jangkauan_diabaikan(self) -> None:
        """Model kadang menyebut potongan yang tidak ada; itu tidak boleh fatal."""
        chunks = [_chunk(0)]
        hasil = _rakit_konsep([{"nama": "K", "potongan": [1, 99, -4, "x"]}], chunks)

        assert len(hasil) == 1
        assert hasil[0].chunk_ids == ["c0"]

    def test_diurutkan_mengikuti_urutan_materi(self) -> None:
        chunks = [_chunk(i) for i in range(6)]
        hasil = _rakit_konsep(
            [
                {"nama": "Akhir", "potongan": [5, 6]},
                {"nama": "Awal", "potongan": [1, 2]},
                {"nama": "Tengah", "potongan": [3]},
            ],
            chunks,
        )

        assert [k.nama for k in hasil] == ["Awal", "Tengah", "Akhir"]
        assert [k.urutan for k in hasil] == [1, 2, 3]

    def test_tanpa_potongan_sah_didorong_ke_belakang(self) -> None:
        chunks = [_chunk(0)]
        hasil = _rakit_konsep(
            [
                {"nama": "Melayang", "potongan": []},
                {"nama": "Berlabuh", "potongan": [1]},
            ],
            chunks,
        )

        assert [k.nama for k in hasil] == ["Berlabuh", "Melayang"]

    def test_nama_kosong_dibuang(self) -> None:
        assert _rakit_konsep([{"nama": "   ", "potongan": [1]}], [_chunk(0)]) == []

    def test_dibatasi_maks_konsep(self) -> None:
        chunks = [_chunk(i) for i in range(20)]
        butir = [{"nama": f"K{i}", "potongan": [i + 1]} for i in range(12)]
        assert len(_rakit_konsep(butir, chunks)) == MAKS_KONSEP


class TestAmbilChunkMinggu:
    async def test_terurut_mengikuti_berkas_lalu_halaman(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(
            return_value=[
                {"chunk_id": "x", "payload": {
                    "text": "t", "source_file": "b.pdf", "page_number": 1,
                    "chunk_index": 0, "element_type": "text"}},
                {"chunk_id": "y", "payload": {
                    "text": "t", "source_file": "a.pdf", "page_number": 5,
                    "chunk_index": 0, "element_type": "text"}},
                {"chunk_id": "z", "payload": {
                    "text": "t", "source_file": "a.pdf", "page_number": 2,
                    "chunk_index": 0, "element_type": "text"}},
            ]
        )

        hasil = await ambil_chunk_minggu(store, "class-1", 3, tenant_id=T)

        assert [c["chunk_id"] for c in hasil] == ["z", "y", "x"]
        assert store.scroll_all.await_args.kwargs["section_ids"] == [3]
        assert store.scroll_all.await_args.kwargs["tenant_id"] == T

    async def test_chunk_tanpa_teks_dibuang(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(
            return_value=[
                {"chunk_id": "a", "payload": {"text": "  ", "source_file": "a.pdf"}},
                {"chunk_id": "b", "payload": {"text": "isi", "source_file": "a.pdf"}},
            ]
        )

        hasil = await ambil_chunk_minggu(store, "class-1", tenant_id=T)

        assert [c["chunk_id"] for c in hasil] == ["b"]


class TestSusunPetaKonsep:
    async def test_minggu_kosong_tidak_memanggil_llm(self) -> None:
        """Belum ada materi bukan kondisi luar biasa — dan tidak boleh berbiaya."""
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[])
        generator = AsyncMock()

        peta = await susun_peta_konsep(store, generator, "class-9", 1, tenant_id=T)

        assert peta.konsep == []
        assert peta.jumlah_chunk == 0
        generator._call_with_retry.assert_not_awaited()

    async def test_jalur_normal(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(
            return_value=[
                {"chunk_id": f"c{i}", "payload": {
                    "text": f"materi {i}", "source_file": "a.pdf",
                    "page_number": i + 1, "chunk_index": 0, "element_type": "text"}}
                for i in range(3)
            ]
        )
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            return_value='[{"nama": "Konsep A", "deskripsi": "d", "potongan": [1, 2]}]'
        )

        peta = await susun_peta_konsep(store, generator, "class-1", 2, tenant_id=T)

        assert peta.jumlah_chunk == 3
        assert peta.chunk_dipakai == 3
        assert peta.berkas == ["a.pdf"]
        assert peta.chunk_per_berkas == {"a.pdf": ["c0", "c1", "c2"]}
        assert [k.nama for k in peta.konsep] == ["Konsep A"]

        # Panggilan bantu di luar jalur jawaban mahasiswa: reasoning harus mati,
        # karena token berpikir berbagi jatah dengan keluaran JSON-nya.
        assert generator._call_with_retry.await_args.kwargs["reasoning"] is False

        # Materi di bawah anggaran: yang dibaca = seluruhnya.
        assert peta.total_per_berkas == {"a.pdf": 3}

    async def test_materi_melebihi_anggaran_melaporkan_total_sebenarnya(self) -> None:
        """Materi besar hanya dibaca sebagian, dan itu harus terlihat di angkanya.

        Dulu cakupan dihitung dari peta potongan yang SUDAH tersaring
        `pilih_merata`, sehingga berkas 150 potongan dilaporkan bertotal 60.
        Angkanya terlihat masuk akal dan tidak pernah memicu galat, padahal
        menyembunyikan satu-satunya hal yang perlu diketahui dosen: ada materi
        yang tidak pernah dibaca model.
        """
        from src.indexing.concepts import MAKS_CHUNK_KONTEKS

        jumlah = MAKS_CHUNK_KONTEKS * 2 + 30
        store = AsyncMock()
        store.scroll_all = AsyncMock(
            return_value=[
                {"chunk_id": f"c{i}", "payload": {
                    "text": f"materi {i}",
                    # Dua berkas supaya terbukti totalnya dihitung per berkas,
                    # bukan satu angka untuk seluruh minggu lalu dibagi rata.
                    "source_file": "a.pdf" if i % 2 == 0 else "b.pdf",
                    "page_number": i + 1, "chunk_index": 0, "element_type": "text"}}
                for i in range(jumlah)
            ]
        )
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            return_value='[{"nama": "Konsep A", "deskripsi": "d", "potongan": [1]}]'
        )

        peta = await susun_peta_konsep(store, generator, "class-1", 3, tenant_id=T)

        assert peta.jumlah_chunk == jumlah
        assert peta.chunk_dipakai == MAKS_CHUNK_KONTEKS

        # Total per berkas mengikuti materi yang SEBENARNYA ada...
        assert sum(peta.total_per_berkas.values()) == jumlah
        assert peta.total_per_berkas == {"a.pdf": (jumlah + 1) // 2, "b.pdf": jumlah // 2}

        # ...sedangkan yang disodorkan ke model jauh lebih sedikit.
        dibaca = sum(len(v) for v in peta.chunk_per_berkas.values())
        assert dibaca == MAKS_CHUNK_KONTEKS
        assert dibaca < sum(peta.total_per_berkas.values())

    async def test_keluaran_kacau_tidak_melempar(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(
            return_value=[{"chunk_id": "c0", "payload": {
                "text": "materi", "source_file": "a.pdf", "page_number": 1,
                "chunk_index": 0, "element_type": "text"}}]
        )
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(return_value="maaf, tidak bisa")

        peta = await susun_peta_konsep(store, generator, "class-1", 1, tenant_id=T)

        assert peta.konsep == []
        assert peta.jumlah_chunk == 1


@pytest.fixture
def client(override_tenant) -> Generator[TestClient, None, None]:
    """Klien dengan pipeline tiruan.

    `override_tenant` dari conftest menimpa SELURUH dependency hak akses, bukan
    hanya gerbang `require_tenant`. Tanpa itu rute yang memasang hak lebih
    spesifik membalas 403, dan uji bentuk permintaan tidak pernah sampai ke
    pemeriksaan yang dimaksud.
    """
    app.dependency_overrides[get_pipeline] = lambda: AsyncMock()
    yield TestClient(app)
    app.dependency_overrides.pop(get_pipeline, None)


class TestPreviewEndpoint:
    def test_mengembalikan_konsep_dan_cakupan(self, client: TestClient) -> None:
        peta = PetaKonsep(
            content_id="class-1",
            section_id=1,
            jumlah_chunk=4,
            chunk_dipakai=4,
            chunk_per_berkas={"a.pdf": ["c0", "c1"], "nyasar.pdf": ["c2", "c3"]},
        )
        peta.konsep = _rakit_konsep(
            [{"nama": "K", "deskripsi": "d", "potongan": [1, 2]}],
            [_chunk(0, "a.pdf", 1), _chunk(1, "a.pdf", 4)],
        )

        with patch(
            "src.api.routes.concepts.susun_peta_konsep",
            new_callable=AsyncMock,
            return_value=peta,
        ):
            resp = client.get("/concepts/preview?content_id=class-1&section_id=1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["concepts"][0]["nama"] == "K"
        assert data["concepts"][0]["jumlah_potongan"] == 2
        assert data["concepts"][0]["sumber"][0]["halaman_akhir"] == 4

        # Berkas yang tidak dipakai sama sekali harus terlihat sebagai 0/2 —
        # itu satu-satunya petunjuk bagi dosen bahwa ada materi yang diabaikan.
        cakupan = {c["source_file"]: c for c in data["cakupan"]}
        assert cakupan["nyasar.pdf"]["tercakup"] == 0
        assert cakupan["nyasar.pdf"]["total"] == 2

    def test_content_id_wajib(self, client: TestClient) -> None:
        assert client.get("/concepts/preview").status_code == 422

    def test_kegagalan_jadi_500(self, client: TestClient) -> None:
        with patch(
            "src.api.routes.concepts.susun_peta_konsep",
            new_callable=AsyncMock,
            side_effect=RuntimeError("qdrant mati"),
        ):
            resp = client.get("/concepts/preview?content_id=class-1")

        assert resp.status_code == 500
        assert "qdrant mati" not in resp.text
        # Rincian 5xx sengaja disembunyikan: pesan galat Qdrant tidak boleh
        # bocor ke pemanggil.
        assert resp.json()["detail"] == "Terjadi galat internal"
