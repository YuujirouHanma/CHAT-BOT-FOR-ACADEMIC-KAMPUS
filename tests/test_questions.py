"""Uji penyusunan bank soal per konsep.

Yang diuji di sini adalah hal-hal yang tidak bisa dinilai dengan membaca
keluaran model: penyaringan potongan ke rentang halaman sebuah konsep,
penolakan soal yang cacat, dan pemulihan saat satu konsep gagal sementara
sisanya berhasil. Mutu soalnya sendiri dinilai dosen di panel tinjau, bukan di
sini — memang itu gunanya panel itu ada.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_pipeline
from src.api.main import app
from src.indexing.questions import (
    JUMLAH_OPSI,
    SOAL_PER_KONSEP_DEFAULT,
    SOAL_PER_KONSEP_MAKS,
    BankSoal,
    KonsepMasukan,
    OpsiSoal,
    SoalKonsep,
    _rakit_soal,
    saring_chunk_konsep,
    susun_bank_soal,
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


def _titik(chunk: dict) -> dict:
    """Bentuk apa adanya dari Qdrant, sebelum diolah ambil_chunk_minggu."""
    return {
        "chunk_id": chunk["chunk_id"],
        "payload": {
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "element_type": chunk["element_type"],
        },
    }


def _soal_json(tipe: str = "ingatan", kunci: int = 0) -> dict:
    return {
        "tipe": tipe,
        "pertanyaan": f"pertanyaan {tipe}?",
        "opsi": ["alfa", "beta", "gama", "delta"],
        "kunci": kunci,
        "pembahasan": "karena begitu",
    }


class TestSaringChunkKonsep:
    def test_tanpa_sumber_mengembalikan_kosong(self) -> None:
        """Konsep yang ditambahkan dosen sendiri tidak punya concept_source.

        Yang memutuskan artinya adalah pemanggil (ia mundur ke materi minggu),
        bukan penyaring ini.
        """
        assert saring_chunk_konsep([_chunk(0)], []) == []

    def test_hanya_berkas_yang_disebut(self) -> None:
        chunks = [_chunk(0, "a.pdf"), _chunk(1, "b.pdf")]
        hasil = saring_chunk_konsep(chunks, [{"source_file": "a.pdf"}])
        assert [c["chunk_id"] for c in hasil] == ["c0"]

    def test_menghormati_rentang_halaman(self) -> None:
        chunks = [_chunk(i, "a.pdf", halaman=i) for i in range(1, 8)]
        hasil = saring_chunk_konsep(
            chunks, [{"source_file": "a.pdf", "halaman_awal": 3, "halaman_akhir": 5}]
        )
        assert [c["page_number"] for c in hasil] == [3, 4, 5]

    def test_halaman_kosong_ikut_terambil(self) -> None:
        """Berkas .txt tidak punya nomor halaman.

        Menolaknya berarti membuang materi yang sah hanya karena formatnya
        tidak bernomor — konsep dari berkas teks jadi tidak pernah dapat soal.
        """
        chunks = [_chunk(0, "catatan.txt", halaman=None)]
        hasil = saring_chunk_konsep(
            chunks, [{"source_file": "catatan.txt", "halaman_awal": 2, "halaman_akhir": 4}]
        )
        assert len(hasil) == 1

    def test_dua_rentang_pada_berkas_sama_digabung(self) -> None:
        chunks = [_chunk(i, "a.pdf", halaman=i) for i in range(1, 10)]
        hasil = saring_chunk_konsep(
            chunks,
            [
                {"source_file": "a.pdf", "halaman_awal": 1, "halaman_akhir": 2},
                {"source_file": "a.pdf", "halaman_awal": 8, "halaman_akhir": 9},
            ],
        )
        assert [c["page_number"] for c in hasil] == [1, 2, 8, 9]

    def test_tidak_menggandakan_chunk_yang_cocok_dua_rentang(self) -> None:
        chunks = [_chunk(1, "a.pdf", halaman=3)]
        hasil = saring_chunk_konsep(
            chunks,
            [
                {"source_file": "a.pdf", "halaman_awal": 1, "halaman_akhir": 5},
                {"source_file": "a.pdf", "halaman_awal": 2, "halaman_akhir": 4},
            ],
        )
        assert len(hasil) == 1


class TestRakitSoal:
    _konsep = KonsepMasukan(concept_id=7, nama="Konsep A")

    def test_jalur_normal(self) -> None:
        hasil = _rakit_soal(
            [_soal_json("ingatan", 0), _soal_json("penerapan", 2)], self._konsep, acak=False
        )

        assert len(hasil) == SOAL_PER_KONSEP_DEFAULT
        assert [s.tipe for s in hasil] == ["ingatan", "penerapan"]
        assert [s.urutan for s in hasil] == [1, 2]
        assert all(s.concept_id == 7 for s in hasil)
        assert hasil[0].opsi[0].benar is True
        assert hasil[1].opsi[2].benar is True
        assert sum(o.benar for o in hasil[0].opsi) == 1

    def test_jumlah_opsi_salah_dibuang(self) -> None:
        rusak = _soal_json()
        rusak["opsi"] = ["a", "b"]
        assert _rakit_soal([rusak], self._konsep) == []

    def test_opsi_kosong_dibuang(self) -> None:
        rusak = _soal_json()
        rusak["opsi"] = ["alfa", "", "gama", "delta"]
        assert _rakit_soal([rusak], self._konsep) == []

    def test_opsi_kembar_dibuang(self) -> None:
        """Dua pilihan yang sama membuat soal tidak punya satu jawaban benar."""
        rusak = _soal_json()
        rusak["opsi"] = ["alfa", "Alfa", "gama", "delta"]
        assert _rakit_soal([rusak], self._konsep) == []

    def test_kunci_di_luar_jangkauan_dibuang(self) -> None:
        assert _rakit_soal([_soal_json(kunci=9)], self._konsep) == []
        assert _rakit_soal([_soal_json(kunci=-1)], self._konsep) == []

    def test_kunci_bukan_angka_dibuang(self) -> None:
        rusak = _soal_json()
        rusak["kunci"] = "A"
        assert _rakit_soal([rusak], self._konsep) == []

    def test_pertanyaan_kosong_dibuang(self) -> None:
        rusak = _soal_json()
        rusak["pertanyaan"] = "   "
        assert _rakit_soal([rusak], self._konsep) == []

    def test_soal_rusak_tidak_ikut_membuang_yang_sehat(self) -> None:
        """Satu soal penerapan yang cacat tidak boleh membatalkan soal ingatan."""
        rusak = _soal_json("penerapan")
        rusak["opsi"] = ["a"]

        hasil = _rakit_soal([_soal_json("ingatan"), rusak], self._konsep)

        assert len(hasil) == 1
        assert hasil[0].tipe == "ingatan"

    def test_tipe_tak_dikenal_diisi_menurut_urutan(self) -> None:
        """Tipe yang ngawur tidak membuang soalnya — posisinya yang menentukan.

        Prompt sudah meminta urutan ingatan lalu penerapan, jadi soal pertama
        yang tipenya tidak dikenal memang soal ingatan.
        """
        aneh = [_soal_json(), _soal_json()]
        for s in aneh:
            s["tipe"] = "analisis"

        hasil = _rakit_soal(aneh, self._konsep)

        assert [s.tipe for s in hasil] == ["ingatan", "penerapan"]

    def test_tipe_sah_dari_model_dipertahankan(self) -> None:
        hasil = _rakit_soal([_soal_json("penerapan")], self._konsep)
        assert hasil[0].tipe == "penerapan"

    def test_kelebihan_soal_dipangkas(self) -> None:
        hasil = _rakit_soal([_soal_json() for _ in range(5)], self._konsep)
        assert len(hasil) == SOAL_PER_KONSEP_DEFAULT

    def test_jumlah_soal_kustom_dihormati(self) -> None:
        hasil = _rakit_soal(
            [_soal_json("ingatan"), _soal_json("penerapan"), _soal_json("ingatan"),
             _soal_json("penerapan")],
            self._konsep,
            jumlah_soal=4,
        )
        assert len(hasil) == 4

    def test_kelebihan_soal_dipangkas_sesuai_jumlah_soal(self) -> None:
        hasil = _rakit_soal([_soal_json() for _ in range(5)], self._konsep, jumlah_soal=3)
        assert len(hasil) == 3

    def test_tipe_tak_dikenal_beralih_lebih_dari_dua_soal(self) -> None:
        """Dengan jumlah_soal > 2, tipe ngawur tidak boleh menumpuk di satu tipe.

        Sebelum ada jumlah_soal kustom, tipe tak dikenal capping ke tipe
        terakhir tidak masalah karena selalu tepat dua soal. Dengan lebih dari
        dua, capping akan membuat soal ketiga dst semuanya 'penerapan'.
        """
        aneh = [_soal_json() for _ in range(4)]
        for s in aneh:
            s["tipe"] = "analisis"

        hasil = _rakit_soal(aneh, self._konsep, jumlah_soal=4)

        assert [s.tipe for s in hasil] == ["ingatan", "penerapan", "ingatan", "penerapan"]

    def test_pengacakan_memindahkan_kunci_dari_posisi_pertama(self) -> None:
        """Model cenderung menaruh jawaban benar di A.

        Bank soal yang kuncinya selalu di posisi pertama bisa dikerjakan tanpa
        membaca soalnya sama sekali, jadi posisi ini harus benar-benar berubah.
        """
        posisi = set()
        for _ in range(40):
            hasil = _rakit_soal([_soal_json(kunci=0)], self._konsep, acak=True)
            posisi.add(next(i for i, o in enumerate(hasil[0].opsi) if o.benar))

        assert len(posisi) > 1
        assert all(sum(o.benar for o in _rakit_soal(
            [_soal_json(kunci=0)], self._konsep)[0].opsi) == 1 for _ in range(5))


class TestSusunBankSoal:
    _konsep = [KonsepMasukan(concept_id=1, nama="K1", sumber=[{"source_file": "a.pdf"}])]

    async def test_tanpa_konsep_tidak_menyentuh_qdrant(self) -> None:
        store = AsyncMock()
        generator = AsyncMock()

        bank = await susun_bank_soal(store, generator, "class-1", [], section_id=2, tenant_id=T)

        assert bank.soal == []
        store.scroll_all.assert_not_awaited()
        generator._call_with_retry.assert_not_awaited()

    async def test_minggu_kosong_tidak_memanggil_llm(self) -> None:
        """Belum ada materi bukan kondisi luar biasa — dan tidak boleh berbiaya."""
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[])
        generator = AsyncMock()

        bank = await susun_bank_soal(
            store, generator, "class-1", self._konsep, section_id=2, tenant_id=T,
        )

        assert bank.soal == []
        assert bank.konsep_gagal == ["K1"]
        generator._call_with_retry.assert_not_awaited()

    async def test_jalur_normal(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(i)) for i in range(3)])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            return_value=json.dumps([_soal_json("ingatan"), _soal_json("penerapan")])
        )

        bank = await susun_bank_soal(
            store, generator, "class-1", self._konsep, section_id=2, tenant_id=T,
        )

        assert len(bank.soal) == SOAL_PER_KONSEP_DEFAULT
        assert bank.konsep_gagal == []
        assert all(s.concept_id == 1 for s in bank.soal)
        assert all(len(s.opsi) == JUMLAH_OPSI for s in bank.soal)

        # Panggilan bantu di luar jalur jawaban mahasiswa: reasoning harus mati,
        # karena token berpikir berbagi jatah dengan keluaran JSON-nya.
        assert generator._call_with_retry.await_args.kwargs["reasoning"] is False

    async def test_jumlah_soal_mengubah_anggaran_token(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(i)) for i in range(3)])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(return_value="[]")

        await susun_bank_soal(
            store, generator, "class-1", self._konsep, section_id=2, jumlah_soal=5,
            tenant_id=T,
        )

        assert generator._call_with_retry.await_args.kwargs["max_tokens"] == 200 + 900 * 5

    async def test_jumlah_soal_diklem_ke_batas_atas(self) -> None:
        """API sudah menolak > 10 (lihat TestQuestionsEndpoint), tapi fungsi ini
        dipanggil dari lebih dari satu jalur — klem di sini adalah jaring
        pengaman kedua, bukan gantinya validasi skema."""
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(i)) for i in range(3)])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(return_value="[]")

        await susun_bank_soal(
            store, generator, "class-1", self._konsep, section_id=2, jumlah_soal=999,
            tenant_id=T,
        )

        anggaran = generator._call_with_retry.await_args.kwargs["max_tokens"]
        assert anggaran == 200 + 900 * SOAL_PER_KONSEP_MAKS

    async def test_satu_panggilan_llm_per_konsep(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(i)) for i in range(3)])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(return_value="[]")

        konsep = [
            KonsepMasukan(concept_id=i, nama=f"K{i}", sumber=[{"source_file": "a.pdf"}])
            for i in range(3)
        ]
        await susun_bank_soal(store, generator, "class-1", konsep, section_id=2, tenant_id=T)

        assert generator._call_with_retry.await_count == 3
        # Materinya dibaca sekali untuk seluruh konsep, bukan sekali per konsep.
        assert store.scroll_all.await_count == 1

    async def test_konsep_gagal_tidak_menjatuhkan_yang_lain(self) -> None:
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(0))])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            side_effect=[
                '[{"tipe":"ingatan","pertanyaan":"p","opsi":["a","b","c","d"],'
                '"kunci":1,"pembahasan":"x"}]',
                RuntimeError("model tumbang"),
                "maaf, tidak bisa",
            ]
        )

        konsep = [
            KonsepMasukan(concept_id=i, nama=f"K{i}", sumber=[{"source_file": "a.pdf"}])
            for i in range(3)
        ]
        bank = await susun_bank_soal(
            store, generator, "class-1", konsep, section_id=2, tenant_id=T,
        )

        assert len(bank.soal) == 1
        assert bank.soal[0].concept_id == 0
        assert bank.konsep_gagal == ["K1", "K2"]

    async def test_konsep_tanpa_sumber_mundur_ke_materi_minggu(self) -> None:
        """Konsep yang ditambahkan dosen sendiri tetap harus dapat soal.

        Kalau tidak, konsep yang justru paling dipercaya dosen — karena ia yang
        menulisnya — malah jadi satu-satunya yang tak pernah terukur.
        """
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(i)) for i in range(3)])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            return_value='[{"tipe":"ingatan","pertanyaan":"p","opsi":["a","b","c","d"],'
            '"kunci":0,"pembahasan":"x"}]'
        )

        bank = await susun_bank_soal(
            store, generator, "class-1",
            [KonsepMasukan(concept_id=9, nama="Tambahan Dosen", sumber=[])],
            section_id=2,
            tenant_id=T,
        )

        assert len(bank.soal) == 1
        assert bank.konsep_gagal == []

    async def test_sumber_tak_dikenal_juga_mundur(self) -> None:
        """Berkas sudah dihapus setelah peta disusun — konsepnya masih ada."""
        store = AsyncMock()
        store.scroll_all = AsyncMock(return_value=[_titik(_chunk(0, "b.pdf"))])
        generator = AsyncMock()
        generator._call_with_retry = AsyncMock(
            return_value='[{"tipe":"ingatan","pertanyaan":"p","opsi":["a","b","c","d"],'
            '"kunci":0,"pembahasan":"x"}]'
        )

        bank = await susun_bank_soal(
            store, generator, "class-1",
            [KonsepMasukan(concept_id=1, nama="K", sumber=[{"source_file": "sudah-hilang.pdf"}])],
            section_id=2,
            tenant_id=T,
        )

        assert len(bank.soal) == 1


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


def _permintaan() -> dict:
    return {
        "content_id": "class-1",
        "section_id": 3,
        "class_name": "Basis Data",
        "section_name": "Minggu 3",
        "concepts": [
            {
                "concept_id": 11,
                "nama": "Normalisasi",
                "deskripsi": "d",
                "sumber": [{"source_file": "a.pdf", "halaman_awal": 1, "halaman_akhir": 4}],
            }
        ],
    }


class TestQuestionsEndpoint:
    def test_mengembalikan_soal_beserta_kunci(self, client: TestClient) -> None:
        bank = BankSoal(
            content_id="class-1",
            section_id=3,
            soal=[
                SoalKonsep(
                    concept_id=11,
                    nama_konsep="Normalisasi",
                    tipe="penerapan",
                    pertanyaan="p?",
                    opsi=[OpsiSoal(teks="a"), OpsiSoal(teks="b", benar=True)],
                    pembahasan="x",
                    urutan=1,
                )
            ],
            konsep_gagal=["Konsep Sepi"],
        )

        with patch(
            "src.api.routes.concepts.susun_bank_soal",
            new_callable=AsyncMock,
            return_value=bank,
        ):
            resp = client.post("/concepts/questions", json=_permintaan())

        assert resp.status_code == 200
        data = resp.json()
        assert data["questions"][0]["tipe"] == "penerapan"
        assert data["questions"][0]["opsi"][1]["benar"] is True
        # Konsep yang tidak menghasilkan soal harus terlihat dosen: tanpa ini,
        # ada bagian minggu yang diam-diam tidak akan pernah terukur.
        assert data["konsep_gagal"] == ["Konsep Sepi"]

    def test_konsep_wajib_ada(self, client: TestClient) -> None:
        permintaan = _permintaan()
        permintaan["concepts"] = []
        assert client.post("/concepts/questions", json=permintaan).status_code == 422

    def test_content_id_wajib(self, client: TestClient) -> None:
        permintaan = _permintaan()
        del permintaan["content_id"]
        assert client.post("/concepts/questions", json=permintaan).status_code == 422

    def test_jumlah_soal_di_atas_batas_ditolak(self, client: TestClient) -> None:
        permintaan = _permintaan()
        permintaan["jumlah_soal"] = SOAL_PER_KONSEP_MAKS + 1
        assert client.post("/concepts/questions", json=permintaan).status_code == 422

    def test_jumlah_soal_nol_ditolak(self, client: TestClient) -> None:
        permintaan = _permintaan()
        permintaan["jumlah_soal"] = 0
        assert client.post("/concepts/questions", json=permintaan).status_code == 422

    def test_kegagalan_jadi_500(self, client: TestClient) -> None:
        with patch(
            "src.api.routes.concepts.susun_bank_soal",
            new_callable=AsyncMock,
            side_effect=RuntimeError("model tumbang"),
        ):
            resp = client.post("/concepts/questions", json=_permintaan())

        assert resp.status_code == 500
