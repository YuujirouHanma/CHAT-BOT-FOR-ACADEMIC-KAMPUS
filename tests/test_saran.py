"""Validasi saran pertanyaan terhadap materi.

Yang diuji di sini bukan "apakah fungsinya berjalan", melainkan empat keputusan
yang kalau salah akan menimbulkan kemunduran yang sulit terlihat:

1. saran yang tak terjawab DIBUANG — itu seluruh alasan modul ini ada;
2. kegagalan infrastruktur TIDAK menghapus saran — kalau salah, satu gangguan
   Qdrant mengosongkan saran di seluruh kampus tanpa jejak;
3. penyaring (minggu, materi, kelas) ikut terkirim — kalau tidak, pemeriksaannya
   mengukur pencarian yang BERBEDA dari yang akan dijalankan saat saran diklik,
   dan justru meloloskan saran yang gagal;
4. pemeriksaan yang menolak SEMUANYA mengembalikan daftar kosong, bukan daftar
   asli — kelalaian yang membuat fitur ini tampak bekerja padahal tidak.
"""
from __future__ import annotations

import pytest

from src.config import settings
from src.retrieval.saran import saring_saran
from tests.conftest import TEST_TENANT_ID as TENANT


class RetrieverSkor:
    """Retriever yang skornya ditentukan per teks query.

    Mencatat setiap panggilan beserta argumennya, karena sebagian uji di bawah
    memeriksa CARA pemeriksaan dijalankan, bukan hanya hasilnya.
    """

    def __init__(self, skor: dict[str, float], bawaan: float = 0.0) -> None:
        self._skor = skor
        self._bawaan = bawaan
        self.panggilan: list[dict] = []

    async def retrieve(self, query: str, **kw) -> list[dict]:
        self.panggilan.append({"query": query, **kw})
        skor = self._skor.get(query, self._bawaan)
        return [{"chunk_id": "c1", "rerank_score": skor, "payload": {}}]


class RetrieverTumbang:
    def __init__(self) -> None:
        self.panggilan = 0

    async def retrieve(self, query: str, **kw) -> list[dict]:
        self.panggilan += 1
        raise RuntimeError("Qdrant tidak terjangkau")


class RetrieverKosong:
    """Mengembalikan nol hasil — keadaan nyata ketika kelasnya belum terindeks."""

    async def retrieve(self, query: str, **kw) -> list[dict]:
        return []


@pytest.fixture(autouse=True)
def _ambang_tetap(monkeypatch):
    """Patok ambangnya, supaya uji tidak ikut berubah saat bawaannya disetel ulang."""
    monkeypatch.setattr(settings, "suggestion_min_score", 0.15)
    monkeypatch.setattr(settings, "suggestion_check_top_k", 10)


class TestPenyaringan:
    async def test_saran_tak_terjawab_dibuang(self) -> None:
        r = RetrieverSkor({"Apa itu indeks?": 0.9, "Siapa presiden Prancis?": 0.001})

        hasil = await saring_saran(
            ["Apa itu indeks?", "Siapa presiden Prancis?"],
            retriever=r, tenant_id=TENANT,
        )

        assert hasil == ["Apa itu indeks?"]

    async def test_urutan_asli_dipertahankan(self) -> None:
        """Model menyusun saran dari paling dekat ke paling jauh dari jawaban;
        urutan itu informasi, dan mengacaknya memindahkan saran terbaik ke bawah."""
        r = RetrieverSkor({"satu": 0.9, "dua": 0.001, "tiga": 0.8, "empat": 0.7})

        hasil = await saring_saran(
            ["satu", "dua", "tiga", "empat"], retriever=r, tenant_id=TENANT,
        )

        assert hasil == ["satu", "tiga", "empat"]

    async def test_semua_ditolak_mengembalikan_kosong_bukan_daftar_asli(self) -> None:
        """Kelalaian yang membuat fitur tampak bekerja padahal tidak: pemanggil
        yang menerima daftar asli saat semuanya ditolak tidak menyaring apa pun."""
        r = RetrieverSkor({}, bawaan=0.0)

        hasil = await saring_saran(["a", "b", "c"], retriever=r, tenant_id=TENANT)

        assert hasil == []

    async def test_hasil_pencarian_kosong_membuang_saran(self) -> None:
        hasil = await saring_saran(
            ["apa pun"], retriever=RetrieverKosong(), tenant_id=TENANT,
        )

        assert hasil == []

    async def test_tepat_di_ambang_diloloskan(self) -> None:
        """Perbandingannya >=, bukan >. Diuji karena satu karakter yang salah di
        sini membuang saran yang persis memenuhi syarat."""
        r = RetrieverSkor({"pas": 0.15})

        assert await saring_saran(["pas"], retriever=r, tenant_id=TENANT) == ["pas"]


class TestTidakMenghapusSaranKarenaGangguan:
    async def test_retriever_tumbang_mempertahankan_saran(self) -> None:
        r = RetrieverTumbang()

        hasil = await saring_saran(["a", "b"], retriever=r, tenant_id=TENANT)

        assert hasil == ["a", "b"]
        assert r.panggilan == 2, "tiap kandidat tetap dicoba, bukan menyerah di yang pertama"

    async def test_satu_gagal_tidak_menjatuhkan_yang_lain(self) -> None:
        """`asyncio.gather(return_exceptions=True)` — tanpa itu, satu kandidat
        yang gagal akan melempar dan seluruh daftar hilang."""

        class Campuran:
            async def retrieve(self, query: str, **kw):
                if query == "rusak":
                    raise RuntimeError("gagal")
                return [{"rerank_score": 0.9 if query == "bagus" else 0.0}]

        hasil = await saring_saran(
            ["bagus", "rusak", "jelek"], retriever=Campuran(), tenant_id=TENANT,
        )

        assert hasil == ["bagus", "rusak"], "yang gagal diperiksa dipertahankan"


class TestCaraPemeriksaanDijalankan:
    async def test_penyaring_diteruskan_apa_adanya(self) -> None:
        """Kalau penyaring tidak ikut, pemeriksaannya mengukur pencarian yang
        BERBEDA dari yang akan dijalankan saat saran diklik — dan meloloskan
        saran yang justru akan ditolak."""
        r = RetrieverSkor({"x": 0.9})

        await saring_saran(
            ["x"], retriever=r, content_id="kelas-3", course_id="sbd",
            weeks=[1, 2], source_filter="bab1.pdf", tenant_id=TENANT,
        )

        p = r.panggilan[0]
        assert p["content_id"] == "kelas-3"
        assert p["course_id"] == "sbd"
        assert p["weeks"] == [1, 2]
        assert p["source_filter"] == "bab1.pdf"
        assert p["tenant_id"] == TENANT

    async def test_pemeriksaan_dibuat_murah(self) -> None:
        """Hanya skor tertinggi yang dipakai memutuskan, jadi tidak ada gunanya
        membayar reranker untuk lima potongan."""
        r = RetrieverSkor({"x": 0.9})

        await saring_saran(["x"], retriever=r, tenant_id=TENANT)

        assert r.panggilan[0]["rerank_top_k"] == 1
        assert r.panggilan[0]["retrieval_top_k"] == settings.suggestion_check_top_k

    async def test_ambang_nol_mematikan_tanpa_satu_pun_pencarian(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "suggestion_min_score", 0.0)
        r = RetrieverSkor({}, bawaan=0.0)

        hasil = await saring_saran(["a", "b"], retriever=r, tenant_id=TENANT)

        assert hasil == ["a", "b"]
        assert r.panggilan == [], "pemeriksaan mati berarti tidak ada biaya sama sekali"


class TestKandidatBermasalah:
    async def test_kandidat_kosong_tidak_dikirim_ke_retriever(self) -> None:
        """`retrieve()` melempar ValueError pada query kosong. Tanpa penjagaan,
        satu entri kosong dari parser akan membuat SELURUH kandidat lain
        diloloskan tanpa diperiksa — lewat jalur 'gangguan infrastruktur'."""
        r = RetrieverSkor({"nyata": 0.9})

        hasil = await saring_saran(
            ["", "   ", "nyata"], retriever=r, tenant_id=TENANT,
        )

        assert [p["query"] for p in r.panggilan] == ["nyata"]
        assert hasil == ["nyata"], "yang kosong tidak ikut diloloskan"

    async def test_daftar_kosong_tidak_memanggil_apa_pun(self) -> None:
        r = RetrieverSkor({})

        assert await saring_saran([], retriever=r, tenant_id=TENANT) == []
        assert r.panggilan == []

    async def test_kandidat_kepanjangan_dilewati(self) -> None:
        """Kandidat sepanjang paragraf pertanda parsernya bermasalah, bukan
        saran. Mengirimnya ke embedder tidak mengukur apa pun yang berguna."""
        r = RetrieverSkor({"pendek": 0.9})
        panjang = "a" * 400

        hasil = await saring_saran(
            [panjang, "pendek"], retriever=r, tenant_id=TENANT,
        )

        assert [p["query"] for p in r.panggilan] == ["pendek"]
        assert hasil == ["pendek"]

    async def test_semua_kandidat_dilewati_mengembalikan_daftar_asli(self) -> None:
        """Beda dari "diperiksa lalu ditolak": di sini TIDAK ADA yang diperiksa,
        jadi tidak ada dasar untuk membuang apa pun."""
        r = RetrieverSkor({})

        hasil = await saring_saran(["", "  "], retriever=r, tenant_id=TENANT)

        assert hasil == ["", "  "]
        assert r.panggilan == []
