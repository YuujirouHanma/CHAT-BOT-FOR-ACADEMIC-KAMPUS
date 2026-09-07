"""Latihan koding: analisis statis, penilaian, dan penyimpanan kiriman.

Yang paling penting dijaga di sini:

1. Kode mahasiswa TIDAK PERNAH dieksekusi. Analisis hanya membaca strukturnya.
2. Galat yang sudah pasti (sintaks, fungsi wajib hilang) tidak memanggil LLM —
   diagnosisnya sudah tepat tanpa penalaran, dan memanggil model hanya membuang
   biaya serta membuat mahasiswa menunggu.
3. Kunci penilaian tidak pernah dikirim ke klien.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src import livecode as lc
from src.pipeline import RAGPipeline
from src.tenancy import TenantScopeError

A = "kampus-a"
B = "kampus-b"

LAT = lc.Exercise(
    exercise_id="dasprog-2-0-faktorial",
    title="Faktorial",
    prompt="Buat fungsi faktorial(n) yang mengembalikan n!",
    required_function="faktorial",
    forbidden_names=("math",),
    expected_behavior="mengembalikan hasil perkalian 1 sampai n",
    rubric=("logika benar", "menangani n=0"),
    test_cases=({"input": 5, "output": 120}, {"input": 0, "output": 1}),
)

BENAR = (
    "def faktorial(n):\n"
    "    hasil = 1\n"
    "    for i in range(1, n + 1):\n"
    "        hasil *= i\n"
    "    return hasil\n"
)


@pytest.fixture(autouse=True)
def _isolasi(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "SUBMISSION_DIR", tmp_path / "livecode")
    yield


class TestAnalisisStatis:
    def test_kode_benar_lolos(self) -> None:
        temuan, layak = lc.analyze_code(BENAR, LAT)
        assert not lc.has_blocking_error(temuan)
        assert layak

    def test_galat_sintaks_menyebut_nomor_baris(self) -> None:
        # Nomor baris jauh lebih berguna bagi pemula daripada "kode salah".
        temuan, layak = lc.analyze_code("def faktorial(n)\n    return 1", LAT)
        assert lc.has_blocking_error(temuan)
        assert not layak                     # tidak perlu LLM: sudah pasti salah
        assert temuan[0].line == 1

    def test_kode_kosong_ditolak(self) -> None:
        for kode in ("", "   ", "# cuma komentar\n", "pass\n"):
            temuan, layak = lc.analyze_code(kode, LAT)
            assert lc.has_blocking_error(temuan), kode
            assert not layak

    def test_fungsi_wajib_belum_ada(self) -> None:
        temuan, _ = lc.analyze_code("def factorial(n):\n    return 1", LAT)
        assert lc.has_blocking_error(temuan)
        assert "faktorial" in temuan[0].message

    def test_konstruksi_terlarang_hanya_peringatan(self) -> None:
        # Memakai `math` tidak membuat programnya salah — hanya melenceng dari
        # maksud latihan. Jadi diperingatkan, bukan diblokir.
        kode = "import math\ndef faktorial(n):\n    return math.factorial(n)"
        temuan, layak = lc.analyze_code(kode, LAT)
        assert not lc.has_blocking_error(temuan)
        assert layak
        assert any(t.severity == "peringatan" for t in temuan)

    def test_kode_terlalu_panjang_ditolak(self) -> None:
        temuan, layak = lc.analyze_code("x = 1\n" * 20_000, LAT)
        assert lc.has_blocking_error(temuan)
        assert not layak

    def test_tidak_mengeksekusi_kode(self) -> None:
        # Kode yang kalau dijalankan akan merusak: analisis harus tetap aman.
        berbahaya = (
            "import os\n"
            "def faktorial(n):\n"
            "    os.system('rm -rf /')\n"
            "    return 1\n"
        )
        temuan, layak = lc.analyze_code(berbahaya, LAT)
        # Terurai dengan selamat, tidak ada yang dijalankan.
        assert layak
        assert not lc.has_blocking_error(temuan)


class TestBentukPublik:
    def test_kunci_penilaian_tidak_ikut_terkirim(self) -> None:
        # Kalau rubrik dan keluaran kasus uji ikut dikirim, mahasiswa dapat
        # menuliskan jawabannya tanpa menulis programnya.
        publik = LAT.as_public()
        assert "rubric" not in publik
        assert "test_cases" not in publik
        teks = str(publik)
        # Keluaran kasus uji adalah jawabannya; hanya masukan yang boleh tampil.
        assert "120" not in teks
        assert "logika benar" not in teks      # isi rubrik

    def test_spesifikasi_soal_tetap_terkirim(self) -> None:
        # `expected_behavior` bukan kunci jawaban melainkan spesifikasi —
        # mahasiswa justru butuh tahu program itu harus melakukan apa.
        assert LAT.as_public()["expected_behavior"]

    def test_masukan_contoh_tetap_ditampilkan(self) -> None:
        # Contoh pemakaian membantu memahami soal tanpa membocorkan jawaban.
        assert LAT.as_public()["example_inputs"] == [5, 0]


class TestPenilaian:
    @staticmethod
    def _pipeline(review):
        gen = MagicMock()
        gen.review_code = AsyncMock(return_value=review)
        return RAGPipeline(
            summarizer=MagicMock(), chunker=MagicMock(), embedder=MagicMock(),
            store=MagicMock(), reranker=MagicMock(), generator=gen,
        ), gen

    LULUS = {
        "lulus": True, "skor": 1.0, "ragu": False, "ringkasan": "Sudah benar.",
        "benar": ["perulangan tepat"], "keliru": [], "petunjuk": [],
    }

    @pytest.mark.asyncio
    async def test_galat_statis_tidak_memanggil_llm(self) -> None:
        pipe, gen = self._pipeline(self.LULUS)
        hasil = await pipe.grade_livecode(
            LAT, "def faktorial(n)\n  return", tenant_id=A,
        )
        gen.review_code.assert_not_awaited()
        assert hasil["lulus"] is False
        assert hasil["skor"] == 0.0

    @pytest.mark.asyncio
    async def test_kode_benar_lulus(self) -> None:
        pipe, gen = self._pipeline(self.LULUS)
        hasil = await pipe.grade_livecode(LAT, BENAR, tenant_id=A)
        gen.review_code.assert_awaited_once()
        assert hasil["lulus"] is True
        assert hasil["skor"] == 1.0
        assert hasil["benar"] == ["perulangan tepat"]

    @pytest.mark.asyncio
    async def test_tinjauan_gagal_ditandai_bukan_dinolkan(self) -> None:
        pipe, _ = self._pipeline(None)
        hasil = await pipe.grade_livecode(LAT, BENAR, tenant_id=A)
        assert hasil["perlu_tinjauan_dosen"] is True
        assert hasil["lulus"] is False

    @pytest.mark.asyncio
    async def test_llm_ragu_ditandai(self) -> None:
        pipe, _ = self._pipeline({**self.LULUS, "ragu": True})
        hasil = await pipe.grade_livecode(LAT, BENAR, tenant_id=A)
        assert hasil["perlu_tinjauan_dosen"] is True

    @pytest.mark.asyncio
    async def test_umpan_balik_tidak_memuat_kode_jadi(self) -> None:
        # Petunjuk mengarahkan; menyodorkan jawaban menghapus proses belajarnya.
        pipe, _ = self._pipeline({
            "lulus": False, "skor": 0.5, "ragu": False,
            "ringkasan": "hampir", "benar": ["struktur loop benar"],
            "keliru": [{"baris": 3, "masalah": "batas range", "akibat": "kurang 1"}],
            "petunjuk": ["periksa batas atas range"],
        })
        hasil = await pipe.grade_livecode(LAT, BENAR, tenant_id=A)
        assert hasil["keliru"][0]["baris"] == 3
        assert hasil["petunjuk"] == ["periksa batas atas range"]

    @pytest.mark.asyncio
    async def test_tanpa_tenant_ditolak(self) -> None:
        pipe, _ = self._pipeline(self.LULUS)
        with pytest.raises(TenantScopeError):
            await pipe.grade_livecode(LAT, BENAR, tenant_id="")


class TestPenyimpananKiriman:
    def test_kiriman_gagal_ikut_tersimpan(self) -> None:
        # Berapa kali mahasiswa mencoba sebelum berhasil adalah data paling
        # berguna untuk menilai apakah sebuah gaya belajar membantu.
        for lulus in (False, False, True):
            lc.record_submission(
                tenant_id=A, exercise_id="e1", student_id="2021001",
                code=BENAR, result={"lulus": lulus, "skor": 1.0 if lulus else 0.2},
            )
        riwayat = lc.list_submissions(tenant_id=A)
        assert len(riwayat) == 3
        assert [r["lulus"] for r in riwayat] == [True, False, False]  # terbaru dulu

    def test_statistik_percobaan(self) -> None:
        for lulus in (False, True, True, False):
            lc.record_submission(
                tenant_id=A, exercise_id="e1", student_id="s",
                code="x=1", result={"lulus": lulus, "skor": 0.0},
            )
        s = lc.attempt_stats(tenant_id=A, exercise_id="e1")
        assert s["total_kiriman"] == 4
        assert s["lulus"] == 2
        assert s["rasio_lulus"] == 0.5

    def test_statistik_kosong_tidak_mengaku_nol(self) -> None:
        s = lc.attempt_stats(tenant_id=A, exercise_id="belum-ada")
        assert s["total_kiriman"] == 0
        assert s["rasio_lulus"] is None

    def test_identitas_mahasiswa_disamarkan(self) -> None:
        lc.record_submission(
            tenant_id=A, exercise_id="e1", student_id="2021001234",
            code="x=1", result={"lulus": True, "skor": 1.0},
        )
        tersimpan = lc.list_submissions(tenant_id=A)[0]
        assert tersimpan["student_id"] != "2021001234"

    def test_isolasi_antar_tenant(self) -> None:
        lc.record_submission(
            tenant_id=A, exercise_id="e1", student_id="s",
            code="milik A", result={"lulus": True, "skor": 1.0},
        )
        assert lc.list_submissions(tenant_id=B) == []
        assert len(lc.list_submissions(tenant_id=A)) == 1

    def test_tanpa_tenant_ditolak(self) -> None:
        with pytest.raises(TenantScopeError):
            lc.list_submissions(tenant_id="")
