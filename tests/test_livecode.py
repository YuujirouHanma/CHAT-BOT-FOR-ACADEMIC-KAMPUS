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

from src import gen_cache
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
    """Kiriman DAN definisi latihan diarahkan ke direktori sementara.

    Keduanya menulis ke disk. Tanpa pengalihan ini, menjalankan uji akan
    mengotori cache sungguhan — dan uji berikutnya membaca sisa uji sebelumnya.
    """
    monkeypatch.setattr(lc, "SUBMISSION_DIR", tmp_path / "livecode")
    monkeypatch.setattr(gen_cache, "_CACHE_DIR", tmp_path / "gen_cache")
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


class TestIdentitasLatihan:
    """`exercise_id` harus menamai SOAL, bukan posisinya dalam daftar.

    Versi sebelumnya memakai nomor urut: `dasprog-2-0-faktorial`. Akibatnya
    identitas sebuah latihan ikut bergantung pada berapa banyak latihan yang
    kebetulan diminta — mendaftar dengan `count=5` lalu mengirim kode dengan
    `count=3` menyusun himpunan soal yang berbeda, dan nomor urut yang sama
    menunjuk ke soal yang lain sama sekali.
    """

    BUTIR_A = {
        "question": "Buat fungsi faktorial(n)",
        "expected_behavior": "mengembalikan n!",
        "required_function": "faktorial",
    }
    BUTIR_B = {
        "question": "Buat fungsi fibonacci(n)",
        "expected_behavior": "mengembalikan suku ke-n",
        "required_function": "fibonacci",
    }

    BUTIR_C = {
        "question": "Buat fungsi prima(n)",
        "expected_behavior": "True bila n prima",
        "required_function": "prima",
    }

    @staticmethod
    def _pipeline_dengan(hasil_per_count: dict[int, list[dict]]):
        """Pipeline yang pembuat soalnya berbeda-beda menurut `count`.

        Begitulah perilaku sebenarnya: `count` masuk ke kunci cache evaluasi,
        sehingga jumlah yang diminta berbeda menghasilkan HIMPUNAN SOAL yang
        berbeda pula — bukan sekadar potongan pertama dari daftar yang sama.
        Tanpa meniru itu, uji ini tidak akan pernah menyentuh cacatnya.
        """
        pipe = RAGPipeline(
            summarizer=MagicMock(), chunker=MagicMock(), embedder=MagicMock(),
            store=MagicMock(), reranker=MagicMock(), generator=MagicMock(),
        )

        async def evaluation(course_id, plan, *, tenant_id, model=None):
            jumlah = plan.blueprint.counts["koding"]
            return [{**b, "type": "koding"} for b in hasil_per_count[jumlah]]

        pipe.evaluation = evaluation
        return pipe

    @pytest.mark.asyncio
    async def test_id_tidak_bergantung_pada_jumlah_yang_diminta(self) -> None:
        """Mendaftar `count=3` lalu `count=5`: soal yang sama harus satu id.

        Versi lama memberi nomor urut, jadi BUTIR_A yang muncul di posisi 0 pada
        daftar bertiga dan posisi 1 pada daftar berlima mendapat dua id berbeda.
        Mahasiswa yang mendaftar dengan satu nilai lalu mengirim kode dengan
        nilai lain menerima "latihan tidak ditemukan" untuk soal yang sedang
        terbuka di layarnya.
        """
        pipe = self._pipeline_dengan({
            3: [self.BUTIR_A, self.BUTIR_B, self.BUTIR_C],
            5: [self.BUTIR_C, self.BUTIR_A, self.BUTIR_B],  # posisi bergeser
        })
        bertiga = await pipe.livecode_exercises("dasprog", [2], tenant_id=A, count=3)
        berlima = await pipe.livecode_exercises("dasprog", [2], tenant_id=A, count=5)

        id_a = next(e.exercise_id for e in bertiga if "faktorial" in e.prompt)
        id_b = next(e.exercise_id for e in berlima if "faktorial" in e.prompt)
        assert id_a == id_b, (
            f"soal yang sama mendapat dua identitas: {id_a} vs {id_b} — "
            f"kiriman mahasiswa akan ditolak 'tidak ditemukan'"
        )

    @pytest.mark.asyncio
    async def test_definisinya_tersimpan_saat_didaftar(self) -> None:
        """Supaya pengiriman kode tidak perlu menyusun ulang daftarnya."""
        pipe = self._pipeline_dengan({3: [self.BUTIR_A, self.BUTIR_B]})
        latihan = await pipe.livecode_exercises("dasprog", [2], tenant_id=A, count=3)
        assert latihan
        for e in latihan:
            assert lc.load_exercise(e.exercise_id, tenant_id=A) == e

    def test_soal_berbeda_tetap_beda_id(self) -> None:
        a = lc.exercise_from_item(self.BUTIR_A, course_id="dasprog", week=2)
        b = lc.exercise_from_item(self.BUTIR_B, course_id="dasprog", week=2)
        assert a.exercise_id != b.exercise_id

    def test_judul_mirip_spesifikasi_beda_tetap_beda_id(self) -> None:
        """Slug dipangkas 40 aksara, jadi judul panjang yang mirip bisa sama.

        Kalau identitasnya hanya dari judul, kiriman mahasiswa akan dinilai
        dengan rubrik soal lain — lebih buruk daripada ditolak.
        """
        judul = "Buat sebuah fungsi yang membaca berkas lalu menghitung sesuatu"
        a = lc.exercise_from_item(
            {"question": judul, "required_function": "hitung_rata"},
            course_id="dasprog", week=2,
        )
        b = lc.exercise_from_item(
            {"question": judul, "required_function": "hitung_median"},
            course_id="dasprog", week=2,
        )
        assert a.exercise_id[:45] == b.exercise_id[:45], "prasyarat: slug-nya memang sama"
        assert a.exercise_id != b.exercise_id, "dua soal berbeda berbagi satu identitas"


class TestSimpananDefinisiLatihan:
    """Definisi diambil dari sisi server, tidak pernah dari klien.

    Kalau klien boleh mengirimkan rubrik dan kasus ujinya, mahasiswa cukup
    mengirim rubrik kosong untuk lulus.
    """

    def test_bolak_balik_utuh(self) -> None:
        lc.save_exercise(LAT, tenant_id=A)
        kembali = lc.load_exercise(LAT.exercise_id, tenant_id=A)
        assert kembali == LAT, "definisi berubah setelah disimpan dan dibaca lagi"

    def test_rubrik_dan_kasus_uji_ikut_tersimpan(self) -> None:
        """Justru bagian inilah yang tidak boleh hilang — itu kunci penilaiannya."""
        lc.save_exercise(LAT, tenant_id=A)
        kembali = lc.load_exercise(LAT.exercise_id, tenant_id=A)
        assert kembali is not None
        assert kembali.rubric == LAT.rubric
        assert kembali.test_cases == LAT.test_cases

    def test_isolasi_antar_tenant(self) -> None:
        """Latihan kampus A tidak boleh terbaca oleh kampus B.

        `exercise_id` adalah sidik isi soal, jadi dua kampus yang mengunggah
        materi serupa memang akan menghasilkan id yang sama — tabrakan di sini
        lumrah, bukan kebetulan langka.
        """
        lc.save_exercise(LAT, tenant_id=A)
        assert lc.load_exercise(LAT.exercise_id, tenant_id=B) is None

    def test_latihan_tak_dikenal_mengembalikan_none(self) -> None:
        assert lc.load_exercise("tidak-pernah-ada", tenant_id=A) is None

    def test_id_kosong_tidak_menyentuh_disk(self) -> None:
        assert lc.load_exercise("", tenant_id=A) is None

    def test_medan_baru_tidak_mematahkan_entri_lama(self) -> None:
        """Entri yang disimpan sebelum sebuah medan ada harus tetap terbaca.

        Menolaknya berarti kiriman mahasiswa gagal dinilai hanya karena bentuk
        simpanannya bertambah.
        """
        data = LAT.as_dict()
        data["medan_yang_belum_ada"] = 123
        pulih = lc.Exercise.from_dict(data)
        assert pulih == LAT

    def test_medan_hilang_tidak_mematahkan_entri_lama(self) -> None:
        data = LAT.as_dict()
        del data["rubric"]
        del data["forbidden_names"]
        pulih = lc.Exercise.from_dict(data)
        assert pulih.rubric == ()
        assert pulih.exercise_id == LAT.exercise_id


class TestBahasaLatihan:
    """Bahasa soal harus sampai ke penilainya.

    Ditemukan saat menjalankan portal sungguhan pada mata kuliah Sistem Basis
    Data: soalnya berbunyi "Buatlah query SQL...", lalu kiriman mahasiswa
    dibalas "Kode belum bisa dibaca Python: invalid syntax".

    Analisisnya sendiri sudah benar — kode berbahasa di luar SUPPORTED_LANGUAGES
    memang dilewatkan ke tinjauan LLM. Yang salah adalah `Exercise.language`
    tidak pernah diisi, sehingga nilainya selalu jatuh ke bawaan "python".

    Akibatnya bukan "penilaiannya kurang tepat": untuk SETIAP mata kuliah yang
    tidak mengajarkan Python, semua kiriman pasti gagal dengan pesan yang
    menuduh mahasiswa menulis Python yang salah.
    """

    SQL = "SELECT mhs_nama FROM mahasiswa WHERE mhs_nim = '123';"

    def test_sql_tidak_diurai_sebagai_python(self) -> None:
        latihan = lc.exercise_from_item(
            {"question": "Tulis query SELECT", "language": "sql"},
        )
        assert latihan.language == "sql"
        temuan, lanjut_llm = lc.analyze_code(self.SQL, latihan)
        assert lanjut_llm is True, (
            "kode SQL dihentikan di analisis statis Python — "
            "seluruh kiriman mata kuliah basis data akan gagal"
        )
        assert not any("Python" in t.message for t in temuan)

    def test_python_tetap_dianalisis_statis(self) -> None:
        """Pelonggaran di atas tidak boleh ikut mematikan pemeriksaan Python."""
        latihan = lc.exercise_from_item(
            {"question": "Tulis fungsi", "language": "python"},
        )
        temuan, lanjut_llm = lc.analyze_code("def f(\n", latihan)
        assert lanjut_llm is False
        assert any("Python" in t.message for t in temuan)

    def test_tanpa_bahasa_tetap_python(self) -> None:
        """Bawaan tidak berubah: model lama tidak menyebut bahasa sama sekali.

        Menebak bahasa lain saat tidak disebut akan mematikan pemeriksaan
        sintaks untuk mata kuliah yang justru memakai Python.
        """
        assert lc.exercise_from_item({"question": "x"}).language == "python"

    @pytest.mark.parametrize(
        "ditulis, jadi",
        [("PY", "python"), ("python3", "python"), ("MySQL", "sql"),
         ("postgres", "sql"), ("JS", "javascript"), ("c++", "cpp")],
    )
    def test_nama_lain_dibakukan(self, ditulis: str, jadi: str) -> None:
        """Model menyebut bahasa dengan nama yang berbeda-beda tiap kali."""
        latihan = lc.exercise_from_item({"question": "x", "language": ditulis})
        assert latihan.language == jadi

    def test_bahasa_asing_diterima_apa_adanya(self) -> None:
        """Nama di luar daftar tidak dipaksa kembali ke "python".

        Yang menentukan hanyalah "bukan python" — memaksanya kembali ke bawaan
        justru mengembalikan persis cacat yang sedang diperbaiki.
        """
        latihan = lc.exercise_from_item({"question": "x", "language": "kotlin"})
        assert latihan.language == "kotlin"
        _, lanjut_llm = lc.analyze_code("fun main() {}", latihan)
        assert lanjut_llm is True

    def test_bahasa_ikut_tersimpan(self) -> None:
        """Definisi dibaca kembali saat kiriman dinilai — bahasanya harus ikut."""
        latihan = lc.exercise_from_item({"question": "x", "language": "sql"})
        lc.save_exercise(latihan, tenant_id=A)
        assert lc.load_exercise(latihan.exercise_id, tenant_id=A).language == "sql"
