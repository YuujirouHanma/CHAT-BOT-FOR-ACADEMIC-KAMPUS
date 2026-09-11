"""Regresi jalur penilaian evaluasi: jawaban benar/salah dan butir koding.

Dua cacat yang dijaga di sini pernah membuat nilai mahasiswa salah tanpa
terlihat — keduanya menghasilkan angka yang tampak wajar, jadi hanya uji yang
bisa menahannya kembali:

1. `bool` adalah subkelas `int` di Python, sehingga jawaban benar/salah yang
   datang sebagai `true` sempat dinilai sebagai "memilih opsi nomor 1".
2. Butir `koding` sempat dinilai pemeriksa PROSA lewat /evaluation/submit,
   padahal kode yang sama dinilai pemeriksa KODE lewat /livecode/submit —
   satu kiriman, dua standar penilaian.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src import evaluation as ev
from src.pipeline import RAGPipeline

# Urutan opsi benar/salah yang dipakai parser: 0 = Benar, 1 = Salah.
BS_BENAR = {"type": "benar_salah", "question": "bs", "answer_index": 0}
BS_SALAH = {"type": "benar_salah", "question": "bs", "answer_index": 1}
PG = {"type": "pilihan_ganda", "question": "pg", "answer_index": 1}


class TestJawabanBenarSalah:
    """`true`/`false` adalah pernyataan, bukan nomor opsi."""

    @pytest.mark.parametrize(
        ("jawaban", "butir"),
        [(True, BS_BENAR), (False, BS_SALAH), (0, BS_BENAR), (1, BS_SALAH),
         ("Benar", BS_BENAR), ("salah", BS_SALAH), ("true", BS_BENAR)],
    )
    def test_bentuk_jawaban_sah_dinilai_benar(self, jawaban, butir: dict) -> None:
        g = ev.grade_objective(butir, jawaban, 0)
        assert g.is_correct is True
        assert g.score == 1.0

    @pytest.mark.parametrize(
        ("jawaban", "butir"),
        [(True, BS_SALAH), (False, BS_BENAR), ("salah", BS_BENAR)],
    )
    def test_jawaban_keliru_tetap_salah(self, jawaban, butir: dict) -> None:
        assert ev.grade_objective(butir, jawaban, 0).is_correct is False

    def test_true_bukan_opsi_nomor_satu(self) -> None:
        # Inti cacatnya: `isinstance(True, int)` bernilai True, jadi `true`
        # sempat lolos sebagai indeks 1 dan cocok dengan answer_index 1.
        g = ev.grade_objective(BS_SALAH, True, 0)
        assert g.student_answer == 0        # "Benar", bukan indeks 1
        assert g.is_correct is False

    def test_boolean_bukan_jawaban_pilihan_ganda(self) -> None:
        # Pada pilihan ganda tidak ada tafsiran "benar/salah" sama sekali;
        # boolean di situ jawaban yang tidak dikenali, karena itu salah.
        for jawaban in (True, False):
            g = ev.grade_objective(PG, jawaban, 0)
            assert g.student_answer is None
            assert g.is_correct is False

    def test_indeks_di_luar_dua_opsi_tidak_sah(self) -> None:
        # Soal benar/salah hanya punya dua opsi; indeks 2 pasti bukan pilihan.
        assert ev.grade_objective(BS_BENAR, 2, 0).is_correct is False

    def test_jawaban_kosong_dihitung_salah(self) -> None:
        for jawaban in (None, "", "mungkin", 3.0):
            assert ev.grade_objective(BS_BENAR, jawaban, 0).is_correct is False


KODE_BENAR = (
    "def faktorial(n):\n"
    "    hasil = 1\n"
    "    for i in range(2, n + 1):\n"
    "        hasil *= i\n"
    "    return hasil\n"
)

# Butir koding apa adanya dari cache: TANPA test_cases dan required_function.
# Ketiadaan spesifikasi itu keadaan yang sah, bukan galat.
BUTIR_KODING = {
    "type": "koding",
    "question": "Tulis fungsi faktorial.",
    "expected_behavior": "mengembalikan n!",
    "rubric": ["benar secara logika"],
    "explanation": "pakai perulangan",
}

TINJAUAN_LULUS = {
    "lulus": True, "skor": 1.0, "ragu": False, "ringkasan": "Sudah benar.",
    "benar": ["perulangan tepat"], "keliru": [], "petunjuk": [],
}


def _pipeline(tinjauan) -> tuple[RAGPipeline, MagicMock]:
    gen = MagicMock()
    gen.review_code = AsyncMock(return_value=tinjauan)
    gen.grade_open_answer = AsyncMock(
        return_value={"skor": 1.0, "benar": True, "ragu": False, "feedback": "ok"},
    )
    return RAGPipeline(
        summarizer=MagicMock(), chunker=MagicMock(), embedder=MagicMock(),
        store=MagicMock(), reranker=MagicMock(), generator=gen,
    ), gen


class TestButirKodingDinilaiSebagaiKode:
    """Kode yang sama harus satu standar, dari endpoint mana pun dikirim."""

    @pytest.mark.asyncio
    async def test_koding_lewat_peninjau_kode_bukan_prosa(self) -> None:
        pipe, gen = _pipeline(TINJAUAN_LULUS)
        hasil = await pipe.grade_evaluation(
            [BUTIR_KODING], [KODE_BENAR], tenant_id="kampus-a",
        )
        gen.review_code.assert_awaited_once()
        gen.grade_open_answer.assert_not_awaited()
        assert hasil["butir"][0]["is_correct"] is True
        assert hasil["butir"][0]["score"] == 1.0
        assert hasil["per_jenis"]["koding"]["persen"] == 100.0

    @pytest.mark.asyncio
    async def test_soal_tanpa_spesifikasi_tetap_dinilai(self) -> None:
        # Butir di cache tidak punya test_cases/required_function; jalur kode
        # harus tetap berjalan, bukan menolak menilai.
        pipe, gen = _pipeline(TINJAUAN_LULUS)
        await pipe.grade_evaluation([BUTIR_KODING], [KODE_BENAR], tenant_id="kampus-a")
        panggilan = gen.review_code.await_args.kwargs
        assert panggilan["test_cases"] == []
        assert panggilan["code"] == KODE_BENAR

    @pytest.mark.asyncio
    async def test_galat_sintaks_tidak_memanggil_llm(self) -> None:
        # Persis seperti /livecode/submit: galat statis sudah pasti tanpa
        # penalaran, memanggil LLM untuk itu hanya membuang biaya.
        pipe, gen = _pipeline(TINJAUAN_LULUS)
        hasil = await pipe.grade_evaluation(
            [BUTIR_KODING], ["def faktorial(n)\n  return"], tenant_id="kampus-a",
        )
        gen.review_code.assert_not_awaited()
        assert hasil["butir"][0]["score"] == 0.0
        assert hasil["butir"][0]["is_correct"] is False

    @pytest.mark.asyncio
    async def test_tanpa_jawaban_dinilai_nol_tanpa_llm(self) -> None:
        pipe, gen = _pipeline(TINJAUAN_LULUS)
        hasil = await pipe.grade_evaluation([BUTIR_KODING], [None], tenant_id="kampus-a")
        gen.review_code.assert_not_awaited()
        assert hasil["butir"][0]["score"] == 0.0
        assert hasil["perlu_tinjauan"] == 0     # tidak menjawab bukan keraguan

    @pytest.mark.asyncio
    async def test_tinjauan_gagal_ditandai_bukan_dinolkan(self) -> None:
        # Kegagalan mesin bukan kesalahan mahasiswa.
        pipe, _ = _pipeline(None)
        hasil = await pipe.grade_evaluation(
            [BUTIR_KODING], [KODE_BENAR], tenant_id="kampus-a",
        )
        assert hasil["butir"][0]["is_correct"] is None
        assert hasil["perlu_tinjauan"] == 1

    @pytest.mark.asyncio
    async def test_umpan_balik_memuat_koreksi_jalur_kode(self) -> None:
        # Butir evaluasi hanya punya satu ruang umpan balik; catatan keliru dan
        # petunjuk tidak boleh hilang di jalur ini.
        pipe, _ = _pipeline({
            "lulus": False, "skor": 0.5, "ragu": False, "ringkasan": "hampir",
            "benar": [], "petunjuk": ["periksa batas atas range"],
            "keliru": [{"baris": 3, "masalah": "batas range", "akibat": "kurang 1"}],
        })
        hasil = await pipe.grade_evaluation(
            [BUTIR_KODING], [KODE_BENAR], tenant_id="kampus-a",
        )
        umpan = hasil["butir"][0]["feedback"]
        assert "hampir" in umpan
        assert "batas range" in umpan
        assert "periksa batas atas range" in umpan

    @pytest.mark.asyncio
    async def test_jenis_lain_tetap_lewat_pemeriksa_prosa(self) -> None:
        # Perbaikannya khusus koding: esai dan isian tetap dinilai sebagai prosa.
        pipe, gen = _pipeline(TINJAUAN_LULUS)
        await pipe.grade_evaluation(
            [{"type": "esai", "question": "e", "rubric": ["r"]}],
            ["uraian jawaban"],
            tenant_id="kampus-a",
        )
        gen.grade_open_answer.assert_awaited_once()
        gen.review_code.assert_not_awaited()


class TestBatasSkemaHTTP:
    """Perbaikan `grade_objective` percuma bila skema sudah merusak jawabannya.

    Ditambahkan saat peninjauan akhir: penilaian benar/salah di
    `TestJawabanBenarSalah` sudah benar untuk pemanggilan langsung, tetapi lewat
    HTTP jawaban melewati pydantic lebih dulu. Selama union `answers` menaruh
    `int` di depan tanpa `bool`, `true` dipaksa menjadi 1 di batas skema — dan 1
    berarti "Salah". Nilainya terbalik tanpa satu pun galat, jadi urutan union
    itu perlu dikunci uji, bukan hanya komentar.
    """

    def test_boolean_tidak_dipaksa_menjadi_indeks(self) -> None:
        from src.api.schemas import EvaluationSubmitRequest

        req = EvaluationSubmitRequest.model_validate(
            {"course_id": "sbd", "week": 3, "answers": [True, False, 1, "benar"]},
        )
        assert req.answers[0] is True
        assert req.answers[1] is False
        # Bentuk jawaban lain tidak ikut berubah tafsirannya.
        assert req.answers[2] == 1 and not isinstance(req.answers[2], bool)
        assert req.answers[3] == "benar"

    def test_boolean_dari_skema_dinilai_utuh_sampai_penilaian(self) -> None:
        from src.api.schemas import EvaluationSubmitRequest

        req = EvaluationSubmitRequest.model_validate(
            {"course_id": "sbd", "week": 3, "answers": [True]},
        )
        # `true` pada butir berkunci "Benar" harus benar, bukan "memilih opsi 1".
        assert ev.grade_objective(BS_BENAR, req.answers[0], 0).is_correct is True
        assert ev.grade_objective(BS_SALAH, req.answers[0], 0).is_correct is False
