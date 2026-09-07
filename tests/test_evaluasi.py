"""Evaluasi berkala: kuis, ETS, EAS atas rentang minggu, dengan soal bercampur jenis.

Dua hal yang paling penting dijaga di sini:

1. Soal objektif TIDAK boleh dinilai LLM — jawabannya pasti, dan menyerahkannya
   ke model menukar kepastian dengan biaya serta kemungkinan salah nilai.
2. Butir yang gagal atau ragu dinilai ditandai perlu tinjauan, BUKAN diberi nol.
   Kegagalan mesin bukan kesalahan mahasiswa.
"""
from __future__ import annotations

import pytest

from src import evaluation as ev
from src.generation.prompts import build_evaluation_prompt, parse_evaluation_json


class TestJadwal:
    def test_minggu_evaluasi_sesuai_rencana(self) -> None:
        assert ev.EVALUATION_WEEKS == (4, 8, 12, 16)

    @pytest.mark.parametrize("week", [4, 8, 12, 16])
    def test_minggu_evaluasi_punya_rencana(self, week: int) -> None:
        assert ev.is_evaluation_week(week)
        assert ev.plan_for_week(week) is not None

    @pytest.mark.parametrize("week", [1, 3, 5, 7, 9, 15])
    def test_minggu_biasa_tidak_punya_rencana(self, week: int) -> None:
        assert not ev.is_evaluation_week(week)
        assert ev.plan_for_week(week) is None

    def test_ets_kumulatif_kuis_tidak(self) -> None:
        # Inilah beda ETS dari kuis: ia menguji separuh semester, bukan satu blok.
        kuis = ev.plan_for_week(4)
        ets = ev.plan_for_week(8)
        assert kuis.weeks_covered == (1, 2, 3, 4)
        assert ets.weeks_covered == tuple(range(1, 9))
        assert ets.kind == "ets"

    def test_ets_dan_eas_memuat_soal_produksi(self) -> None:
        # Pilihan ganda hanya mengukur pengenalan; esai & koding mengukur
        # apakah mahasiswa bisa memproduksi jawabannya sendiri.
        for w in (8, 16):
            jenis = ev.plan_for_week(w).blueprint.types
            assert "esai" in jenis
            assert "koding" in jenis

    def test_teks_rentang_terbaca_manusia(self) -> None:
        assert ev.plan_for_week(4).range_text == "minggu 1-4"
        assert ev.custom_plan([5]).range_text == "minggu 5"


class TestKomposisiSoal:
    def test_jenis_asing_ditolak(self) -> None:
        with pytest.raises(ValueError, match="tidak dikenal"):
            ev.Blueprint({"menggambar": 3}).validate()

    def test_kosong_ditolak(self) -> None:
        with pytest.raises(ValueError, match="setidaknya satu soal"):
            ev.Blueprint({"esai": 0}).validate()

    def test_terlalu_banyak_ditolak(self) -> None:
        # Melampaui ini, keluaran model mulai terpotong di tengah soal.
        with pytest.raises(ValueError, match="maksimal"):
            ev.Blueprint({"pilihan_ganda": 50}).validate()

    def test_rentang_bebas_boleh_komposisi_sendiri(self) -> None:
        p = ev.custom_plan([5, 6, 7], "kuis", {"esai": 2, "koding": 1})
        assert p.weeks_covered == (5, 6, 7)
        assert p.blueprint.total == 3
        assert "pilihan_ganda" not in p.blueprint.types

    def test_rentang_kosong_ditolak(self) -> None:
        with pytest.raises(ValueError, match="tidak boleh kosong"):
            ev.custom_plan([])


class TestPemisahanCaraPenilaian:
    @pytest.mark.parametrize("tipe", ["pilihan_ganda", "benar_salah"])
    def test_objektif_dinilai_mesin(self, tipe: str) -> None:
        assert not ev.needs_llm_grading({"type": tipe})

    @pytest.mark.parametrize("tipe", ["isian_singkat", "esai", "koding"])
    def test_terbuka_dinilai_llm(self, tipe: str) -> None:
        assert ev.needs_llm_grading({"type": tipe})

    def test_jawaban_benar_bernilai_penuh(self) -> None:
        butir = {"type": "pilihan_ganda", "question": "?", "answer_index": 2}
        g = ev.grade_objective(butir, 2, 0)
        assert g.is_correct and g.score == 1.0

    def test_jawaban_tidak_dikenali_dihitung_salah(self) -> None:
        # Berbeda dari kuis dalam chat yang menanyakan ulang: pada evaluasi
        # resmi, menanyakan ulang berarti memberi kesempatan menebak lagi.
        butir = {"type": "pilihan_ganda", "question": "?", "answer_index": 2}
        for jawaban in (None, "entah", 99):
            g = ev.grade_objective(butir, jawaban, 0)
            assert g.is_correct is False
            assert g.score == 0.0


class TestRingkasan:
    def _butir(self, tipe: str, skor: float, benar: bool | None) -> ev.GradedItem:
        return ev.GradedItem(
            index=0, type=tipe, question="?", student_answer=None,
            correct_answer=None, is_correct=benar, score=skor,
        )

    def test_kosong_aman(self) -> None:
        assert ev.summarize([])["total"] == 0

    def test_rincian_per_jenis_terpisah(self) -> None:
        # Nilai total yang sama bisa berarti dua hal berbeda: kuat di pilihan
        # ganda tetapi lemah di esai belum tentu paham materinya.
        hasil = ev.summarize([
            self._butir("pilihan_ganda", 1.0, True),
            self._butir("pilihan_ganda", 1.0, True),
            self._butir("esai", 0.0, False),
            self._butir("esai", 0.5, False),
        ])
        assert hasil["skor"] == 62.5
        assert hasil["per_jenis"]["pilihan_ganda"]["persen"] == 100.0
        assert hasil["per_jenis"]["esai"]["persen"] == 25.0

    def test_butir_ragu_ditandai_bukan_dinolkan(self) -> None:
        hasil = ev.summarize([
            self._butir("pilihan_ganda", 1.0, True),
            self._butir("esai", 0.6, None),          # LLM ragu
        ])
        assert hasil["perlu_tinjauan"] == 1
        assert hasil["benar"] == 1                    # yang ragu tidak dihitung benar
        assert hasil["skor"] == 80.0                  # tetapi skornya tetap dihitung


class TestPromptDanParser:
    def test_prompt_menyebut_setiap_jenis_yang_diminta(self) -> None:
        p = ev.plan_for_week(8)
        teks = build_evaluation_prompt(p.label, p.range_text, p.blueprint.counts)
        for kata in ("PILIHAN GANDA", "BENAR/SALAH", "ISIAN SINGKAT", "ESAI", "KODING"):
            assert kata in teks
        assert "menghubungkan" in teks.lower()   # soal lintas minggu diminta eksplisit

    def test_parser_menerima_semua_jenis(self) -> None:
        raw = """[
          {"type":"pilihan_ganda","question":"a","options":["1","2","3","4"],"answer_index":0},
          {"type":"benar_salah","question":"b","answer_index":1},
          {"type":"isian_singkat","question":"c","expected_answer":"jwb"},
          {"type":"esai","question":"d","rubric":["r1"]},
          {"type":"koding","question":"e","expected_behavior":"jalan"}
        ]"""
        hasil = parse_evaluation_json(raw)
        assert [h["type"] for h in hasil] == [
            "pilihan_ganda", "benar_salah", "isian_singkat", "esai", "koding",
        ]

    def test_benar_salah_dapat_opsi_bawaan(self) -> None:
        # Model kerap menghilangkan options untuk benar/salah.
        hasil = parse_evaluation_json(
            '[{"type":"benar_salah","question":"x","answer_index":0}]'
        )
        assert hasil[0]["options"] == ["Benar", "Salah"]

    def test_butir_cacat_dibuang_bukan_ditambal(self) -> None:
        # Soal evaluasi menentukan nilai; menambal butir cacat berarti menilai
        # dengan soal yang tidak pernah benar-benar tersusun.
        raw = """[
          {"type":"pilihan_ganda","question":"kurang opsi","options":["1","2"],"answer_index":0},
          {"type":"isian_singkat","question":"tanpa kunci"},
          {"type":"esai","question":""},
          {"type":"pilihan_ganda","question":"sah","options":["1","2","3","4"],"answer_index":3}
        ]"""
        hasil = parse_evaluation_json(raw)
        assert len(hasil) == 1
        assert hasil[0]["question"] == "sah"

    def test_keluaran_bukan_json_menghasilkan_kosong(self) -> None:
        assert parse_evaluation_json("maaf, saya tidak bisa") == []
        assert parse_evaluation_json("") == []


class TestPenilaianTerpadu:
    """Penilaian evaluasi bercampur jenis lewat pipeline."""

    @staticmethod
    def _pipeline(grade_result):
        from unittest.mock import AsyncMock, MagicMock

        from src.pipeline import RAGPipeline

        gen = MagicMock()
        gen.grade_open_answer = AsyncMock(return_value=grade_result)
        return RAGPipeline(
            summarizer=MagicMock(), chunker=MagicMock(), embedder=MagicMock(),
            store=MagicMock(), reranker=MagicMock(), generator=gen,
        ), gen

    BUTIR = [
        {"type": "pilihan_ganda", "question": "pg", "options": list("abcd"),
         "answer_index": 1, "explanation": "karena b"},
        {"type": "benar_salah", "question": "bs", "options": ["Benar", "Salah"],
         "answer_index": 0},
        {"type": "isian_singkat", "question": "isi", "expected_answer": "jawabannya"},
    ]

    @pytest.mark.asyncio
    async def test_objektif_tidak_memanggil_llm(self) -> None:
        # Jawabannya pasti. Menyerahkannya ke model menukar kepastian dengan
        # biaya, kelambatan, dan kemungkinan salah nilai — tanpa keuntungan.
        pipe, gen = self._pipeline(
            {"skor": 1.0, "benar": True, "ragu": False, "feedback": "tepat"}
        )
        await pipe.grade_evaluation(
            self.BUTIR, [1, 0, "jawabannya"], tenant_id="kampus-a",
        )
        assert gen.grade_open_answer.await_count == 1   # hanya isian singkat

    @pytest.mark.asyncio
    async def test_skor_gabungan_dan_rincian_per_jenis(self) -> None:
        pipe, _ = self._pipeline(
            {"skor": 0.5, "benar": False, "ragu": False, "feedback": "kurang lengkap"}
        )
        hasil = await pipe.grade_evaluation(
            self.BUTIR, [1, 1, "separuh benar"], tenant_id="kampus-a",
        )
        assert hasil["total"] == 3
        assert hasil["benar"] == 1                       # hanya pilihan ganda
        assert hasil["per_jenis"]["benar_salah"]["persen"] == 0.0
        assert hasil["per_jenis"]["isian_singkat"]["skor"] == 0.5

    @pytest.mark.asyncio
    async def test_penilaian_gagal_ditandai_bukan_dinolkan(self) -> None:
        # Kegagalan mesin bukan kesalahan mahasiswa; memberi nol diam-diam
        # adalah kerugian yang tidak terlihat siapa pun.
        pipe, _ = self._pipeline(None)
        hasil = await pipe.grade_evaluation(
            self.BUTIR, [1, 0, "jawaban panjang"], tenant_id="kampus-a",
        )
        assert hasil["perlu_tinjauan"] == 1
        butir_isian = hasil["butir"][2]
        assert butir_isian["is_correct"] is None
        assert "tinjau" in butir_isian["feedback"].lower()

    @pytest.mark.asyncio
    async def test_llm_ragu_juga_ditandai(self) -> None:
        pipe, _ = self._pipeline(
            {"skor": 0.7, "benar": True, "ragu": True, "feedback": "ambigu"}
        )
        hasil = await pipe.grade_evaluation(
            self.BUTIR, [1, 0, "jawaban"], tenant_id="kampus-a",
        )
        assert hasil["perlu_tinjauan"] == 1
        assert hasil["butir"][2]["is_correct"] is None
        assert hasil["butir"][2]["score"] == 0.7        # skornya tetap dipakai

    @pytest.mark.asyncio
    async def test_jawaban_kurang_tidak_menggagalkan(self) -> None:
        # Mahasiswa yang mengirim jawaban lebih sedikit dari jumlah soal tetap
        # dinilai; sisanya dianggap tidak dijawab.
        pipe, _ = self._pipeline(
            {"skor": 0.0, "benar": False, "ragu": False, "feedback": "kosong"}
        )
        hasil = await pipe.grade_evaluation(self.BUTIR, [1], tenant_id="kampus-a")
        assert hasil["total"] == 3

    @pytest.mark.asyncio
    async def test_tanpa_tenant_ditolak(self) -> None:
        from src.tenancy import TenantScopeError

        pipe, _ = self._pipeline(None)
        with pytest.raises(TenantScopeError):
            await pipe.grade_evaluation(self.BUTIR, [1, 0, "x"], tenant_id="")


class TestKonsistensiBenarDanSkor:
    """`benar` harus sejalan dengan skornya.

    Ditemukan saat uji ujung-ke-ujung: model mengembalikan `benar: true`
    bersama `skor: 0.5` — bermaksud "arahnya sudah betul". Dipercaya apa adanya,
    hitungan "berapa soal yang benar" menggelembung, dan pada penelitian angka
    itulah yang dilaporkan.
    """

    @pytest.mark.parametrize(
        ("raw", "skor_harap", "benar_harap"),
        [
            ('{"skor":1.0,"benar":true}', 1.0, True),
            ('{"skor":0.5,"benar":true}', 0.5, False),   # inti perbaikannya
            ('{"skor":0.99,"benar":true}', 0.99, False),
            ('{"skor":0.0,"benar":false}', 0.0, False),
            ('{"skor":1.0}', 1.0, True),                 # tanpa field `benar`
        ],
    )
    def test_benar_menuntut_skor_penuh(
        self, raw: str, skor_harap: float, benar_harap: bool,
    ) -> None:
        from src.generation.prompts import parse_grade_json

        hasil = parse_grade_json(raw)
        assert hasil is not None
        assert hasil["skor"] == pytest.approx(skor_harap)
        assert hasil["benar"] is benar_harap

    def test_skor_parsial_tetap_tersimpan(self) -> None:
        # Nilai parsialnya tidak boleh ikut hilang — hanya label "benar" yang
        # dikoreksi, bukan penilaiannya.
        from src.generation.prompts import parse_grade_json

        assert parse_grade_json('{"skor":0.5,"benar":true}')["skor"] == 0.5
