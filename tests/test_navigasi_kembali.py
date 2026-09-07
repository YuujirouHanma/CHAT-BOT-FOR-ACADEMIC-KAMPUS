"""Tombol kembali pada alur terpandu.

Mahasiswa yang salah memilih minggu ingin mengganti minggunya saja. Sebelum ini
satu-satunya jalan keluar adalah "ulangi dari awal", yang membuang seluruh
konteks — satu salah klik jadi terasa seperti hukuman.
"""
from __future__ import annotations

import pytest

from src import guided
from src.api.session import Session

TENANT = "kampus-uji"


def _sesi_lengkap() -> Session:
    s = Session(session_id="s1", tenant_id=TENANT)
    s.course_id = "sbd"
    s.course_name = "Sistem Basis Data"
    s.weeks = [2]
    s.topic = "Perulangan dan percabangan"
    s.content_id = "sbd-minggu-2"
    s.source_filter = "bab2.pdf"
    s.style = "visual"
    return s


class TestPengenalanMaksud:
    @pytest.mark.parametrize(
        "teks",
        ["kembali", "balik", "back", "sebelumnya", "salah pilih",
         "salah klik", "bukan itu", "Kembali — pilih materi lain"],
    )
    def test_dikenali(self, teks: str) -> None:
        assert guided.wants_back(teks)

    @pytest.mark.parametrize(
        "teks",
        ["apa itu SQL", "jelaskan perulangan", "minggu 3", "kuis"],
    )
    def test_pertanyaan_biasa_tidak_dianggap_kembali(self, teks: str) -> None:
        assert not guided.wants_back(teks)


class TestPemetaanLangkah:
    @pytest.mark.parametrize(
        ("dari", "ke"),
        [("week", "course"), ("material", "week"),
         ("style", "material"), ("question", "style")],
    )
    def test_langkah_sebelumnya(self, dari: str, ke: str) -> None:
        assert guided.previous_step(dari) == ke

    def test_langkah_pertama_tidak_punya_pendahulu(self) -> None:
        assert guided.previous_step("course") is None
        assert guided.back_choice("course") is None

    def test_tombol_menyebut_tujuan_bukan_sekadar_kembali(self) -> None:
        # Mahasiswa harus tahu akan dibawa ke mana sebelum menekannya.
        tombol = guided.back_choice("style")
        assert tombol is not None
        assert tombol.kind == "back"
        assert tombol.value == "material"        # nilai = langkah TUJUAN
        assert "materi" in tombol.label.lower()


class TestMundurSatuLangkah:
    def test_dari_gaya_hanya_membuang_materi_dan_gaya(self) -> None:
        s = _sesi_lengkap()
        s.step_back_to("material")
        assert s.course_id == "sbd"              # dipertahankan
        assert s.weeks == [2]                    # dipertahankan
        assert s.content_id is None              # dibuang
        assert s.source_filter is None
        assert s.style is None
        assert s.awaiting == "material"

    def test_dari_materi_membuang_minggu_ke_bawah(self) -> None:
        s = _sesi_lengkap()
        s.step_back_to("week")
        assert s.course_id == "sbd"
        assert s.weeks == []
        assert s.topic is None
        assert s.content_id is None

    def test_kembali_ke_awal_membuang_semua_pilihan(self) -> None:
        s = _sesi_lengkap()
        s.step_back_to("course")
        assert (s.course_id, s.course_name) == (None, None)
        assert s.weeks == []
        assert (s.content_id, s.source_filter, s.style) == (None, None, None)

    def test_riwayat_percakapan_tidak_ikut_terbuang(self) -> None:
        # Yang dibatalkan adalah pilihan navigasi, bukan apa yang sudah dipelajari.
        s = _sesi_lengkap()
        s.add_turn("user", "apa itu perulangan")
        s.add_turn("assistant", "Perulangan adalah…")
        s.record("user", "apa itu perulangan")
        s.step_back_to("course")
        assert len(s.history) == 2
        assert len(s.transcript) == 1

    def test_kuis_berjalan_dihentikan(self) -> None:
        # Soalnya melekat pada materi yang barusan ditinggalkan.
        s = _sesi_lengkap()
        s.start_quiz([{"question": "?", "options": ["a", "b"], "answer_index": 0}])
        s.step_back_to("material")
        assert s.quiz is None


class TestLewatRuteSungguhan:
    """Tombol kembali dilalui lewat POST /chat/ask, bukan hanya di session.

    Ditambahkan setelah uji ujung-ke-ujung menemukan jalur ini melempar 500:
    pemanggilan `guided_turn` di dalamnya memakai nama argumen yang salah.
    Seluruh tes sebelumnya menguji `step_back_to()` dan `wants_back()` secara
    terpisah, jadi tidak satu pun melewati rutenya — dan cacatnya lolos.
    """

    @pytest.fixture
    def klien(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api.dependencies import get_pipeline
        from src.api.main import app
        from src.config import settings
        from src.pipeline import RAGPipeline

        monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)

        class StorePalsu:
            async def list_courses(self, *, tenant_id):
                return [{"course_id": "dasprog-rka",
                         "course_name": "Dasar Pemrograman (RKA)"}]

            async def list_weeks(self, course_id, *, tenant_id):
                return [2]

            async def list_materials_for_weeks(self, course_id, weeks, *, tenant_id):
                return [{"source_file": "bab2.pdf",
                         "content_id": "dasprog-rka-minggu-2", "week": 2}]

            async def get_week_text(self, *a, **kw):
                return "Perulangan dan percabangan."

            async def get_material_text(self, *a, **kw):
                return "Perulangan dan percabangan."

        class GeneratorPalsu:
            async def summarize_week_topic(self, *a, **kw):
                return "Perulangan dan percabangan"

            async def generate_starter_questions(self, *a, **kw):
                return ["Apa itu perulangan?"]

        class PipelinePalsu(RAGPipeline):
            def __init__(self):
                self._store = StorePalsu()
                self._generator = GeneratorPalsu()

        app.dependency_overrides[get_pipeline] = lambda: PipelinePalsu()
        yield TestClient(app)
        app.dependency_overrides.clear()

    @staticmethod
    def _ask(c, pesan, sid=None):
        body = {"question": pesan, "student_id": "2021001"}
        if sid:
            body["session_id"] = sid
        r = c.post("/chat/ask", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_maju_lalu_mundur_lewat_tombol(self, klien) -> None:
        sid = self._ask(klien, "halo")["session_id"]

        d = self._ask(klien, "Dasar Pemrograman (RKA)", sid)
        assert d["step"] == "week"
        # Nama mata kuliah TIDAK boleh menetapkan gaya belajar.
        assert d["context"]["style"] is None

        d = self._ask(klien, "Minggu 2", sid)
        assert d["step"] == "material"

        d = self._ask(klien, "bab2.pdf", sid)
        assert d["step"] == "style"          # langkah gaya memang ditanyakan
        tombol = [c for c in d["choices"] if c["kind"] == "back"]
        assert tombol and tombol[0]["value"] == "material"

        # Mundur: gaya -> materi
        d = self._ask(klien, tombol[0]["label"], sid)
        assert d["step"] == "material"
        assert d["context"]["course_id"] == "dasprog-rka"   # dipertahankan
        assert d["context"]["weeks"] == [2]                 # dipertahankan
        assert d["context"]["source_file"] is None          # dibuang

    def test_mundur_dari_teks_bebas(self, klien) -> None:
        sid = self._ask(klien, "halo")["session_id"]
        self._ask(klien, "Dasar Pemrograman (RKA)", sid)
        d = self._ask(klien, "Minggu 2", sid)
        assert d["step"] == "material"

        d = self._ask(klien, "salah pilih", sid)
        assert d["step"] == "week"
        assert d["context"]["weeks"] == []


class TestPemilihanGayaLewatRute:
    """Klik tombol gaya harus melanjutkan alur, bukan memicu jawaban.

    Uji ini melewati POST /chat/ask sungguhan. Versi unit-nya lolos karena
    `is_choice_label` benar sendirian — yang keliru adalah URUTAN pemeriksaan
    di dalam `guided_turn`, dan itu hanya terlihat dari rutenya.
    """

    @pytest.fixture
    def klien(self, monkeypatch):
        from fastapi.testclient import TestClient

        from src.api.dependencies import get_pipeline
        from src.api.main import app
        from src.config import settings
        from src.pipeline import RAGPipeline

        monkeypatch.setattr(settings, "rate_limit_enabled", False, raising=False)

        class StorePalsu:
            async def list_courses(self, *, tenant_id):
                return [{"course_id": "dasprog-rka",
                         "course_name": "Dasar Pemrograman (RKA)"}]

            async def list_weeks(self, course_id, *, tenant_id):
                return [2]

            async def list_materials_for_weeks(self, course_id, weeks, *, tenant_id):
                return [{"source_file": "bab2.pdf",
                         "content_id": "dasprog-rka-minggu-2", "week": 2}]

            async def get_week_text(self, *a, **kw):
                return "Perulangan dan percabangan."

            async def get_material_text(self, *a, **kw):
                return "Perulangan dan percabangan."

        class GeneratorPalsu:
            async def summarize_week_topic(self, *a, **kw):
                return "Perulangan dan percabangan"

            async def generate_starter_questions(self, *a, **kw):
                return ["Apa itu perulangan?"]

        class PipelinePalsu(RAGPipeline):
            def __init__(self):
                self._store = StorePalsu()
                self._generator = GeneratorPalsu()

            async def query(self, *a, **kw):        # noqa: D102
                raise AssertionError(
                    "query() terpanggil — label tombol gaya salah dibaca "
                    "sebagai pertanyaan dan dijawab sungguhan."
                )

        app.dependency_overrides[get_pipeline] = lambda: PipelinePalsu()
        yield TestClient(app)
        app.dependency_overrides.clear()

    @staticmethod
    def _ask(c, pesan, sid=None):
        body = {"question": pesan, "student_id": "2021001"}
        if sid:
            body["session_id"] = sid
        r = c.post("/chat/ask", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    @pytest.mark.parametrize(
        "gaya", ["naratif", "visual", "praktik", "ringkas", "sokratik"],
    )
    def test_klik_tombol_gaya_melanjutkan_alur(self, klien, gaya: str) -> None:
        from src import learning_styles

        sid = self._ask(klien, "halo")["session_id"]
        self._ask(klien, "Dasar Pemrograman (RKA)", sid)
        self._ask(klien, "Minggu 2", sid)
        d = self._ask(klien, "bab2.pdf", sid)
        assert d["step"] == "style"

        label = learning_styles.choice_label(learning_styles.resolve(gaya))
        d = self._ask(klien, label, sid)

        # `query()` di pipeline palsu melempar bila terpanggil — sampai di sini
        # berarti label memang diperlakukan sebagai pilihan.
        assert d["mode"] == "choices"
        assert d["step"] == "question"
        assert d["context"]["style"] == gaya


class TestBentukKiriman:
    """Tombol kembali harus menerima label MAUPUN value.

    Ditemukan saat menyusun dokumen untuk tim BE: klien yang mengirim `value`
    ("material") — pola yang memang dipakai untuk jenis pilihan lain — menekan
    tombol tanpa terjadi apa pun. Kegagalan senyap: tidak ada galat, tidak ada
    perubahan, jadi tidak ada yang menyadarinya sampai ada yang mengeluh.
    """

    @pytest.mark.parametrize(
        "kiriman",
        ["Kembali — pilih materi lain", "material",
         "Kembali — pilih minggu lain", "week", "course", "style"],
    )
    def test_kedua_bentuk_dikenali(self, kiriman: str) -> None:
        assert guided.wants_back(kiriman)

    @pytest.mark.parametrize(
        "kiriman",
        ["apa itu perulangan", "materi kuliah apa saja",
         "jelaskan minggu 2", "Dasar Pemrograman (RKA)"],
    )
    def test_pesan_biasa_tidak_dianggap_kembali(self, kiriman: str) -> None:
        assert not guided.wants_back(kiriman)

    def test_target_sah_hanya_langkah_yang_ada(self) -> None:
        assert guided.BACK_TARGETS == {"course", "week", "material", "style"}
