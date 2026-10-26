"""Saran pertanyaan pada jalur PENOLAKAN, dan gerbang relevansi di mode terpandu.

Dua cacat yang diuji di sini dilaporkan dari ujicoba sebagai satu gejala:

    "apa resep nasi goreng?" — jawabannya sudah benar, materi tidak mencakupnya,
    tetapi rekomendasi yang menyertainya malah berhubungan dengan nasi goreng,
    padahal di materi tidak ada.

Ternyata itu DUA cacat yang berbeda:

1. Gerbang relevansi hanya terpasang di `query()`. Jalur terpandu — yang justru
   dipakai mode belajar — hanya menolak ketika pencarian mengembalikan NOL
   hasil, dan dengan penyaring kampus yang selalu aktif itu hampir tak pernah
   terjadi. Jadi di mode terpandu pertanyaan di luar materi tetap DIJAWAB.

2. Penolakan mengosongkan sarannya. Itu menghilangkan gejalanya, tetapi
   mahasiswa yang pertanyaannya ditolak justru paling butuh arah: ia belum tahu
   apa yang ada di materi. Yang benar bukan nol saran, melainkan saran yang
   mengacu ke materi.

Satu hal lagi yang dijaga uji di bawah dan mudah hilang saat seseorang
"menyederhanakan": jalur penolakan TIDAK BOLEH memanggil model. Gerbang
relevansi ada justru untuk menghentikan pembakaran panggilan model pada
pertanyaan yang tidak terjawab materi.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src import gen_cache
from src.config import settings
from src.pipeline import RAGPipeline, _bebas_gerbang
from tests.conftest import TEST_TENANT_ID as TENANT

RENDAH = 0.001   # di luar materi — terukur 0,000-0,001
TINGGI = 0.95    # di dalam materi — terukur 0,937-0,960


class RetrieverSkor:
    """Retriever yang skornya sama untuk semua query, dan mencatat penyaringnya."""

    def __init__(self, skor: float) -> None:
        self.skor = skor
        self.panggilan: list[dict] = []

    async def retrieve(self, query: str, **kw) -> list[dict]:
        self.panggilan.append({"query": query, **kw})
        return [{
            "chunk_id": "c1",
            "rerank_score": self.skor,
            "payload": {"text": "isi", "source_file": "bab1.pdf", "page_number": 1},
        }]


def _pipa(skor: float) -> tuple[RAGPipeline, AsyncMock, RetrieverSkor]:
    """Pipeline tanpa memuat model; hanya store, generator, dan retriever."""
    p = RAGPipeline.__new__(RAGPipeline)

    store = AsyncMock()
    store.get_material_text = AsyncMock(return_value="isi materi contoh")
    store.list_indexed_files = AsyncMock(
        return_value=[{"source_file": "bab1.pdf", "content_id": "kelas-1"}]
    )
    store.list_weeks = AsyncMock(return_value=[1, 2])
    store.list_materials_for_weeks = AsyncMock(
        return_value=[{"source_file": "bab1.pdf", "content_id": "kelas-1", "week": 1}]
    )

    gen = AsyncMock()
    gen.decompose_query = AsyncMock(return_value={"query_diperkaya": "x"})
    gen.generate = AsyncMock(return_value="Jawaban.")
    gen.generate_followup = AsyncMock(return_value=["saran tentang nasi goreng"])
    gen.generate_starter_questions = AsyncMock(
        return_value=["JANGAN DIPANGGIL DI JALUR PENOLAKAN"]
    )
    gen.generate_with_custom_prompt = AsyncMock(return_value="Jawaban.")

    retr = RetrieverSkor(skor)
    p._store, p._generator, p._retriever = store, gen, retr
    p._reranker = MagicMock()
    return p, gen, retr


def _hangatkan_cache(saran: list[str], *, gaya: str | None = None) -> None:
    """Isi cache pertanyaan pembuka seperti yang terjadi saat mahasiswa membuka materi."""
    from src import learning_styles

    spec = learning_styles.resolve(gaya)
    gen_cache.save(
        "starter", "kelas-1", f"bab1.pdf::{spec.key}", saran,
        tenant_id=TENANT, material_hash=gen_cache.hash_material("isi materi contoh"),
    )


@pytest.fixture(autouse=True)
def _ambang_tetap(monkeypatch):
    monkeypatch.setattr(settings, "relevance_min_score", 0.15)
    monkeypatch.setattr(settings, "suggestion_min_score", 0.15)


class TestGerbangDiModeTerpandu:
    """Cacat 1: pertanyaan di luar materi DIJAWAB di mode terpandu."""

    async def test_topik_di_luar_materi_ditolak(self) -> None:
        p, gen, _ = _pipa(RENDAH)

        hasil = await p.guided_query(
            topic="apa resep nasi goreng",
            template_id="understand_concept",
            class_name="Basis Data",
            guided_system_prompt="sistem",
            content_id="kelas-1",
            section_ids=[1],
            tenant_id=TENANT,
        )

        assert "tidak ada materi" in hasil.answer.lower() or (
            "tidak mencakup" in hasil.answer.lower()
        ), hasil.answer
        assert hasil.sources == []
        gen.generate_with_custom_prompt.assert_not_called()

    async def test_topik_di_dalam_materi_tetap_dijawab(self) -> None:
        """Pasangan uji di atas. Tanpa ini, gerbang yang menolak SEMUANYA juga lulus."""
        p, gen, _ = _pipa(TINGGI)

        hasil = await p.guided_query(
            topic="apa itu normalisasi",
            template_id="understand_concept",
            class_name="Basis Data",
            guided_system_prompt="sistem",
            content_id="kelas-1",
            section_ids=[1],
            tenant_id=TENANT,
        )

        assert hasil.answer == "Jawaban."
        gen.generate_with_custom_prompt.assert_awaited()

    async def test_template_ringkasan_dikecualikan(self) -> None:
        """Permintaan ringkasan tidak memuat kandungan topik untuk dicocokkan,
        jadi skornya rendah walau materinya SUDAH tepat di tangan. Terukur pada
        kasus sepadan di chat bebas: skor 0,009 untuk materi yang benar."""
        p, gen, _ = _pipa(RENDAH)

        hasil = await p.guided_query(
            topic="materi minggu ini",
            template_id="structured_summary",
            class_name="Basis Data",
            guided_system_prompt="sistem",
            content_id="kelas-1",
            section_ids=[1],
            tenant_id=TENANT,
        )

        assert hasil.answer == "Jawaban.", "permintaan ringkasan tidak boleh ditolak"
        gen.generate_with_custom_prompt.assert_awaited()

    async def test_jalur_streaming_juga_bergerbang(self) -> None:
        p, gen, _ = _pipa(RENDAH)

        peristiwa = [
            ev async for ev in p.guided_query_stream(
                topic="apa resep nasi goreng",
                template_id="understand_concept",
                class_name="Basis Data",
                guided_system_prompt="sistem",
                content_id="kelas-1",
                section_ids=[1],
                tenant_id=TENANT,
            )
        ]

        selesai = [e for e in peristiwa if e["type"] == "done"]
        assert len(selesai) == 1
        assert "tidak ada materi" in selesai[0]["answer"].lower() or (
            "tidak mencakup" in selesai[0]["answer"].lower()
        )
        gen.stream_with_custom_prompt.assert_not_called()


class TestSaranSaatDitolak:
    """Cacat 2: saran harus mengacu ke MATERI, bukan ke pertanyaan yang ditolak."""

    async def test_saran_diambil_dari_materi(self) -> None:
        _hangatkan_cache(["Apa itu normalisasi?", "Bagaimana membuat indeks?"])
        p, gen, _ = _pipa(RENDAH)
        # Pemeriksa saran memakai retriever yang sama. Dibuat berskor TINGGI
        # supaya pertanyaan pembuka lolos, sementara pertanyaan mahasiswanya
        # sudah ditolak lebih dulu oleh gerbang.
        p._retriever = RetrieverSkor(TINGGI)

        saran = await p._saran_dari_materi(
            tenant_id=TENANT, content_id="kelas-1", weeks=[1],
        )

        assert saran == ["Apa itu normalisasi?", "Bagaimana membuat indeks?"]
        assert "nasi goreng" not in " ".join(saran).lower()
        gen.generate_starter_questions.assert_not_called()

    async def test_jalur_penolakan_tidak_memanggil_model(self) -> None:
        """Cache kosong -> [] , BUKAN panggilan model. Gerbang relevansi ada
        justru untuk berhenti membakar panggilan model pada pertanyaan yang tidak
        terjawab materi; membangkitkan pertanyaan pembuka di sini akan
        mengembalikan biaya yang baru saja dihemat."""
        p, gen, _ = _pipa(RENDAH)

        saran = await p._saran_dari_materi(
            tenant_id=TENANT, content_id="kelas-1", weeks=[1],
        )

        assert saran == []
        gen.generate_starter_questions.assert_not_called()

    async def test_saran_disaring_dengan_penyaring_yang_sama(self) -> None:
        """Pertanyaan pembuka dibuat untuk SATU materi, dan materi itu belum
        tentu ada di minggu yang sedang aktif. Menawarkannya tanpa diperiksa
        berarti saran berikutnya ditolak juga — gejala yang sedang diperbaiki."""
        _hangatkan_cache(["Apa itu normalisasi?"])
        p, _, _ = _pipa(RENDAH)
        p._retriever = RetrieverSkor(RENDAH)  # materinya di luar minggu aktif

        saran = await p._saran_dari_materi(
            tenant_id=TENANT, content_id="kelas-1", weeks=[9],
        )

        assert saran == [], "saran yang akan ditolak saat diklik tidak boleh ditampilkan"

    async def test_penolakan_chat_bebas_membawa_saran_materi(self) -> None:
        _hangatkan_cache(["Apa itu normalisasi?"])
        p, _, _ = _pipa(RENDAH)

        # Retriever berskor rendah menolak pertanyaannya; pemeriksa saran perlu
        # skor tinggi. Dipisah supaya dua peran itu tidak saling menutupi.
        class Dua:
            def __init__(self) -> None:
                self.n = 0

            async def retrieve(self, query: str, **kw):
                self.n += 1
                skor = RENDAH if self.n == 1 else TINGGI
                return [{"rerank_score": skor, "payload": {"source_file": "bab1.pdf"}}]

        p._retriever = Dua()

        hasil = await p.query(
            "apa resep nasi goreng", content_id="kelas-1", course_id="sbd",
            weeks=[1], tenant_id=TENANT,
        )

        assert hasil.sources == []
        assert hasil.recommendations == ["Apa itu normalisasi?"]

    async def test_katalog_gagal_tidak_menggagalkan_penolakan(self) -> None:
        """Mahasiswa sedang menunggu jawaban atas pertanyaannya, bukan laporan
        masalah internal."""
        p, _, _ = _pipa(RENDAH)
        p._store.list_indexed_files = AsyncMock(side_effect=RuntimeError("qdrant mati"))

        saran = await p._saran_dari_materi(
            tenant_id=TENANT, content_id="kelas-1", weeks=[1],
        )

        assert saran == []


class TestPengecualianGerbang:
    def test_template_ringkasan_bebas(self) -> None:
        assert _bebas_gerbang("structured_summary", "apa pun")

    def test_template_lain_tidak_bebas(self) -> None:
        assert not _bebas_gerbang("understand_concept", "apa itu normalisasi")

    def test_permintaan_ringkasan_yang_ditulis_sendiri_bebas(self) -> None:
        """Dikenali dengan pengenal yang SAMA seperti chat bebas
        (`guided.minta_ikhtisar`), bukan daftar kata kedua yang akan berbeda
        pelan-pelan dari yang pertama."""
        assert _bebas_gerbang("understand_concept", "ringkas materi minggu 1")

    def test_tanpa_template_tetap_diperiksa_teksnya(self) -> None:
        assert _bebas_gerbang(None, "rangkum bab ini")
        assert not _bebas_gerbang(None, "apa resep nasi goreng")


class TestPesanPenolakanTetapMenyebutMataKuliah:
    """Mata kuliah diturunkan dari content_id ketika rutenya mengosongkannya.

    DITEMUKAN SAAT DIJALANKAN
    `chat.py` sengaja menulis `course_id=session.course_id if not
    session.content_id else None` — supaya pencarian tidak disaring dua kali
    oleh hal yang sama. Benar untuk penyaringan, tetapi pesan penolakan ikut
    kehilangan mata kuliahnya dan jatuh ke kalimat terakhir: "Coba periksa
    apakah mata kuliahnya sudah dipilih."

    Terukur pada tumpukan yang berjalan: permintaan dengan course_id="sbd" DAN
    content_id="sbd-minggu-1" dibalas kalimat itu — menyuruh mahasiswa memeriksa
    pilihan yang baru saja ia buat. Itu kelas kesalahan yang sama dengan
    "membantah nomor minggu mahasiswa", dan satu-satunya cara menemukannya
    adalah menjalankannya.
    """

    @pytest.fixture
    def pipa(self) -> RAGPipeline:
        p = RAGPipeline.__new__(RAGPipeline)
        store = AsyncMock()
        store.list_weeks = AsyncMock(return_value=[1, 2, 3])
        p._store = store
        return p

    async def test_mata_kuliah_diturunkan_dari_content_id(self, pipa) -> None:
        pesan = await pipa._pesan_tak_terjawab(
            course_id=None, weeks=None, tenant_id=TENANT,
            content_id="sbd-minggu-1",
        )

        assert "Minggu yang sudah punya materi: 1, 2, 3" in pesan
        assert "sudah dipilih" not in pesan, (
            "mahasiswa yang sudah memilih materi tidak boleh disuruh "
            "memeriksa apakah mata kuliahnya sudah dipilih"
        )

    async def test_course_id_eksplisit_tetap_menang(self, pipa) -> None:
        pesan = await pipa._pesan_tak_terjawab(
            course_id="sbd", weeks=None, tenant_id=TENANT,
            content_id="kka-minggu-9",
        )

        pipa._store.list_weeks.assert_awaited_once()
        assert pipa._store.list_weeks.await_args.args[0] == "sbd"

    async def test_tanpa_keduanya_kalimat_itu_memang_benar(self, pipa) -> None:
        """Pasangannya: kalimat "sudah dipilih" TIDAK dihapus, hanya berhenti
        muncul di tempat yang salah. Tanpa uji ini, menghapusnya sama sekali
        juga lulus."""
        pesan = await pipa._pesan_tak_terjawab(
            course_id=None, weeks=None, tenant_id=TENANT, content_id=None,
        )

        assert "sudah dipilih" in pesan
