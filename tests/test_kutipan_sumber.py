"""Isi potongan yang dikutip ikut di balasan, tetapi TIDAK ikut ke log.

MENGAPA MEDAN INI ADA
.pptx dan .docx tidak punya penampil bawaan peramban, jadi panel sitasi hanya
bisa menyebutkan "halaman 7, buka berkasnya sendiri". Mahasiswa yang ingin
memeriksa satu kalimat harus mengunduh satu dek enam puluh slide lalu mencari
sendiri. Dengan `SourceInfo.text`, panelnya bisa menampilkan bagian yang
BENAR-BENAR dikutip tanpa penampil pihak ketiga — artinya tanpa mengirim materi
kelas ke layanan luar.

MENGAPA PEMISAHANNYA DIUJI, BUKAN SEKADAR DIKOMENTARI
Teks materi sudah tersimpan di Qdrant. Kalau ia ikut tercatat ke log HITL, ada
dua salinan materi kampus di dua penyimpanan dengan masa simpan dan jalur
penghapusan yang berbeda — dan yang terlupakan saat satu kampus dihapus tidak
akan terlihat sebagai galat.

Pemisahan seperti itu adalah hal pertama yang hilang ketika seseorang
"menyederhanakan" dengan memakai satu daftar untuk dua tujuan. Karena itu
diuji, bukan dipercayakan pada komentar.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.config import settings
from src.pipeline import (
    _MAKS_CUPLIKAN_SUMBER,
    RAGPipeline,
    _cuplikan_sumber,
    _sumber_tanpa_teks,
)
from tests.conftest import TEST_TENANT_ID as TENANT


def _hasil(teks: str = "isi potongan yang dikutip") -> dict:
    return {
        "chunk_id": "c1",
        "rerank_score": 0.95,
        "payload": {
            "text": teks,
            "source_file": "materi.pptx",
            "page_number": 7,
            "element_type": "text",
            "content_id": "kelas-1",
        },
    }


class TestCuplikan:
    def test_isi_diteruskan_apa_adanya(self) -> None:
        """Yang ingin diperiksa mahasiswa adalah apa yang BENAR-BENAR dibaca
        model, bukan versi yang sudah dipercantik."""
        assert _cuplikan_sumber("  SELECT * FROM mahasiswa;  ") == "SELECT * FROM mahasiswa;"

    def test_kosong_jadi_none_bukan_string_kosong(self) -> None:
        """Supaya panel bisa membedakan "tidak ada kutipan" dari "kutipan kosong"
        dengan satu pemeriksaan, bukan dua."""
        assert _cuplikan_sumber("") is None
        assert _cuplikan_sumber("   ") is None
        assert _cuplikan_sumber(None) is None

    def test_potongan_raksasa_dipangkas(self) -> None:
        """Penjaga, bukan pilihan rancangan: potongan sudah dibatasi pemecah
        teks, jadi ini hanya menahan yang entah bagaimana lolos jauh lebih besar
        — tabel lebar, lampiran yang terbaca sebagai satu blok."""
        hasil = _cuplikan_sumber("x" * (_MAKS_CUPLIKAN_SUMBER + 500))

        assert hasil is not None
        assert len(hasil) == _MAKS_CUPLIKAN_SUMBER + 1  # + penanda potong
        assert hasil.endswith("…")

    def test_tepat_di_batas_tidak_dipangkas(self) -> None:
        hasil = _cuplikan_sumber("x" * _MAKS_CUPLIKAN_SUMBER)

        assert hasil == "x" * _MAKS_CUPLIKAN_SUMBER
        assert not hasil.endswith("…")


class TestSumberDibangun:
    def test_text_ikut_di_daftar_sumber(self) -> None:
        sumber = RAGPipeline._build_sources([_hasil()])

        assert sumber[0]["text"] == "isi potongan yang dikutip"
        assert sumber[0]["source_file"] == "materi.pptx"
        assert sumber[0]["page_number"] == 7

    def test_potongan_tanpa_teks_tidak_menggagalkan(self) -> None:
        """Elemen gambar diindeks lewat deskripsinya; yang tanpa teks tetap sah
        sebagai sumber, hanya tidak punya kutipan untuk ditampilkan."""
        hasil = _hasil()
        hasil["payload"].pop("text")

        sumber = RAGPipeline._build_sources([hasil])

        assert sumber[0]["text"] is None
        assert sumber[0]["source_file"] == "materi.pptx"


class TestTidakIkutKeLog:
    def test_text_dibuang_medan_lain_dipertahankan(self) -> None:
        sumber = RAGPipeline._build_sources([_hasil()])

        bersih = _sumber_tanpa_teks(sumber)

        assert "text" not in bersih[0]
        # Sitasi di riwayat dosen tetap menyebut berkas dan halamannya, jadi
        # jawaban tetap dapat ditelusuri ke sumbernya.
        assert bersih[0]["source_file"] == "materi.pptx"
        assert bersih[0]["page_number"] == 7
        assert bersih[0]["index"] == 1
        assert bersih[0]["rerank_score"] == 0.95

    def test_daftar_asli_tidak_diubah(self) -> None:
        """Kalau salinannya dangkal, membuang `text` untuk log ikut membuangnya
        dari yang dikirim ke klien — dan panelnya kosong tanpa satu pun galat."""
        sumber = RAGPipeline._build_sources([_hasil()])

        _sumber_tanpa_teks(sumber)

        assert sumber[0]["text"] == "isi potongan yang dikutip"

    async def test_balasan_membawa_teks_log_tidak(self, monkeypatch) -> None:
        """Uji yang sesungguhnya: satu permintaan utuh, lalu bandingkan apa yang
        sampai ke klien dengan apa yang sampai ke log."""
        monkeypatch.setattr(settings, "relevance_min_score", 0.15)
        monkeypatch.setattr(settings, "suggestion_min_score", 0.0)

        p = RAGPipeline.__new__(RAGPipeline)
        p._store = AsyncMock()
        p._reranker = MagicMock()

        gen = AsyncMock()
        gen.decompose_query = AsyncMock(return_value={"query_diperkaya": "indeks"})
        gen.generate = AsyncMock(return_value="Jawaban.")
        gen.generate_followup = AsyncMock(return_value=[])
        p._generator = gen

        class Retr:
            async def retrieve(self, query: str, **kw):
                return [_hasil()]

        p._retriever = Retr()

        with patch("src.pipeline.log_interaction", return_value="int_1") as log:
            hasil = await p.query(
                "apa itu indeks", content_id="kelas-1", tenant_id=TENANT,
            )

        assert hasil.sources[0]["text"] == "isi potongan yang dikutip"

        dicatat = log.call_args.kwargs["sources"]
        assert "text" not in dicatat[0], (
            "teks materi tidak boleh tersalin ke log HITL — ia sudah tersimpan "
            "di Qdrant, dan dua salinan berarti dua tempat yang harus diingat "
            "saat satu kampus dihapus"
        )
        assert dicatat[0]["source_file"] == "materi.pptx"
