"""Pertanyaan lanjutan harus dipahami sebagai lanjutan, bukan pertanyaan baru.

Ditemukan saat uji manual: menanyakan sesuatu, lalu menyusul "jelaskan lebih
detail", menghasilkan jawaban atas materi yang SAMA SEKALI LAIN. Buktinya ada
pada nomor halaman sumbernya — giliran pertama mengutip halaman 182 dan
411-419, giliran kedua mengutip 31, 40, 41, 308, dan 355. Bukan jawaban yang
lebih dangkal; jawaban atas pertanyaan lain.

Sebabnya riwayat percakapan hanya dipakai menyusun saran pertanyaan lanjutan
(tahap 5), dan tidak pernah sampai ke dua tempat yang menentukan:

1. Penguraian pertanyaan (tahap 1), yang hasilnya dipakai MENCARI materi.
   "Jelaskan lebih detail" tidak memuat satu pun kata benda yang bisa dicari.
2. Pembuatan jawaban, sehingga model tidak tahu apa yang barusan dibahas.

Yang diuji di sini: riwayatnya benar-benar sampai. Apakah model kemudian
menguraikannya dengan benar adalah perkara mutu model, bukan perkara kode, dan
tidak bisa dipaku oleh uji.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.llm import LLMGenerator
from src.generation.prompts import FormattedContext

RIWAYAT = [
    {"role": "user", "content": "Apa itu normalisasi basis data?"},
    {"role": "assistant", "content": "Normalisasi adalah proses menyusun tabel..."},
]


def _generator(balasan: str = '{"query_diperkaya": "x"}'):
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=balasan), finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
        ),
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    return LLMGenerator(client=client), create  # type: ignore[arg-type]


def _konteks() -> FormattedContext:
    return FormattedContext(text_block="[Sumber 1] Isi materi.", image_payloads=[])


class TestPenguraianPertanyaan:
    """Tahap 1 — hasilnya dipakai MENCARI, jadi di sinilah akibatnya terbesar."""

    @pytest.mark.asyncio
    async def test_riwayat_masuk_ke_prompt(self) -> None:
        gen, create = _generator()
        await gen.decompose_query("jelaskan lebih detail", history=RIWAYAT)

        isi = create.await_args.kwargs["messages"][-1]["content"]
        assert "RIWAYAT PERCAKAPAN" in isi
        assert "normalisasi basis data" in isi.lower(), (
            "pokok bahasan giliran sebelumnya tidak sampai ke penguraian, "
            "sehingga pencarian berjalan atas kalimat 'jelaskan lebih detail'"
        )
        assert "jelaskan lebih detail" in isi

    @pytest.mark.asyncio
    async def test_tanpa_riwayat_prompt_tidak_berubah(self) -> None:
        """Giliran pertama tidak boleh ikut membawa bagian kosong."""
        gen, create = _generator()
        await gen.decompose_query("apa itu normalisasi?")

        isi = create.await_args.kwargs["messages"][-1]["content"]
        assert "RIWAYAT PERCAKAPAN" not in isi
        assert isi.startswith("[PERTANYAAN MAHASISWA]")

    @pytest.mark.asyncio
    async def test_aturan_penguraian_rujukan_ada_di_prompt_sistem(self) -> None:
        """Riwayat saja tidak cukup — model harus disuruh menguraikannya.

        Tanpa aturan ini model cenderung menyalin pertanyaannya apa adanya ke
        `query_diperkaya`, dan riwayat yang sudah dikirim jadi tidak berguna.
        """
        gen, create = _generator()
        await gen.decompose_query("kenapa begitu", history=RIWAYAT)

        sistem = create.await_args.kwargs["messages"][0]["content"]
        assert "RIWAYAT PERCAKAPAN" in sistem
        assert "TANPA membaca riwayat" in sistem


class TestPembuatanJawaban:
    @pytest.mark.asyncio
    async def test_riwayat_dikirim_sebagai_pesan_terpisah(self) -> None:
        """Bentuk percakapan, bukan teks yang disisipkan ke prompt pengguna."""
        gen, create = _generator("Jawaban.")
        await gen.generate("jelaskan lebih detail", _konteks(), history=RIWAYAT)

        pesan = create.await_args.kwargs["messages"]
        peran = [p["role"] for p in pesan]
        assert peran == ["system", "user", "assistant", "user"], peran
        assert "normalisasi" in pesan[1]["content"].lower()

    @pytest.mark.asyncio
    async def test_tanpa_riwayat_hanya_dua_pesan(self) -> None:
        gen, create = _generator("Jawaban.")
        await gen.generate("apa itu normalisasi?", _konteks())

        peran = [p["role"] for p in create.await_args.kwargs["messages"]]
        assert peran == ["system", "user"]

    @pytest.mark.asyncio
    async def test_riwayat_panjang_dipangkas(self) -> None:
        """Riwayat berbagi jatah token dengan jawabannya."""
        panjang = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000}
            for i in range(20)
        ]
        gen, create = _generator("Jawaban.")
        await gen.generate("lanjut", _konteks(), history=panjang)

        pesan = create.await_args.kwargs["messages"]
        assert len(pesan) <= 8, f"riwayat tidak dipangkas: {len(pesan)} pesan"
        for p in pesan[1:-1]:
            assert len(p["content"]) < 2000


class TestPipelineMengoperRiwayat:
    """Jalur lengkap: `/chat/ask` sampai ke kedua tempat itu.

    Kedua potongan di atas bisa benar sendiri-sendiri dan tetap tidak berguna
    bila pipeline tidak pernah mengoperkan riwayatnya — dan itulah yang terjadi
    sebelum perbaikan ini.
    """

    @pytest.mark.asyncio
    async def test_riwayat_sampai_ke_penguraian_dan_jawaban(self) -> None:
        from src.pipeline import RAGPipeline

        gen = MagicMock()
        gen.decompose_query = AsyncMock(return_value={"query_diperkaya": "normalisasi"})
        gen.generate = AsyncMock(return_value="Jawaban.")
        gen.generate_followup = AsyncMock(return_value=[])

        retriever = MagicMock()
        retriever.retrieve = AsyncMock(return_value=[
            {"payload": {"source_file": "a.pdf", "page_number": 1}, "rerank_score": 0.9},
        ])

        pipe = RAGPipeline(
            summarizer=MagicMock(), chunker=MagicMock(), embedder=MagicMock(),
            store=MagicMock(), reranker=MagicMock(), generator=gen,
        )
        pipe._retriever = retriever

        await pipe.query(
            "jelaskan lebih detail", history=RIWAYAT, tenant_id="kampus-a",
        )

        assert gen.decompose_query.await_args.kwargs["history"] == RIWAYAT, (
            "riwayat tidak sampai ke tahap 1 — pencarian tetap memakai kalimat mentah"
        )
        assert gen.generate.await_args.kwargs["history"] == RIWAYAT, (
            "riwayat tidak sampai ke pembuat jawaban"
        )
