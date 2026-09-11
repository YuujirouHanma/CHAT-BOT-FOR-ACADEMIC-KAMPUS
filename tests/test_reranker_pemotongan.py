"""Uji regresi: pemotongan pasangan pada reranker harus disengaja dan terlihat.

Cacat aslinya: `compute_score` dipanggil tanpa `max_length`, sehingga batas 512
token per pasangan (kueri + potongan) datang dari bawaan pustaka, dan potongan
yang melebihinya dinilai dalam keadaan terpotong tanpa jejak apa pun.

Dua hal yang dikunci di sini: batas dikirim eksplisit oleh kode kita, dan
pasangan yang terpotong dicatat beserta identitas potongannya.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import settings
from src.retrieval import reranker as reranker_mod
from src.retrieval.reranker import Reranker


def _candidate(text: str, chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": 0.5,
        "payload": {"text": text, "source_file": "doc.pdf"},
    }


class _FakeTokenizer:
    """Tokenizer sekadarnya: satu kata = satu token, plus token khusus pasangan.

    Meniru bentuk keluaran tokenizer HuggingFace (`{"input_ids": [[...], ...]}`)
    supaya kode yang diuji tetap yang sesungguhnya, tanpa memuat model 2 GB.
    """

    SPECIAL_TOKENS = 4  # <s> kueri </s></s> dokumen </s>, seperti XLM-R

    def __call__(
        self,
        queries: list[str],
        documents: list[str],
        **_kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        ids = [
            [0] * (len(q.split()) + len(d.split()) + self.SPECIAL_TOKENS)
            for q, d in zip(queries, documents, strict=True)
        ]
        return {"input_ids": ids}


def _model(scores: list[float], *, tokenizer: Any = None) -> MagicMock:
    model = MagicMock()
    model.compute_score = MagicMock(return_value=scores)
    # MagicMock membuatkan atribut apa pun secara otomatis, jadi ketiadaan
    # tokenizer harus dinyatakan tegas — kalau tidak, `getattr` mengembalikan
    # mock yang mengaku bisa dipanggil.
    model.tokenizer = tokenizer
    return model


def _panjang_kata(n: int) -> str:
    return " ".join(["kata"] * n)


@pytest.fixture
def catat_peringatan(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Tangkap panggilan `logger.warning` dari modul reranker."""
    peringatan: list[tuple] = []
    fake = MagicMock()
    fake.warning = lambda msg, *args, **kwargs: peringatan.append((msg, args))
    monkeypatch.setattr(reranker_mod, "logger", fake)
    return peringatan


class TestBatasPanjangEksplisit:
    @pytest.mark.asyncio
    async def test_compute_score_menerima_max_length(self) -> None:
        """Tanpa ini, batas token ditentukan bawaan FlagReranker, bukan kode kita."""
        model = _model([0.6, 0.4])

        await Reranker(model=model).rerank("q", [_candidate("a", "c1"), _candidate("b", "c2")])

        kwargs = model.compute_score.call_args.kwargs
        assert kwargs["max_length"] == settings.reranker_max_length
        # Skor ternormalisasi dipakai di tempat lain; jangan ikut berubah.
        assert kwargs["normalize"] is True

    @pytest.mark.asyncio
    async def test_batas_dapat_ditimpa_per_instans(self) -> None:
        model = _model([0.5])

        await Reranker(model=model, max_length=1024).rerank("q", [_candidate("a", "c1")])

        assert model.compute_score.call_args.kwargs["max_length"] == 1024


class TestPeringatanPemotongan:
    @pytest.mark.asyncio
    async def test_pasangan_terpotong_dicatat_dengan_id_potongan(
        self, catat_peringatan: list[tuple]
    ) -> None:
        """Potongan 519 token + kueri melewati 512 → wajib ada peringatan."""
        candidates = [
            _candidate(_panjang_kata(300), "aman"),
            _candidate(_panjang_kata(519), "kepanjangan"),
        ]
        model = _model([0.3, 0.9], tokenizer=_FakeTokenizer())

        await Reranker(model=model).rerank(_panjang_kata(8), candidates, top_k=2)

        assert len(catat_peringatan) == 1
        _, args = catat_peringatan[0]
        assert args[0] == 1                    # satu pasangan terpotong
        assert args[1] == 2                    # dari dua pasangan
        assert args[2] == settings.reranker_max_length
        assert args[3] == 519 + 8 + 4          # panjang pasangan sebenarnya
        assert "kepanjangan" in args[-1]
        assert "aman" not in args[-1]

    @pytest.mark.asyncio
    async def test_pasangan_muat_tidak_memicu_peringatan(
        self, catat_peringatan: list[tuple]
    ) -> None:
        candidates = [_candidate(_panjang_kata(100), "c1")]
        model = _model([0.8], tokenizer=_FakeTokenizer())

        await Reranker(model=model).rerank("kueri pendek", candidates, top_k=1)

        assert catat_peringatan == []

    @pytest.mark.asyncio
    async def test_batas_dinaikkan_membuat_potongan_yang_sama_muat(
        self, catat_peringatan: list[tuple]
    ) -> None:
        """Peringatan mengikuti batas yang berlaku, bukan angka mati 512."""
        candidates = [_candidate(_panjang_kata(519), "kepanjangan")]
        model = _model([0.9], tokenizer=_FakeTokenizer())

        await Reranker(model=model, max_length=768).rerank("q", candidates, top_k=1)

        assert catat_peringatan == []


class TestKetahananPemeriksaan:
    @pytest.mark.asyncio
    async def test_model_tanpa_tokenizer_tetap_menilai(
        self, catat_peringatan: list[tuple]
    ) -> None:
        """Pemeriksaan hanya pencatatan; ketiadaannya tidak boleh menggagalkan rerank."""
        model = _model([0.7], tokenizer=None)

        hasil = await Reranker(model=model).rerank("q", [_candidate("teks", "c1")], top_k=1)

        assert hasil[0]["rerank_score"] == 0.7
        assert catat_peringatan == []

    @pytest.mark.asyncio
    async def test_tokenizer_bermasalah_tidak_menggagalkan_rerank(
        self, catat_peringatan: list[tuple]
    ) -> None:
        def tokenizer_rusak(*_args: Any, **_kwargs: Any) -> dict:
            raise RuntimeError("tokenizer tidak mendukung pasangan")

        model = _model([0.42], tokenizer=tokenizer_rusak)

        hasil = await Reranker(model=model).rerank("q", [_candidate("teks", "c1")], top_k=1)

        assert hasil[0]["rerank_score"] == 0.42
        assert catat_peringatan == []
