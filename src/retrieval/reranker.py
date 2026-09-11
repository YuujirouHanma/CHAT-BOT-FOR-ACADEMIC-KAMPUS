"""Reranking layer using bge-reranker-v2-m3.

Cross-encoder reranker — takes (query, candidate) pairs and emits a more
accurate relevance score than the initial vector retrieval. This is what
turns "decent recall@20" into "high precision@5" before LLM generation.

The model is sync; we wrap encoding in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from src.config import settings
from src.utils.logger import logger

# Anggaran token untuk SATU pasangan — kueri dan potongan digabung, bukan per
# sisi. Dulu argumen ini tidak pernah dikirim ke `compute_score`, sehingga
# FlagReranker diam-diam memakai 512: potongan indeks kami bermedian 432 token
# dengan maksimum 519 (15 dari 44 potongan di atas 480), jadi begitu token kueri
# ikut dihitung, sebagian ekor dokumen memang dibuang tanpa satu pun catatan.
#
# Angkanya sengaja TETAP 512, bukan dinaikkan. Cross-encoder ini berjalan di CPU
# dengan ongkos ±1,96 detik per pasangan (39 detik untuk 20 pasangan) dan ongkos
# itu naik seiring panjang urutan; melebarkan batas ke 768 demi menyelamatkan
# belasan token ekor pada sebagian kecil potongan berarti menambah puluhan detik
# pada SETIAP pertanyaan mahasiswa — pertukaran yang buruk, karena kalimat
# terakhir sebuah potongan jarang menjadi penentu peringkatnya.
#
# Yang diperbaiki di sini bukan angkanya, melainkan ketidaktahuannya: batas kini
# dikirim eksplisit (perilaku sama, tetapi jadi keputusan yang tertulis) dan
# setiap pasangan yang terpotong dicatat lewat `_log_truncated_pairs`. Dengan
# begitu, menaikkan batas kelak menjadi keputusan berdasar data terukur — dan
# menaikkannya kini cukup lewat `RERANKER_MAX_LENGTH` di .env, tanpa menyunting
# berkas ini.


@lru_cache(maxsize=1)
def _load_reranker_model() -> Any:
    """Load reranker once. Deferred import so module loads without FlagEmbedding."""
    from FlagEmbedding import FlagReranker

    logger.info(f"Loading reranker model {settings.reranker_model} on {settings.embed_device}")
    return FlagReranker(
        settings.reranker_model,
        use_fp16=settings.embed_device != "cpu",
        device=settings.embed_device,
    )


class Reranker:
    """Async wrapper around the cross-encoder reranker.

    The model is loaded lazily on first use unless one is injected (testing).
    """

    def __init__(self, model: Any = None, max_length: int | None = None) -> None:
        self._model = model
        # Bawaannya dari settings (RERANKER_MAX_LENGTH di .env); masih dapat
        # ditimpa per instans untuk percobaan dan pengujian.
        self._max_length = (
            max_length if max_length is not None else settings.reranker_max_length
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_reranker_model()
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: Sequence[dict],
        top_k: int | None = None,
        text_field: str = "text",
    ) -> list[dict]:
        """Rerank candidates by relevance to the query.

        Args:
            query: User query.
            candidates: List of dicts from QdrantStore.search(); each must
                contain `payload[text_field]`.
            top_k: How many top-scored items to return; defaults to
                settings.rerank_top_k.
            text_field: Which payload key holds the text to compare.

        Returns:
            Top-k candidates sorted by reranker score, with `rerank_score`
            field added. Original `score` (from Qdrant) is preserved.
        """
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k

        pairs = []
        # Penanda dibawa terpisah supaya peringatan pemotongan menyebut potongan
        # yang mana — tanpa itu, catatan "3 pasangan terpotong" tidak bisa
        # ditindaklanjuti karena tak ada yang tahu dokumen apa yang dirugikan.
        labels = []
        for i, c in enumerate(candidates):
            text = (c.get("payload") or {}).get(text_field, "")
            pairs.append([query, text])
            labels.append(str(c.get("chunk_id") or f"#{i}"))

        scores = await asyncio.to_thread(self._compute_scores, pairs, labels)

        scored = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        result = scored[:top_k]
        logger.info(
            f"Reranked {len(candidates)} → top {len(result)}, best score={result[0]['rerank_score'] if result else 0.0:.3f}"
        )
        return result

    def _compute_scores(
        self, pairs: list[list[str]], labels: list[str] | None = None
    ) -> list[float]:
        """Sync score computation. Called inside asyncio.to_thread."""
        model = self._ensure_model()
        self._log_truncated_pairs(model, pairs, labels)
        # `max_length` dikirim eksplisit: batasnya harus milik kode ini, bukan
        # bawaan pustaka yang bisa berubah di versi berikutnya tanpa kami sadari.
        scores = model.compute_score(pairs, normalize=True, max_length=self._max_length)

        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    def _log_truncated_pairs(
        self, model: Any, pairs: list[list[str]], labels: list[str] | None
    ) -> None:
        """Catat pasangan yang ekornya dibuang karena melewati `max_length`.

        Pemotongan itu sendiri diterima (lihat settings.reranker_max_length), yang tidak
        diterima adalah pemotongan yang tak terlihat: skor dokumen yang separuh
        isinya tidak pernah dibaca model tampak sama saja dengan skor dokumen
        utuh. Catatan ini yang memberi tahu bahwa peringkat sedang dihitung atas
        teks yang tidak lengkap.
        """
        lengths = self._pair_token_lengths(model, pairs)
        if not lengths:
            return

        over = [(i, n) for i, n in enumerate(lengths) if n > self._max_length]
        if not over:
            return

        names = [labels[i] if labels and i < len(labels) else f"#{i}" for i, _ in over]
        logger.warning(
            "Reranker memotong {} dari {} pasangan pada max_length={} "
            "(terpanjang {} token, kelebihan {}); potongan terdampak: {}",
            len(over),
            len(pairs),
            self._max_length,
            max(n for _, n in over),
            max(n for _, n in over) - self._max_length,
            ", ".join(names),
        )

    def _pair_token_lengths(self, model: Any, pairs: list[list[str]]) -> list[int]:
        """Panjang token tiap pasangan SEBELUM dipotong, atau [] bila tak terukur.

        Memakai tokenizer milik model itu sendiri, bukan taksiran dari jumlah
        karakter: rasio karakter-per-token teks kuliah berbahasa Indonesia yang
        bercampur istilah asing dan rumus terlalu berubah-ubah untuk dipakai
        memutuskan "terpotong atau tidak". Ongkosnya satu lintasan tokenisasi
        tambahan — hitungan milidetik di samping inferensi yang memakan detik.

        Model suntikan pada uji (atau versi FlagEmbedding yang menaruh
        tokenizer-nya di tempat lain) tidak wajib punya tokenizer; bila tidak
        ada, pemeriksaan dilewati. Ini pencatatan, bukan syarat penilaian.
        """
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return []
        try:
            encoded = tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                # Justru panjang asli yang ingin diketahui, jadi pemotongan dan
                # padding tokenizer harus dimatikan di sini.
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
            return [len(ids) for ids in encoded["input_ids"]]
        except Exception as exc:  # noqa: BLE001 - tokenizer pihak ketiga
            # Pencatatan tidak boleh menggagalkan reranking yang sudah berjalan.
            logger.debug("Lewati pemeriksaan pemotongan reranker: {}", exc)
            return []