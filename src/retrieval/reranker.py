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

import httpx

from src.config import settings
from src.utils.logger import logger
from src.utils.openrouter import OPENROUTER_API_BASE_URL, openrouter_headers

# Anggaran token untuk SATU pasangan pada jalur LOKAL — kueri dan potongan
# digabung, bukan per sisi. Dulu argumen ini tidak pernah dikirim ke
# `compute_score`, sehingga FlagReranker diam-diam memakai 512: potongan
# indeks kami bermedian 432 token dengan maksimum 519 (15 dari 44 potongan di
# atas 480), jadi begitu token kueri ikut dihitung, sebagian ekor dokumen
# memang dibuang tanpa satu pun catatan.
#
# Angkanya sengaja TETAP 512, bukan dinaikkan. Cross-encoder ini berjalan di
# CPU dengan ongkos kira-kira 1,96 detik per pasangan, dan ongkos itu naik
# seiring panjang urutan; melebarkan batas demi menyelamatkan belasan token
# ekor pada sebagian kecil potongan berarti menambah puluhan detik pada
# SETIAP pertanyaan mahasiswa.
#
# Yang diperbaiki bukan angkanya, melainkan ketidaktahuannya: batas kini
# dikirim eksplisit dan setiap pasangan yang terpotong dicatat. Menaikkannya
# kelak cukup lewat RERANKER_MAX_LENGTH di .env, tanpa menyunting berkas ini.
#
# TIDAK berlaku untuk jalur openrouter: model rerank terkelola (cohere/
# rerank-4-pro) memotong sendiri di sisi penyedia dengan jendela jauh lebih
# lebar, dan API-nya tidak menerima parameter panjang sama sekali.


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

    async def warmup(self) -> None:
        """Load the local reranker model during startup when using local mode."""
        if settings.reranker_backend == "local":
            await asyncio.to_thread(self._ensure_model)

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
        # Penanda dibawa terpisah supaya peringatan pemotongan menyebut
        # potongan yang mana — tanpa itu, catatan "3 pasangan terpotong"
        # tidak bisa ditindaklanjuti karena tak ada yang tahu dokumen apa
        # yang dirugikan.
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
            "Reranked {} → top {}, best score={:.3f}",
            len(candidates), len(result),
            result[0]["rerank_score"] if result else 0.0,
        )
        return result

    def _compute_scores(
        self, pairs: list[list[str]], labels: list[str] | None = None
    ) -> list[float]:
        """Sync score computation. Called inside asyncio.to_thread."""
        if settings.reranker_backend == "openrouter":
            # Jalur terkelola: batas panjangnya milik penyedia, bukan milik
            # kode ini, jadi tidak ada pemotongan yang bisa dicatat di sini.
            return self._compute_scores_openrouter(pairs)

        model = self._ensure_model()
        self._log_truncated_pairs(model, pairs, labels)
        # `max_length` dikirim eksplisit: batasnya harus milik kode ini, bukan
        # bawaan pustaka yang bisa berubah di versi berikutnya tanpa kami
        # sadari.
        scores = model.compute_score(
            pairs, normalize=True, max_length=self._max_length
        )

        if isinstance(scores, int | float):
            return [float(scores)]
        return [float(s) for s in scores]

    def _log_truncated_pairs(
        self, model: Any, pairs: list[list[str]], labels: list[str] | None
    ) -> None:
        """Catat pasangan yang ekornya dibuang karena melewati `max_length`.

        Pemotongannya sendiri diterima (lihat catatan di kepala modul); yang
        tidak diterima adalah pemotongan yang tak terlihat: skor dokumen yang
        separuh isinya tidak pernah dibaca model tampak sama saja dengan skor
        dokumen utuh. Catatan ini yang memberi tahu bahwa peringkat sedang
        dihitung atas teks yang tidak lengkap.
        """
        lengths = self._pair_token_lengths(model, pairs)
        if not lengths:
            return

        over = [(i, n) for i, n in enumerate(lengths) if n > self._max_length]
        if not over:
            return

        names = [
            labels[i] if labels and i < len(labels) else f"#{i}" for i, _ in over
        ]
        terpanjang = max(n for _, n in over)
        logger.warning(
            "Reranker memotong {} dari {} pasangan pada max_length={} "
            "(terpanjang {} token, kelebihan {}); potongan terdampak: {}",
            len(over),
            len(pairs),
            self._max_length,
            terpanjang,
            terpanjang - self._max_length,
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

    def _compute_scores_openrouter(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []

        query = pairs[0][0]
        documents = [pair[1] for pair in pairs]

        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "provider": {"allow_fallbacks": True},
        }

        try:
            response = httpx.post(
                f"{OPENROUTER_API_BASE_URL}/rerank",
                headers=openrouter_headers(),
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"OpenRouter rerank request failed: {exc}") from exc

        data = response.json()
        raw_results = data.get("results") or data.get("data") or data.get("scores") or []

        scores = [0.0] * len(documents)
        if isinstance(raw_results, list):
            for position, item in enumerate(raw_results):
                if isinstance(item, int | float):
                    if position < len(scores):
                        scores[position] = float(item)
                    continue

                if not isinstance(item, dict):
                    continue

                index = item.get("index", position)
                score = item.get("relevance_score")
                if score is None:
                    score = item.get("score")
                if score is None:
                    score = item.get("rerank_score")
                if score is None:
                    continue

                try:
                    scores[int(index)] = float(score)
                except (TypeError, ValueError, IndexError):
                    continue

        return scores