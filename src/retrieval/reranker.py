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


@lru_cache(maxsize=1)
def _load_reranker_model() -> Any:
    """Load reranker once. Deferred import so module loads without FlagEmbedding."""
    from FlagEmbedding import FlagReranker

    logger.info(
        "Loading reranker model {} on {}",
        settings.reranker_model,
        settings.embed_device,
    )
    return FlagReranker(
        settings.reranker_model,
        use_fp16=settings.embed_device != "cpu",
        device=settings.embed_device,
    )


class Reranker:
    """Async wrapper around the cross-encoder reranker.

    The model is loaded lazily on first use unless one is injected (testing).
    """

    def __init__(self, model: Any = None) -> None:
        self._model = model

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
        for c in candidates:
            text = (c.get("payload") or {}).get(text_field, "")
            pairs.append([query, text])

        scores = await asyncio.to_thread(self._compute_scores, pairs)

        scored = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        result = scored[:top_k]
        logger.info(
            "Reranked {} → top {}, best score={:.3f}",
            len(candidates),
            len(result),
            result[0]["rerank_score"] if result else 0.0,
        )
        return result

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        """Sync score computation. Called inside asyncio.to_thread."""
        model = self._ensure_model()
        scores = model.compute_score(pairs, normalize=True)

        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]