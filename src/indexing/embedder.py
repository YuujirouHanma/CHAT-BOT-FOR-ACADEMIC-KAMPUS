"""Embedding layer using BGE-M3.

BGE-M3 is unique: it emits both DENSE (semantic, 1024-dim) and SPARSE
(BM25-like, lexical) vectors from a SINGLE model in one forward pass.
We use both for hybrid retrieval — controlled by settings.enable_hybrid_search.

The underlying FlagEmbedding library is sync and CPU/GPU-bound. We wrap
`encode` in `asyncio.to_thread` so it doesn't block the FastAPI event loop.

Model loading is deferred and cached: the ~2GB checkpoint is loaded once
on first real use, then reused across all Embedder instances.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from src.config import settings
from src.schemas import Chunk
from src.utils.logger import logger

_DEFAULT_BATCH_SIZE = 12


@lru_cache(maxsize=1)
def _load_bgem3_model() -> Any:
    """Load BGE-M3 once. Deferred import keeps module load-time fast and
    means tests can run without FlagEmbedding installed."""
    from FlagEmbedding import BGEM3FlagModel

    logger.info(
        "Loading embedding model {} on {} (this may take ~30s on first run)",
        settings.embed_model,
        settings.embed_device,
    )
    return BGEM3FlagModel(    #bgem3 model
        settings.embed_model, 
        use_fp16=settings.embed_device != "cpu",
        device=settings.embed_device,
    )


class Embedder:
    """Async wrapper around the BGE-M3 model.

    The model is loaded lazily on first use unless one is injected (testing).
    Sparse output is included when settings.enable_hybrid_search is True.
    """

    def __init__(
        self,
        model: Any = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._enable_sparse = settings.enable_hybrid_search

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_bgem3_model()
        return self._model

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Embed each chunk's text. Returns new Chunks with embeddings populated.

        Empty input → empty output. Real model work runs in a worker thread
        so we don't block the event loop.
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        outputs = await asyncio.to_thread(self._encode, texts)

        logger.info("Embedded {} chunks", len(chunks))
        return [
            chunk.model_copy(
                update={
                    "dense_embedding": outputs["dense"][i],
                    "sparse_embedding": (
                        outputs["sparse"][i] if self._enable_sparse else None
                    ),
                }
            )
            for i, chunk in enumerate(chunks)
        ]

    async def embed_query(
        self, query: str
    ) -> tuple[list[float], dict[int, float] | None]:
        """Embed a single query. Returns (dense_vec, sparse_vec_or_None)."""
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("Empty query")

        outputs = await asyncio.to_thread(self._encode, [cleaned])
        dense = outputs["dense"][0]
        sparse = outputs["sparse"][0] if self._enable_sparse else None
        return dense, sparse

    def _encode(self, texts: list[str]) -> dict[str, list[Any]]:
        """Sync encode. Always called inside asyncio.to_thread."""
        model = self._ensure_model()
        raw = model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=self._enable_sparse,
            return_colbert_vecs=False,
        )
        dense_vecs = [self._to_list(v) for v in raw["dense_vecs"]]

        sparse_vecs: list[dict[int, float]] = []
        if self._enable_sparse:
            for weights in raw["lexical_weights"]:
                sparse_vecs.append(
                    {int(k): float(v) for k, v in weights.items()}
                )

        return {"dense": dense_vecs, "sparse": sparse_vecs}

    @staticmethod
    def _to_list(vec: Any) -> list[float]:
        """Convert numpy array or list to plain Python list of floats."""
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(x) for x in vec]