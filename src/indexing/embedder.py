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
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import httpx

from src.config import settings
from src.schemas import Chunk
from src.utils.logger import logger
from src.utils.openrouter import OPENROUTER_API_BASE_URL, openrouter_headers

_DEFAULT_BATCH_SIZE = 12

# Cache embedding KUERI (bukan dokumen), dipakai bersama semua instance Embedder.
#
# Terukur: satu kueri tiga kata memakan 4-20 detik lewat OpenRouter, sekitar 25%
# dari total latensi guided, dan kueri identik berturut-turut tidak menghemat
# apa pun. Topik guided datang dari daftar chip yang TETAP, jadi kueri yang sama
# diembed berulang kali sepanjang umur proses.
#
# Hanya untuk kueri: embedding dokumen terjadi sekali saat indexing dan hasilnya
# sudah disimpan permanen di Qdrant, jadi tidak ada yang bisa dihemat di sana.
_QUERY_CACHE_MAX = 256
_query_cache: OrderedDict[tuple, tuple[list[float], dict[int, float] | None]] = OrderedDict()


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
        self._enable_sparse = (
            settings.enable_hybrid_search and settings.embedding_backend == "local"
        )
        self._embedding_dim: int | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _load_bgem3_model()
        return self._model

    async def warmup(self) -> None:
        """Load the embedding model and infer vector dimensions during startup."""
        await self.embed_query("warmup")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim or settings.embed_dim

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
        """Embed a single query. Returns (dense_vec, sparse_vec_or_None).

        Hasilnya di-cache per proses — lihat _query_cache untuk alasannya.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            raise ValueError("Empty query")

        # Model dan backend ikut jadi kunci: vektor dari model lain punya dimensi
        # dan ruang makna berbeda, jadi memakainya kembali akan merusak hasil
        # pencarian tanpa memunculkan galat apa pun.
        key = (
            settings.embedding_backend,
            settings.embed_model,
            self._enable_sparse,
            cleaned,
        )

        cached = _query_cache.get(key)
        if cached is not None:
            _query_cache.move_to_end(key)
            logger.info("Embedding kueri diambil dari cache (hemat 1 panggilan)")
            return cached

        outputs = await asyncio.to_thread(self._encode, [cleaned])
        dense = outputs["dense"][0]
        sparse = outputs["sparse"][0] if self._enable_sparse else None

        _query_cache[key] = (dense, sparse)
        if len(_query_cache) > _QUERY_CACHE_MAX:
            _query_cache.popitem(last=False)

        return dense, sparse

    def _encode(self, texts: list[str]) -> dict[str, list[Any]]:
        """Sync encode. Always called inside asyncio.to_thread."""
        if settings.embedding_backend == "openrouter":
            return self._encode_openrouter(texts)

        model = self._ensure_model()
        raw = model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=self._enable_sparse,
            return_colbert_vecs=False,
        )
        dense_vecs = [self._to_list(v) for v in raw["dense_vecs"]]

        if dense_vecs:
            self._embedding_dim = len(dense_vecs[0])

        sparse_vecs: list[dict[int, float]] = []
        if self._enable_sparse:
            for weights in raw["lexical_weights"]:
                sparse_vecs.append({int(k): float(v) for k, v in weights.items()})

        return {"dense": dense_vecs, "sparse": sparse_vecs}

    def _encode_openrouter(self, texts: list[str]) -> dict[str, list[Any]]:
        payload = {
            "model": settings.embed_model,
            "input": texts,
            "encoding_format": "float",
        }

        try:
            response = httpx.post(
                f"{OPENROUTER_API_BASE_URL}/embeddings",
                headers=openrouter_headers(),
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"OpenRouter embedding request failed: {exc}") from exc

        data = response.json()
        dense_vecs = [self._to_list(item["embedding"]) for item in (data.get("data") or [])]
        if dense_vecs:
            self._embedding_dim = len(dense_vecs[0])

        return {"dense": dense_vecs, "sparse": []}

    @staticmethod
    def _to_list(vec: Any) -> list[float]:
        """Convert numpy array or list to plain Python list of floats."""
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(x) for x in vec]