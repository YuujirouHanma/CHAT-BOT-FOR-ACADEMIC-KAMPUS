"""Qdrant vector store wrapper.

Collection layout — designed for bge-m3 hybrid search:
- One named DENSE vector ("dense") with cosine distance and dim from settings.
- One named SPARSE vector ("sparse"), enabled when settings.enable_hybrid_search.
- Payload mirrors the Chunk schema except the embeddings themselves.

Operations are async. Real Qdrant calls use `asyncio.to_thread` because the
official client is sync (AsyncQdrantClient exists but has fewer features
and a less stable API).

Hybrid search uses Qdrant's built-in `Prefetch` + `FusionQuery` (RRF) as of
qdrant-client >= 1.10, which fuses dense and sparse results server-side.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.config import settings
from src.schemas import Chunk
from src.utils.logger import logger

_DENSE_VEC = "dense"
_SPARSE_VEC = "sparse"
_UPSERT_BATCH = 64


class QdrantStore:
    """Thin async wrapper around qdrant-client for our chunk schema."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        enable_sparse: bool | None = None,
    ) -> None:
        self._client = client or self._build_client()
        self._collection = collection or settings.qdrant_collection
        self._enable_sparse = (
            enable_sparse if enable_sparse is not None else settings.enable_hybrid_search
        )

    @staticmethod
    def _build_client() -> QdrantClient:
        """Build the Qdrant client from settings.

        local  → embedded on-disk client, no server required (dev/tests).
        server → connect to the Qdrant server (Docker/production).
        """
        if settings.qdrant_mode == "server":
            logger.info(
                "Qdrant client: SERVER mode at {}:{}",
                settings.qdrant_host,
                settings.qdrant_port,
            )
            return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

        logger.info("Qdrant client: LOCAL mode (path={})", settings.qdrant_path)
        return QdrantClient(path=settings.qdrant_path)

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        exists = await asyncio.to_thread(
            self._client.collection_exists, self._collection
        )
        if exists:
            logger.info("Qdrant collection '{}' already exists", self._collection)
            return

        vectors_config = {
            _DENSE_VEC: qm.VectorParams(
                size=settings.embed_dim,
                distance=qm.Distance.COSINE,
            )
        }
        sparse_vectors_config = (
            {_SPARSE_VEC: qm.SparseVectorParams(modifier=qm.Modifier.IDF)}
            if self._enable_sparse
            else None
        )

        await asyncio.to_thread(
            self._client.create_collection,
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        logger.info(
            "Created Qdrant collection '{}' (dim={}, sparse={})",
            self._collection,
            settings.embed_dim,
            self._enable_sparse,
        )

    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks in batches. Returns number of points written."""
        if not chunks:
            return 0

        points = [self._chunk_to_point(c) for c in chunks]
        total = 0
        for batch_start in range(0, len(points), _UPSERT_BATCH):
            batch = points[batch_start : batch_start + _UPSERT_BATCH]
            await asyncio.to_thread(
                self._client.upsert,
                collection_name=self._collection,
                points=batch,
                wait=True,
            )
            total += len(batch)
        logger.info("Upserted {} points to '{}'", total, self._collection)
        return total

    async def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        top_k: int | None = None,
        source_filter: str | None = None,
        content_id: str | None = None,
    ) -> list[dict]:
        """Hybrid search if sparse vector is given and enabled, else dense-only.

        Args:
            dense_vector: 1024-dim dense embedding of the query.
            sparse_vector: token_id → weight map; required for hybrid.
            top_k: Number of results; defaults to settings.retrieval_top_k.
            source_filter: Optional file name to restrict results.
            content_id: Optional content identifier to restrict results.

        Returns:
            List of dicts: {"chunk_id", "score", "payload"}.
        """
        top_k = top_k or settings.retrieval_top_k
        qfilter = self._build_filter(
            source_filter=source_filter, content_id=content_id
        )

        use_hybrid = self._enable_sparse and sparse_vector is not None
        result = None
        if use_hybrid:
            assert sparse_vector is not None
            try:
                result = await asyncio.to_thread(
                    self._client.query_points,
                    collection_name=self._collection,
                    prefetch=[
                        qm.Prefetch(
                            query=dense_vector,
                            using=_DENSE_VEC,
                            limit=top_k * 2,
                            filter=qfilter,
                        ),
                        qm.Prefetch(
                            query=qm.SparseVector(
                                indices=list(sparse_vector.keys()),
                                values=list(sparse_vector.values()),
                            ),
                            using=_SPARSE_VEC,
                            limit=top_k * 2,
                            filter=qfilter,
                        ),
                    ],
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=top_k,
                    with_payload=True,
                )
            except KeyError as exc:
                # Qdrant local mode raises KeyError('sparse') from _rescore_idf
                # when the IDF-modified sparse index has no entries yet — e.g. a
                # collection with zero sparse-indexed points. Degrade gracefully
                # to dense-only instead of surfacing a 500 to the caller.
                logger.warning(
                    "Hybrid search unavailable ({}); falling back to dense-only", exc
                )

        if result is None:
            result = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self._collection,
                query=dense_vector,
                using=_DENSE_VEC,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )

        return [
            {
                "chunk_id": str(p.id),
                "score": float(p.score),
                "payload": dict(p.payload or {}),
            }
            for p in result.points
        ]

    async def list_indexed_files(
        self,
        content_id: str | None = None,
    ) -> list[dict]:
        """Return unique source files indexed for the given content_id.

        Scrolls Qdrant with an optional content_id filter and deduplicates
        by source_file. Returns list of {"source_file", "content_id"}.
        """
        qfilter = self._build_filter(content_id=content_id)

        seen: set[str] = set()
        results: list[dict] = []
        offset: Any = None

        while True:
            batch, next_offset = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self._collection,
                scroll_filter=qfilter,
                limit=100,
                offset=offset,
                with_payload=["source_file", "content_id"],
                with_vectors=False,
            )
            for point in batch:
                sf = (point.payload or {}).get("source_file")
                if sf and sf not in seen:
                    seen.add(sf)
                    results.append({
                        "source_file": sf,
                        "content_id": (point.payload or {}).get("content_id"),
                    })
            if next_offset is None:
                break
            offset = next_offset

        return results

    async def delete_by_source(self, source_file: str) -> None:
        """Delete all chunks belonging to one source file."""
        await asyncio.to_thread(
            self._client.delete,
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source_file",
                            match=qm.MatchValue(value=source_file),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted all chunks for source_file='{}'", source_file)

    @staticmethod
    def _build_filter(
        source_filter: str | None = None,
        content_id: str | None = None,
    ) -> qm.Filter | None:
        conditions: list[Any] = []
        if source_filter:
            conditions.append(
                qm.FieldCondition(key="source_file", match=qm.MatchValue(value=source_filter))
            )
        if content_id:
            conditions.append(
                qm.FieldCondition(key="content_id", match=qm.MatchValue(value=content_id))
            )
        if not conditions:
            return None
        return qm.Filter(must=conditions)

    def _chunk_to_point(self, chunk: Chunk) -> qm.PointStruct:
        if chunk.dense_embedding is None:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no dense embedding; "
                "run Embedder.embed_chunks() first."
            )

        vector: dict[str, Any] = {
            _DENSE_VEC: chunk.dense_embedding
        }
        if self._enable_sparse and chunk.sparse_embedding:
            vector[_SPARSE_VEC] = qm.SparseVector(
                indices=list(chunk.sparse_embedding.keys()),
                values=list(chunk.sparse_embedding.values()),
            )

        payload = {
            "text": chunk.text,
            "parent_element_id": chunk.parent_element_id,
            "element_type": chunk.element_type.value,
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "content_id": chunk.content_id,
            "raw_html": chunk.raw_html,
            "image_base64": chunk.image_base64,
        }

        return qm.PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload=payload,
        )
