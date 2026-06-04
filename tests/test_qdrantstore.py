"""Tests for the QdrantStore.

The Qdrant client is mocked — we test our wrapping logic
(collection creation, batched upsert, hybrid query construction).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.schemas import Chunk, ElementType
from src.storage.qdrant_store import QdrantStore


def _chunk(
    *,
    chunk_id: str = "c1",
    element_type: ElementType = ElementType.TEXT,
    dense: list[float] | None = None,
    sparse: dict[int, float] | None = None,
    raw_html: str | None = None,
    image_base64: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="text content",
        parent_element_id="e1",
        element_type=element_type,
        source_file="doc.pdf",
        page_number=1,
        raw_html=raw_html,
        image_base64=image_base64,
        dense_embedding=dense or [0.1] * 1024,
        sparse_embedding=sparse if sparse is not None else {5: 0.5, 7: 0.3},
    )


def _make_client_and_store(enable_sparse: bool = True) -> tuple[MagicMock, QdrantStore]:
    client = MagicMock()
    client.collection_exists.return_value = False
    store = QdrantStore(client=client, collection="test_collection", enable_sparse=enable_sparse)
    return client, store


class TestEnsureCollection:
    @pytest.mark.asyncio
    async def test_creates_when_missing(self) -> None:
        client, store = _make_client_and_store()
        client.collection_exists.return_value = False

        await store.ensure_collection()

        client.create_collection.assert_called_once()
        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"
        assert "dense" in kwargs["vectors_config"]

    @pytest.mark.asyncio
    async def test_idempotent_when_exists(self) -> None:
        client, store = _make_client_and_store()
        client.collection_exists.return_value = True

        await store.ensure_collection()

        client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_sparse_config_included_when_enabled(self) -> None:
        client, store = _make_client_and_store(enable_sparse=True)

        await store.ensure_collection()

        kwargs = client.create_collection.call_args.kwargs
        assert kwargs.get("sparse_vectors_config") is not None
        assert "sparse" in kwargs["sparse_vectors_config"]

    @pytest.mark.asyncio
    async def test_sparse_config_excluded_when_disabled(self) -> None:
        client, store = _make_client_and_store(enable_sparse=False)

        await store.ensure_collection()

        kwargs = client.create_collection.call_args.kwargs
        assert kwargs.get("sparse_vectors_config") is None


class TestUpsertChunks:
    @pytest.mark.asyncio
    async def test_empty_input_returns_zero(self) -> None:
        client, store = _make_client_and_store()
        result = await store.upsert_chunks([])
        assert result == 0
        client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_chunk_without_dense_raises(self) -> None:
        _, store = _make_client_and_store()
        bad_chunk = _chunk(dense=None)
        bad_chunk.dense_embedding = None

        with pytest.raises(ValueError, match="no dense embedding"):
            await store.upsert_chunks([bad_chunk])

    @pytest.mark.asyncio
    async def test_single_batch_upserts_once(self) -> None:
        client, store = _make_client_and_store()

        result = await store.upsert_chunks([_chunk(chunk_id="a"), _chunk(chunk_id="b")])

        assert result == 2
        assert client.upsert.call_count == 1

    @pytest.mark.asyncio
    async def test_large_input_batches(self) -> None:
        """65 chunks should produce 2 batches (default batch size = 64)."""
        client, store = _make_client_and_store()
        chunks = [_chunk(chunk_id=f"c{i}") for i in range(65)]

        result = await store.upsert_chunks(chunks)

        assert result == 65
        assert client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_payload_contains_metadata(self) -> None:
        client, store = _make_client_and_store()
        chunk = _chunk(
            chunk_id="c1",
            element_type=ElementType.TABLE,
            raw_html="<table/>",
        )

        await store.upsert_chunks([chunk])

        points = client.upsert.call_args.kwargs["points"]
        payload = points[0].payload
        assert payload["text"] == "text content"
        assert payload["element_type"] == "table"
        assert payload["raw_html"] == "<table/>"
        assert payload["source_file"] == "doc.pdf"

    @pytest.mark.asyncio
    async def test_point_vector_includes_sparse_when_enabled(self) -> None:
        client, store = _make_client_and_store(enable_sparse=True)
        chunk = _chunk(chunk_id="c1", sparse={5: 0.5, 7: 0.3})

        await store.upsert_chunks([chunk])

        points = client.upsert.call_args.kwargs["points"]
        assert "sparse" in points[0].vector

    @pytest.mark.asyncio
    async def test_point_vector_excludes_sparse_when_disabled(self) -> None:
        client, store = _make_client_and_store(enable_sparse=False)
        chunk = _chunk(chunk_id="c1", sparse={5: 0.5, 7: 0.3})

        await store.upsert_chunks([chunk])

        points = client.upsert.call_args.kwargs["points"]
        assert "sparse" not in points[0].vector
        assert "dense" in points[0].vector


class TestSearch:
    @pytest.mark.asyncio
    async def test_dense_only_when_no_sparse_vector(self) -> None:
        client, store = _make_client_and_store()
        client.query_points.return_value = SimpleNamespace(points=[])

        await store.search(dense_vector=[0.1] * 1024, sparse_vector=None, top_k=5)

        kwargs = client.query_points.call_args.kwargs
        assert "prefetch" not in kwargs or kwargs.get("prefetch") is None
        assert kwargs["using"] == "dense"

    @pytest.mark.asyncio
    async def test_hybrid_when_sparse_provided(self) -> None:
        client, store = _make_client_and_store()
        client.query_points.return_value = SimpleNamespace(points=[])

        await store.search(
            dense_vector=[0.1] * 1024,
            sparse_vector={5: 0.5},
            top_k=5,
        )

        kwargs = client.query_points.call_args.kwargs
        assert "prefetch" in kwargs
        assert len(kwargs["prefetch"]) == 2

    @pytest.mark.asyncio
    async def test_returns_dicts_with_score_and_payload(self) -> None:
        client, store = _make_client_and_store()
        client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="c1", score=0.85, payload={"text": "hello"}
                ),
                SimpleNamespace(
                    id="c2", score=0.72, payload={"text": "world"}
                ),
            ]
        )

        results = await store.search(dense_vector=[0.1] * 1024, top_k=5)

        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] == 0.85
        assert results[0]["payload"]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_source_filter_applied(self) -> None:
        client, store = _make_client_and_store()
        client.query_points.return_value = SimpleNamespace(points=[])

        await store.search(
            dense_vector=[0.1] * 1024,
            source_filter="specific.pdf",
            top_k=5,
        )

        kwargs = client.query_points.call_args.kwargs
        flt = kwargs.get("query_filter")
        assert flt is not None

    @pytest.mark.asyncio
    async def test_sparse_disabled_forces_dense_even_with_sparse_vector(self) -> None:
        client, store = _make_client_and_store(enable_sparse=False)
        client.query_points.return_value = SimpleNamespace(points=[])

        await store.search(
            dense_vector=[0.1] * 1024,
            sparse_vector={5: 0.5},
            top_k=5,
        )

        kwargs = client.query_points.call_args.kwargs
        assert "prefetch" not in kwargs or kwargs.get("prefetch") is None
        assert kwargs["using"] == "dense"


class TestDeleteBySource:
    @pytest.mark.asyncio
    async def test_calls_delete_with_filter(self) -> None:
        client, store = _make_client_and_store()
        await store.delete_by_source("old_doc.pdf")
        client.delete.assert_called_once()
        kwargs = client.delete.call_args.kwargs
        assert kwargs["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_filter_matches_source_file_value(self) -> None:
        client, store = _make_client_and_store()
        await store.delete_by_source("old_doc.pdf")

        kwargs = client.delete.call_args.kwargs
        selector = kwargs["points_selector"]
        condition = selector.filter.must[0]
        assert condition.key == "source_file"
        assert condition.match.value == "old_doc.pdf"