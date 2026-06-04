"""Tests for the Embedder.

The real BGE-M3 model is mocked — we test our wrapping logic
(batching, sparse output, numpy conversion, async wrapping).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.indexing.embedder import Embedder
from src.schemas import Chunk, ElementType


def _fake_encode_output(
    n: int, *, with_sparse: bool = True
) -> dict[str, Any]:
    """Build a fake BGEM3FlagModel.encode return value."""
    dense_vecs = [[0.1 * (i + 1)] * 1024 for i in range(n)]
    out: dict[str, Any] = {"dense_vecs": dense_vecs}
    if with_sparse:
        out["lexical_weights"] = [
            {str(i * 10): 0.5, str(i * 10 + 1): 0.3} for i in range(n)
        ]
    return out


def _mock_bgem3(*, with_sparse: bool = True) -> MagicMock:
    """Build a mock that mimics BGEM3FlagModel.encode."""
    model = MagicMock()

    def _encode(texts, batch_size, return_dense, return_sparse, return_colbert_vecs):
        return _fake_encode_output(len(texts), with_sparse=return_sparse)

    model.encode = _encode
    return model


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file="doc.pdf",
        page_number=1,
    )


class TestEmbedChunks:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        result = await emb.embed_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_chunk_gets_dense_and_sparse(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        result = await emb.embed_chunks([_chunk("hello")])

        assert len(result) == 1
        assert result[0].dense_embedding is not None
        assert len(result[0].dense_embedding) == 1024
        assert result[0].sparse_embedding is not None

    @pytest.mark.asyncio
    async def test_batch_preserves_order_and_count(self) -> None:
        chunks = [_chunk(f"text{i}", chunk_id=f"c{i}") for i in range(5)]
        result = await Embedder(model=_mock_bgem3()).embed_chunks(chunks)

        assert len(result) == 5
        assert [c.chunk_id for c in result] == [f"c{i}" for i in range(5)]
        for chunk in result:
            assert chunk.dense_embedding is not None

    @pytest.mark.asyncio
    async def test_input_chunks_not_mutated(self) -> None:
        original = _chunk("hello")
        assert original.dense_embedding is None

        result = await Embedder(model=_mock_bgem3()).embed_chunks([original])

        assert original.dense_embedding is None
        assert result[0].dense_embedding is not None
        assert result[0].chunk_id == original.chunk_id

    @pytest.mark.asyncio
    async def test_sparse_disabled_when_hybrid_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import settings

        monkeypatch.setattr(settings, "enable_hybrid_search", False, raising=False)
        emb = Embedder(model=_mock_bgem3(with_sparse=False))
        result = await emb.embed_chunks([_chunk("hello")])

        assert result[0].dense_embedding is not None
        assert result[0].sparse_embedding is None

    @pytest.mark.asyncio
    async def test_sparse_keys_converted_to_int(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        result = await emb.embed_chunks([_chunk("hello")])

        assert result[0].sparse_embedding is not None
        for key in result[0].sparse_embedding.keys():
            assert isinstance(key, int)


class TestEmbedQuery:
    @pytest.mark.asyncio
    async def test_returns_dense_and_sparse_tuple(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        dense, sparse = await emb.embed_query("Apa itu statistik?")

        assert len(dense) == 1024
        assert sparse is not None
        assert isinstance(sparse, dict)

    @pytest.mark.asyncio
    async def test_empty_query_raises(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        with pytest.raises(ValueError, match="Empty query"):
            await emb.embed_query("")

    @pytest.mark.asyncio
    async def test_whitespace_only_query_raises(self) -> None:
        emb = Embedder(model=_mock_bgem3())
        with pytest.raises(ValueError, match="Empty query"):
            await emb.embed_query("   \n  ")

    @pytest.mark.asyncio
    async def test_sparse_none_when_hybrid_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import settings

        monkeypatch.setattr(settings, "enable_hybrid_search", False, raising=False)
        emb = Embedder(model=_mock_bgem3(with_sparse=False))
        dense, sparse = await emb.embed_query("query")

        assert dense is not None
        assert sparse is None


class TestNumpyConversion:
    @pytest.mark.asyncio
    async def test_numpy_array_converted_to_list(self) -> None:
        """Ensure ._to_list() handles objects with .tolist()."""
        import numpy as np

        model = MagicMock()
        model.encode = lambda texts, **_: {
            "dense_vecs": np.array([[0.1, 0.2, 0.3]]),
            "lexical_weights": [{"5": 0.9}],
        }
        emb = Embedder(model=model)
        result = await emb.embed_chunks([_chunk("hi")])

        assert isinstance(result[0].dense_embedding, list)
        assert result[0].dense_embedding == [0.1, 0.2, 0.3]