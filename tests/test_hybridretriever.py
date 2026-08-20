"""Tests for HybridRetriever.

Embedder, QdrantStore, and Reranker are all mocked. We verify the
orchestration: query is embedded, candidates retrieved, then reranked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from tests.conftest import TEST_TENANT_ID


def _candidate(chunk_id: str, score: float = 0.5, text: str = "x") -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "payload": {"text": text, "source_file": "doc.pdf"},
    }


def _make_retriever(
    *,
    candidates: list[dict] | None = None,
    reranked: list[dict] | None = None,
) -> tuple[HybridRetriever, AsyncMock, AsyncMock, AsyncMock]:
    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=([0.1] * 1024, {5: 0.5}))

    store = AsyncMock()
    store.search = AsyncMock(return_value=candidates if candidates is not None else [])

    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=reranked if reranked is not None else [])

    retriever = HybridRetriever(
        embedder=embedder, store=store, reranker=reranker
    )
    return retriever, embedder.embed_query, store.search, reranker.rerank


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_empty_query_raises(self) -> None:
        retriever, _, _, _ = _make_retriever()
        with pytest.raises(ValueError, match="Empty query"):
            await retriever.retrieve("", tenant_id=TEST_TENANT_ID)

    @pytest.mark.asyncio
    async def test_whitespace_query_raises(self) -> None:
        retriever, _, _, _ = _make_retriever()
        with pytest.raises(ValueError, match="Empty query"):
            await retriever.retrieve("   \n  ", tenant_id=TEST_TENANT_ID)

    @pytest.mark.asyncio
    async def test_full_pipeline_called_in_order(self) -> None:
        candidates = [_candidate("c1"), _candidate("c2")]
        reranked = [_candidate("c2", score=0.9)]
        retriever, embed, search, rerank = _make_retriever(
            candidates=candidates, reranked=reranked
        )

        result = await retriever.retrieve("question?", tenant_id=TEST_TENANT_ID)

        embed.assert_awaited_once_with("question?")
        search.assert_awaited_once()
        rerank.assert_awaited_once()
        assert result == reranked

    @pytest.mark.asyncio
    async def test_empty_candidates_skips_rerank(self) -> None:
        retriever, _, _, rerank = _make_retriever(candidates=[])
        result = await retriever.retrieve("q", tenant_id=TEST_TENANT_ID)
        assert result == []
        rerank.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_receives_both_vectors(self) -> None:
        retriever, _, search, _ = _make_retriever(
            candidates=[_candidate("c1")],
            reranked=[_candidate("c1")],
        )
        await retriever.retrieve("q", tenant_id=TEST_TENANT_ID)

        kwargs = search.call_args.kwargs
        assert kwargs["dense_vector"] == [0.1] * 1024
        assert kwargs["sparse_vector"] == {5: 0.5}

    @pytest.mark.asyncio
    async def test_source_filter_propagated_to_store(self) -> None:
        retriever, _, search, _ = _make_retriever(
            candidates=[_candidate("c1")],
            reranked=[_candidate("c1")],
        )
        await retriever.retrieve("q", source_filter="my.pdf", tenant_id=TEST_TENANT_ID)

        kwargs = search.call_args.kwargs
        assert kwargs["source_filter"] == "my.pdf"

    @pytest.mark.asyncio
    async def test_custom_top_k_overrides_defaults(self) -> None:
        retriever, _, search, rerank = _make_retriever(
            candidates=[_candidate("c1")],
            reranked=[_candidate("c1")],
        )
        await retriever.retrieve("q", retrieval_top_k=50, rerank_top_k=3, tenant_id=TEST_TENANT_ID)

        assert search.call_args.kwargs["top_k"] == 50
        assert rerank.call_args.kwargs["top_k"] == 3