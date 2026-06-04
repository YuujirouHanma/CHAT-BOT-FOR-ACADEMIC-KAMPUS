"""Tests for the Reranker.

FlagReranker is mocked — we test that our wrapper sorts correctly,
respects top_k, and handles edge cases.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.reranker import Reranker


def _candidate(text: str, chunk_id: str, score: float = 0.5) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "payload": {"text": text, "source_file": "doc.pdf"},
    }


def _mock_reranker(scores: list[float]) -> MagicMock:
    """Build a fake FlagReranker returning the given scores in order."""
    model = MagicMock()
    model.compute_score = MagicMock(return_value=scores)
    return model


class TestReranker:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self) -> None:
        result = await Reranker(model=_mock_reranker([])).rerank("q", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_sorts_by_rerank_score_descending(self) -> None:
        candidates = [
            _candidate("low", "c1"),
            _candidate("high", "c2"),
            _candidate("mid", "c3"),
        ]
        model = _mock_reranker([0.2, 0.9, 0.5])

        result = await Reranker(model=model).rerank("q", candidates, top_k=3)

        assert [c["chunk_id"] for c in result] == ["c2", "c3", "c1"]
        assert [c["rerank_score"] for c in result] == [0.9, 0.5, 0.2]

    @pytest.mark.asyncio
    async def test_top_k_truncates_results(self) -> None:
        candidates = [_candidate(f"text{i}", f"c{i}") for i in range(10)]
        scores = [i / 10 for i in range(10)]
        model = _mock_reranker(scores)

        result = await Reranker(model=model).rerank("q", candidates, top_k=3)

        assert len(result) == 3
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]
        assert result[1]["rerank_score"] >= result[2]["rerank_score"]

    @pytest.mark.asyncio
    async def test_original_score_preserved(self) -> None:
        candidates = [_candidate("text", "c1", score=0.8)]
        model = _mock_reranker([0.99])

        result = await Reranker(model=model).rerank("q", candidates, top_k=1)

        assert result[0]["score"] == 0.8
        assert result[0]["rerank_score"] == 0.99

    @pytest.mark.asyncio
    async def test_payload_preserved(self) -> None:
        candidates = [_candidate("hello world", "c1")]
        model = _mock_reranker([0.7])

        result = await Reranker(model=model).rerank("q", candidates, top_k=1)

        assert result[0]["payload"]["text"] == "hello world"
        assert result[0]["payload"]["source_file"] == "doc.pdf"

    @pytest.mark.asyncio
    async def test_single_candidate_handles_scalar_score(self) -> None:
        """compute_score returns a scalar for a single pair, not a list."""
        candidates = [_candidate("hello", "c1")]
        model = MagicMock()
        model.compute_score = MagicMock(return_value=0.75)

        result = await Reranker(model=model).rerank("q", candidates, top_k=1)

        assert result[0]["rerank_score"] == 0.75

    @pytest.mark.asyncio
    async def test_compute_score_called_with_query_pairs(self) -> None:
        candidates = [
            _candidate("first", "c1"),
            _candidate("second", "c2"),
        ]
        model = _mock_reranker([0.6, 0.4])

        await Reranker(model=model).rerank("my question", candidates, top_k=2)

        called_pairs = model.compute_score.call_args.args[0]
        assert called_pairs == [
            ["my question", "first"],
            ["my question", "second"],
        ]

    @pytest.mark.asyncio
    async def test_default_top_k_from_settings(self) -> None:
        from src.config import settings

        n = settings.rerank_top_k + 3
        candidates = [_candidate(f"t{i}", f"c{i}") for i in range(n)]
        scores = [1.0 - i * 0.01 for i in range(n)]

        result = await Reranker(model=_mock_reranker(scores)).rerank("q", candidates)
        assert len(result) == settings.rerank_top_k