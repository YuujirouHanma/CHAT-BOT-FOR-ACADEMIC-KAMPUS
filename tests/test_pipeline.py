"""Integration tests for the full RAGPipeline.

All sub-components are mocked. We verify the overall orchestration:
index_document calls every layer in order; query returns the correct
result shape and handles edge cases (empty retrieval).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generation.prompts import FormattedContext
from src.pipeline import RAGPipeline
from src.schemas import Chunk, ElementType, ParsedElement


def _parsed_element(element_id: str = "e1") -> ParsedElement:
    return ParsedElement(
        element_id=element_id,
        element_type=ElementType.TEXT,
        content="hello",
        source_file="doc.pdf",
        page_number=1,
    )


def _chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="hello",
        parent_element_id="e1",
        element_type=ElementType.TEXT,
        source_file="doc.pdf",
        page_number=1,
        dense_embedding=[0.1] * 1024,
    )


def _make_pipeline() -> tuple[RAGPipeline, dict]:
    summarizer = MagicMock()
    chunker = MagicMock()
    chunker.chunk = MagicMock(return_value=[_chunk()])

    embedder = AsyncMock()
    embedder.embed_chunks = AsyncMock(return_value=[_chunk()])

    store = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.upsert_chunks = AsyncMock(return_value=1)

    reranker = AsyncMock()
    generator = AsyncMock()
    generator.generate = AsyncMock(return_value="Jawaban final.")
    generator.decompose_query = AsyncMock(return_value={})
    generator.generate_followup = AsyncMock(return_value=[])

    pipeline = RAGPipeline(
        summarizer=summarizer,
        chunker=chunker,
        embedder=embedder,
        store=store,
        reranker=reranker,
        generator=generator,
    )
    return pipeline, {
        "summarizer": summarizer,
        "chunker": chunker,
        "embedder": embedder,
        "store": store,
        "reranker": reranker,
        "generator": generator,
    }


class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_ensures_collection(self) -> None:
        pipeline, mocks = _make_pipeline()
        await pipeline.setup()
        mocks["store"].ensure_collection.assert_awaited_once()


class TestIndexDocument:
    @pytest.mark.asyncio
    async def test_full_indexing_flow(self, tmp_path: Path) -> None:
        pipeline, mocks = _make_pipeline()
        doc = tmp_path / "test.pdf"
        doc.write_bytes(b"fake pdf content")

        with patch(
            "src.pipeline.parse_document",
            return_value=[_parsed_element()],
        ), patch(
            "src.pipeline.enrich_elements",
            new_callable=AsyncMock,
            return_value=[_parsed_element()],
        ) as enrich:
            result = await pipeline.index_document(doc)

        enrich.assert_awaited_once()
        mocks["chunker"].chunk.assert_called_once()
        mocks["embedder"].embed_chunks.assert_awaited_once()
        mocks["store"].upsert_chunks.assert_awaited_once()

        assert result.source_file == "test.pdf"
        assert result.elements_parsed == 1
        assert result.chunks_created == 1
        assert result.points_stored == 1

    @pytest.mark.asyncio
    async def test_empty_parse_returns_zero_counts(self, tmp_path: Path) -> None:
        pipeline, mocks = _make_pipeline()
        doc = tmp_path / "empty.pdf"
        doc.write_bytes(b"empty")

        with patch("src.pipeline.parse_document", return_value=[]):
            result = await pipeline.index_document(doc)

        assert result.elements_parsed == 0
        assert result.chunks_created == 0
        mocks["embedder"].embed_chunks.assert_not_awaited()
        mocks["store"].upsert_chunks.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_chunks_skips_embedding_and_storage(
        self, tmp_path: Path
    ) -> None:
        pipeline, mocks = _make_pipeline()
        mocks["chunker"].chunk = MagicMock(return_value=[])
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"x")

        with patch(
            "src.pipeline.parse_document",
            return_value=[_parsed_element()],
        ), patch(
            "src.pipeline.enrich_elements",
            new_callable=AsyncMock,
            return_value=[_parsed_element()],
        ):
            result = await pipeline.index_document(doc)

        assert result.chunks_created == 0
        mocks["embedder"].embed_chunks.assert_not_awaited()


class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(self) -> None:
        pipeline, mocks = _make_pipeline()

        retriever_results = [
            {
                "chunk_id": "c1",
                "score": 0.9,
                "rerank_score": 0.95,
                "payload": {
                    "text": "isi konteks",
                    "source_file": "doc.pdf",
                    "page_number": 1,
                    "element_type": "text",
                },
            }
        ]
        pipeline._retriever.retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=retriever_results
        )

        result = await pipeline.query("Apa itu X?")

        assert result.answer == "Jawaban final."
        assert len(result.sources) == 1
        assert result.sources[0]["source_file"] == "doc.pdf"
        assert result.sources[0]["index"] == 1
        mocks["generator"].generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_fallback_message(self) -> None:
        pipeline, mocks = _make_pipeline()
        pipeline._retriever.retrieve = AsyncMock(return_value=[])  # type: ignore[method-assign]

        result = await pipeline.query("Pertanyaan obscure?")

        assert "tidak mencakup" in result.answer
        assert result.sources == []
        mocks["generator"].generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_source_filter_passed_through(self) -> None:
        pipeline, _ = _make_pipeline()
        retrieve_mock = AsyncMock(return_value=[])
        pipeline._retriever.retrieve = retrieve_mock  # type: ignore[method-assign]

        await pipeline.query("q", source_filter="specific.pdf")

        retrieve_mock.assert_awaited_once_with(
            query="q", content_id=None, source_filter="specific.pdf"
        )

    @pytest.mark.asyncio
    async def test_generator_receives_formatted_context(self) -> None:
        pipeline, mocks = _make_pipeline()
        retriever_results = [
            {
                "chunk_id": "c1",
                "score": 0.9,
                "rerank_score": 0.95,
                "payload": {
                    "text": "hello world",
                    "source_file": "doc.pdf",
                    "page_number": 1,
                    "element_type": "text",
                },
            }
        ]
        pipeline._retriever.retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=retriever_results
        )

        await pipeline.query("q")

        call_args = mocks["generator"].generate.call_args
        question_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("question")
        context_arg = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("context")

        assert question_arg == "q"
        assert isinstance(context_arg, FormattedContext)
        assert "hello world" in context_arg.text_block