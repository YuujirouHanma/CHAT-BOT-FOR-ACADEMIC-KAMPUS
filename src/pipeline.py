"""End-to-end pipeline orchestration.

1. INDEXING: parse → enrich → chunk → embed → store
2. QUERYING: retrieve → format context → generate
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.generation.llm import LLMGenerator
from src.generation.prompts import format_retrieval_results
from src.hitl.logger import log_interaction
from src.indexing.chunker import Chunker
from src.indexing.embedder import Embedder
from src.indexing.summarizer import MultimodalSummarizer, enrich_elements
from src.ingestion.parser import parse_document
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.storage.qdrant_store import QdrantStore
from src.utils.logger import logger


@dataclass
class IndexResult:
    source_file: str
    elements_parsed: int
    chunks_created: int
    points_stored: int
    content_id: str | None = None


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    recommendations: list[str]
    interaction_id: str | None = None
    decomposition: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        summarizer: MultimodalSummarizer | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
        reranker: Reranker | None = None,
        generator: LLMGenerator | None = None,
    ) -> None:
        self._summarizer = summarizer or MultimodalSummarizer()
        self._chunker = chunker or Chunker()
        self._embedder = embedder or Embedder()
        self._store = store or QdrantStore()
        self._reranker = reranker or Reranker()
        self._generator = generator or LLMGenerator()
        self._retriever = HybridRetriever(
            embedder=self._embedder, store=self._store, reranker=self._reranker,
        )

    async def setup(self) -> None:
        await self._store.ensure_collection()

    async def index_document(
        self, file_path: Path, content_id: str | None = None,
    ) -> IndexResult:
        logger.info("=== Indexing {} (content_id={}) ===", file_path.name, content_id)

        elements = parse_document(file_path, content_id=content_id)
        if not elements:
            return IndexResult(file_path.name, 0, 0, 0, content_id=content_id)

        enriched = await enrich_elements(elements, self._summarizer)
        chunks = self._chunker.chunk(enriched)
        if not chunks:
            return IndexResult(file_path.name, len(elements), 0, 0, content_id=content_id)

        embedded = await self._embedder.embed_chunks(chunks)
        stored = await self._store.upsert_chunks(embedded)

        logger.info("=== Done {}: {} → {} → {} ===",
                     file_path.name, len(elements), len(chunks), stored)
        return IndexResult(
            source_file=file_path.name,
            elements_parsed=len(elements),
            chunks_created=len(chunks),
            points_stored=stored,
            content_id=content_id,
        )

    async def query(
        self,
        question: str,
        content_id: str | None = None,
        source_filter: str | None = None,
        session_id: str | None = None,
    ) -> QueryResult:
        logger.info("=== Query: '{}' (content_id={}) ===", question[:80], content_id)
        start = time.monotonic()

        # Stage 1: decompose the question into an enriched, more searchable query.
        dq = await self._generator.decompose_query(question)
        enriched_query = dq.get("query_diperkaya") or question

        # Stage 2: retrieve using the enriched query.
        results = await self._retriever.retrieve(
            query=enriched_query, content_id=content_id, source_filter=source_filter,
        )
        if not results:
            result = QueryResult(
                answer="Materi yang tersedia tidak mencakup informasi tersebut.",
                sources=[],
                recommendations=[],
                decomposition=dq,
            )
            result.interaction_id = log_interaction(
                question=question, dq=dq, answer=result.answer, sources=[],
                recommendations=[], elapsed_seconds=time.monotonic() - start,
                content_id=content_id, session_id=session_id,
            )
            return result

        context = format_retrieval_results(results)
        answer = await self._generator.generate(question, context)

        sources = [
            {
                "index": idx + 1,
                "source_file": (r.get("payload") or {}).get("source_file"),
                "page_number": (r.get("payload") or {}).get("page_number"),
                "element_type": (r.get("payload") or {}).get("element_type"),
                "content_id": (r.get("payload") or {}).get("content_id"),
                "rerank_score": r.get("rerank_score"),
            }
            for idx, r in enumerate(results)
        ]

        # Stage 5: generate follow-up questions as a separate call from the main answer.
        recommendations = await self._generator.generate_followup(question, dq, answer)

        interaction_id = log_interaction(
            question=question, dq=dq, answer=answer, sources=sources,
            recommendations=recommendations, elapsed_seconds=time.monotonic() - start,
            content_id=content_id, session_id=session_id,
        )
        return QueryResult(
            answer=answer, sources=sources, recommendations=recommendations,
            interaction_id=interaction_id, decomposition=dq,
        )
