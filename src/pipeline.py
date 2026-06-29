"""End-to-end pipeline orchestration.

1. INDEXING: parse → enrich → chunk → embed → store
2. QUERYING: retrieve → format context → generate
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.generation.llm import LLMGenerator
from src.generation.prompts import format_retrieval_results, parse_answer_and_suggestions
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
    course: str | None = None
    week: int | None = None


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    recommendations: list[str]


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
        self, file_path: Path, course: str | None = None, week: int | None = None,
    ) -> IndexResult:
        logger.info("=== Indexing {} (course={}, week={}) ===", file_path.name, course, week)

        elements = parse_document(file_path, course=course, week=week)
        if not elements:
            return IndexResult(file_path.name, 0, 0, 0, course=course, week=week)

        enriched = await enrich_elements(elements, self._summarizer)
        chunks = self._chunker.chunk(enriched)
        if not chunks:
            return IndexResult(file_path.name, len(elements), 0, 0, course=course, week=week)

        embedded = await self._embedder.embed_chunks(chunks)
        stored = await self._store.upsert_chunks(embedded)

        logger.info("=== Done {}: {} → {} → {} ===",
                     file_path.name, len(elements), len(chunks), stored)
        return IndexResult(
            source_file=file_path.name,
            elements_parsed=len(elements),
            chunks_created=len(chunks),
            points_stored=stored,
            course=course,
            week=week,
        )

    async def query(
        self,
        question: str,
        course: str | None = None,
        week: int | None = None,
        source_filter: str | None = None,
    ) -> QueryResult:
        logger.info("=== Query: '{}' (course={}, week={}) ===", question[:80], course, week)

        results = await self._retriever.retrieve(
            query=question, course=course, week=week, source_filter=source_filter,
        )
        if not results:
            return QueryResult(
                answer="Materi yang tersedia tidak mencakup informasi tersebut.",
                sources=[],
                recommendations=[],
            )

        context = format_retrieval_results(results)
        raw_answer = await self._generator.generate(question, context)
        answer, recommendations = parse_answer_and_suggestions(raw_answer)

        sources = [
            {
                "index": idx + 1,
                "source_file": (r.get("payload") or {}).get("source_file"),
                "page_number": (r.get("payload") or {}).get("page_number"),
                "element_type": (r.get("payload") or {}).get("element_type"),
                "course": (r.get("payload") or {}).get("course"),
                "week": (r.get("payload") or {}).get("week"),
                "rerank_score": r.get("rerank_score"),
            }
            for idx, r in enumerate(results)
        ]
        return QueryResult(answer=answer, sources=sources, recommendations=recommendations)