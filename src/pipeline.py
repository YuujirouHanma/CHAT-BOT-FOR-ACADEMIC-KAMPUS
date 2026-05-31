"""End-to-end pipeline orchestration.

Two flows:

1. INDEXING (offline, when a document is uploaded):
   parse → enrich (summarize tables/images) → chunk → embed → store

2. QUERYING (online, when a user asks):
   retrieve (embed + hybrid search + rerank) → format context → generate

Each component is constructed once and reused, so the pipeline can be
instantiated as a long-lived singleton in the FastAPI app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.generation.llm import LLMGenerator
from src.generation.prompts import format_retrieval_results
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


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]


class RAGPipeline:
    """Long-lived pipeline holding all components.

    Construct once at app startup. Both `index_document` and `query` are
    safe to call concurrently from many requests.
    """

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
            embedder=self._embedder,
            store=self._store,
            reranker=self._reranker,
        )

    async def setup(self) -> None:
        """Ensure the vector collection exists. Call once at startup."""
        await self._store.ensure_collection()

    async def index_document(self, file_path: Path) -> IndexResult:
        """Run the full indexing flow for one document."""
        logger.info("=== Indexing {} ===", file_path.name)

        elements = parse_document(file_path)
        if not elements:
            logger.warning("No elements parsed from {}", file_path.name)
            return IndexResult(file_path.name, 0, 0, 0)

        enriched = await enrich_elements(elements, self._summarizer)
        chunks = self._chunker.chunk(enriched)
        if not chunks:
            logger.warning("No chunks produced for {}", file_path.name)
            return IndexResult(file_path.name, len(elements), 0, 0)

        embedded = await self._embedder.embed_chunks(chunks)
        stored = await self._store.upsert_chunks(embedded)

        logger.info(
            "=== Done {}: {} elements → {} chunks → {} stored ===",
            file_path.name,
            len(elements),
            len(chunks),
            stored,
        )
        return IndexResult(
            source_file=file_path.name,
            elements_parsed=len(elements),
            chunks_created=len(chunks),
            points_stored=stored,
        )

    async def query(
        self,
        question: str,
        source_filter: str | None = None,
    ) -> QueryResult:
        """Answer a user question using retrieved context."""
        logger.info("=== Querying: '{}' ===", question[:80])

        results = await self._retriever.retrieve(
            query=question,
            source_filter=source_filter,
        )
        if not results:
            return QueryResult(
                answer="Materi yang tersedia tidak mencakup informasi tersebut.",
                sources=[],
            )

        context = format_retrieval_results(results)
        answer = await self._generator.generate(question, context)

        sources = [
            {
                "index": idx + 1,
                "source_file": (r.get("payload") or {}).get("source_file"),
                "page_number": (r.get("payload") or {}).get("page_number"),
                "element_type": (r.get("payload") or {}).get("element_type"),
                "rerank_score": r.get("rerank_score"),
            }
            for idx, r in enumerate(results)
        ]
        return QueryResult(answer=answer, sources=sources)