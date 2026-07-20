"""End-to-end pipeline orchestration.

1. INDEXING: parse → enrich → chunk → embed → store
2. QUERYING: retrieve → format context → generate
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src import gen_cache
from src.catalog import resolve_course_week
from src.generation.llm import LLMGenerator
from src.generation.prompts import format_retrieval_results
from src.hitl.logger import log_interaction, log_quiz_attempt
from src.indexing.chunker import Chunker
from src.indexing.embedder import Embedder
from src.indexing.summarizer import MultimodalSummarizer, enrich_elements
from src.ingestion.parser import parse_document
from src.ingestion.transcriber import is_media, parse_media
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
        self,
        file_path: Path,
        content_id: str | None = None,
        course_id: str | None = None,
        course_name: str | None = None,
        week: int | None = None,
    ) -> IndexResult:
        logger.info("=== Indexing {} (content_id={}) ===", file_path.name, content_id)

        # Video/audio go through transcription; everything else through the parser.
        if is_media(file_path):
            elements = await parse_media(file_path, content_id=content_id)
        else:
            elements = parse_document(file_path, content_id=content_id)
        if not elements:
            return IndexResult(file_path.name, 0, 0, 0, content_id=content_id)

        enriched = await enrich_elements(elements, self._summarizer)
        chunks = self._chunker.chunk(enriched)
        if not chunks:
            return IndexResult(file_path.name, len(elements), 0, 0, content_id=content_id)

        # Stamp catalog hierarchy (mata kuliah → minggu) onto every chunk:
        # explicit args win, otherwise derive from content_id.
        r_course_id, r_course_name, r_week = resolve_course_week(
            content_id, course_id=course_id, course_name=course_name, week=week,
        )
        chunks = [
            c.model_copy(update={
                "course_id": r_course_id,
                "course_name": r_course_name,
                "week": r_week,
            })
            for c in chunks
        ]

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
        model: str | None = None,
        level: str | None = None,
    ) -> QueryResult:
        logger.info("=== Query: '{}' (content_id={}, model={}, level={}) ===",
                    question[:80], content_id, model or "default", level or "standar")
        start = time.monotonic()

        # Stage 1: decompose the question into an enriched, more searchable query.
        dq = await self._generator.decompose_query(question, model=model)
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
        answer = await self._generator.generate(question, context, model=model, level=level)

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
        recommendations = await self._generator.generate_followup(question, dq, answer, model=model)

        interaction_id = log_interaction(
            question=question, dq=dq, answer=answer, sources=sources,
            recommendations=recommendations, elapsed_seconds=time.monotonic() - start,
            content_id=content_id, session_id=session_id,
        )
        return QueryResult(
            answer=answer, sources=sources, recommendations=recommendations,
            interaction_id=interaction_id, decomposition=dq,
        )

    async def starter_questions(
        self, content_id: str, source_file: str, model: str | None = None,
    ) -> list[str]:
        """Template opener questions for a material. Cached after first generation."""
        cached = gen_cache.load("starter", content_id, source_file)
        if cached is not None:
            return cached

        text = await self._store.get_material_text(content_id, source_file)
        questions = await self._generator.generate_starter_questions(text, model=model)
        if questions:
            gen_cache.save("starter", content_id, source_file, questions)
        return questions

    async def quiz(
        self, content_id: str, source_file: str, model: str | None = None,
    ) -> list[dict]:
        """Multiple-choice quiz for a material. Cached after first generation."""
        cached = gen_cache.load("quiz", content_id, source_file)
        if cached is not None:
            return cached

        text = await self._store.get_material_text(content_id, source_file)
        quiz = await self._generator.generate_quiz(text, model=model)
        if quiz:
            gen_cache.save("quiz", content_id, source_file, quiz)
        return quiz

    async def grade_quiz(
        self,
        content_id: str,
        source_file: str,
        answers: list[int],
        session_id: str | None = None,
        student_id: str | None = None,
    ) -> dict:
        """Grade submitted answers against the material's quiz and log the attempt.

        Grades against the SAME (cached) quiz the student received. Raises
        ValueError if no quiz exists for the material.
        """
        quiz = await self.quiz(content_id, source_file)
        if not quiz:
            raise ValueError("Kuis tidak tersedia untuk materi ini")

        results: list[dict] = []
        correct = 0
        for idx, q in enumerate(quiz):
            chosen = answers[idx] if idx < len(answers) else None
            is_correct = chosen == q["answer_index"]
            if is_correct:
                correct += 1
            results.append({
                "question": q["question"],
                "options": q["options"],
                "your_answer": chosen,
                "correct_answer": q["answer_index"],
                "is_correct": is_correct,
                "explanation": q["explanation"],
            })

        total = len(quiz)
        score = round(100.0 * correct / total, 1) if total else 0.0
        attempt_id = log_quiz_attempt(
            content_id=content_id, source_file=source_file,
            correct=correct, total=total, score=score,
            session_id=session_id, student_id=student_id,
        )
        return {
            "total": total,
            "correct": correct,
            "score": score,
            "attempt_id": attempt_id,
            "results": results,
        }
