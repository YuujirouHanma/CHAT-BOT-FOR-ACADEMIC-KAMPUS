"""End-to-end pipeline orchestration.

1. INDEXING: parse → enrich → chunk → embed → store
2. QUERYING: retrieve → format context → generate
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src import gen_cache, guided
from src.catalog import humanize_course, resolve_course_week
from src.config import settings
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


@dataclass
class GuidedTurn:
    """Hasil satu putaran guided navigation.

    `answer_question=True` berarti pesan mahasiswa adalah pertanyaan sungguhan —
    pemanggil harus lanjut ke `query()` memakai konteks di dataclass ini. Kalau
    False, `message` + `choices` adalah tanya-balik yang harus ditampilkan.
    """
    step: guided.StepName
    message: str = ""
    choices: list[guided.Choice] = field(default_factory=list)
    answer_question: bool = False
    course_id: str | None = None
    course_name: str | None = None
    week: int | None = None
    content_id: str | None = None
    source_file: str | None = None


def _course_name(courses: list[dict], course_id: str) -> str:
    """Nama tampilan sebuah course dari daftar terindex; fallback ke slug dirapikan."""
    for c in courses:
        if c.get("course_id") == course_id and c.get("course_name"):
            return c["course_name"]
    return humanize_course(course_id) or course_id


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
        if settings.warmup_models:
            await self.warmup()

    async def warmup(self) -> None:
        """Muat model embedding & reranker sekarang, bukan saat permintaan pertama.

        Keduanya dimuat malas (*lazy*): tanpa pemanasan ini, mahasiswa pertama yang
        bertanya ikut menanggung waktu unduh dan muat model ±2 GB. Kegagalan di sini
        sengaja tidak menghentikan server — pemanasan hanyalah optimasi, dan
        pemuatan malas tetap menjadi jaring pengaman.
        """
        start = time.monotonic()
        try:
            await self._embedder.embed_query("pemanasan model")
            await self._reranker.rerank(
                query="pemanasan model",
                candidates=[{"payload": {"text": "pemanasan model"}}],
                top_k=1,
            )
            logger.info("Model siap dipakai ({:.1f}s)", time.monotonic() - start)
        except Exception as exc:
            logger.warning(
                "Pemanasan model gagal, akan dimuat saat permintaan pertama: {}", exc
            )

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
        history: list[dict] | None = None,
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
        recommendations = await self._generator.generate_followup(
            question, dq, answer, model=model, history=history,
        )

        interaction_id = log_interaction(
            question=question, dq=dq, answer=answer, sources=sources,
            recommendations=recommendations, elapsed_seconds=time.monotonic() - start,
            content_id=content_id, session_id=session_id,
        )
        return QueryResult(
            answer=answer, sources=sources, recommendations=recommendations,
            interaction_id=interaction_id, decomposition=dq,
        )

    async def guided_turn(
        self,
        question: str,
        *,
        course_id: str | None = None,
        course_name: str | None = None,
        week: int | None = None,
        content_id: str | None = None,
        source_file: str | None = None,
        awaiting: str | None = None,
        model: str | None = None,
    ) -> GuidedTurn:
        """Satu putaran guided navigation: mata kuliah → minggu → materi → pertanyaan.

        Menerima konteks yang sudah terkumpul di session, membaca pesan terbaru,
        lalu memutuskan: menuntun ke langkah berikutnya, atau menyerahkan pesan ini
        ke `query()` karena ternyata sebuah pertanyaan.

        Semua pilihan diambil dari yang benar-benar terindex, jadi mahasiswa tidak
        pernah ditawari minggu atau materi yang kosong.
        """
        courses = await self._store.list_courses()

        if guided.wants_reset(question):
            return await self._ask_course(courses)

        cur_course, cur_name = course_id, course_name
        cur_week, cur_cid, cur_sf = week, content_id, source_file

        # Fase 0: kalau pesan ini menyebut salah satu materi yang sedang
        # ditawarkan, kenali dulu dan buang namanya dari teks. Tanpa ini nama
        # file seperti "Materi SBD TM9.pptx" terbaca sebagai minggu 9.
        text_for_ctx = question
        picked: dict | None = None
        if cur_course and cur_week is not None:
            offered = await self._store.list_materials(cur_course, cur_week)
            picked = guided.match_material(
                question, offered,
                allow_ordinal=(awaiting == guided.STEP_MATERIAL),
            )
            if picked:
                text_for_ctx = guided.strip_phrase(
                    question, picked.get("source_file") or "",
                )

        # Fase 1: kenali mata kuliah & minggu. Materi belum ditetapkan karena
        # daftarnya bergantung pada mata kuliah+minggu yang mungkin baru berubah.
        refs = guided.resolve_refs(text_for_ctx, courses, awaiting=awaiting)

        if refs.course_id and refs.course_id != cur_course:
            cur_course = refs.course_id
            cur_name = None
            cur_week = cur_cid = cur_sf = None
            picked = None          # daftar materi tadi sudah tidak berlaku
        if refs.week is not None and refs.week != cur_week:
            cur_week = refs.week
            cur_cid = cur_sf = None
            picked = None
        # Nama tampilan bisa belum terisi walau course_id sudah ada — misalnya
        # klien mengirim course_id eksplisit tanpa course_name.
        if cur_course and not cur_name:
            cur_name = _course_name(courses, cur_course)

        # Fase 2: tetapkan materi.
        materials: list[dict] = []
        if cur_course and cur_week is not None:
            materials = await self._store.list_materials(cur_course, cur_week)
            mat = picked or guided.match_material(
                question, materials,
                allow_ordinal=(awaiting == guided.STEP_MATERIAL),
            )
            if mat:
                cur_sf = mat.get("source_file")
                cur_cid = mat.get("content_id") or cur_cid
            elif cur_sf and not cur_cid:
                # Klien boleh mengirim source_filter saja; lengkapi content_id-nya
                # dari katalog supaya starter questions & kuis tetap bisa jalan.
                for m in materials:
                    if m.get("source_file") == cur_sf:
                        cur_cid = m.get("content_id")
                        break

        ctx = {
            "course_id": cur_course, "course_name": cur_name, "week": cur_week,
            "content_id": cur_cid, "source_file": cur_sf,
        }

        # Pertanyaan sungguhan selalu dijawab — mahasiswa tetap bisa memakai ini
        # seperti chatbot biasa, dengan atau tanpa memilih materi lebih dulu.
        if guided.looks_like_question(question):
            return GuidedTurn(step=guided.STEP_ANSWER, answer_question=True, **ctx)

        step = guided.next_step(
            course_id=cur_course, week=cur_week, source_file=cur_sf,
        )
        logger.info(
            "Guided | step={} course={} week={} materi={}",
            step, cur_course, cur_week, cur_sf,
        )

        if step == guided.STEP_COURSE:
            return await self._ask_course(courses)

        if step == guided.STEP_WEEK:
            weeks = await self._store.list_weeks(cur_course or "")
            if not weeks:
                turn = await self._ask_course(courses)
                turn.message = guided.empty_message(
                    guided.STEP_WEEK, course_name=cur_name,
                ) + "\n\n" + turn.message
                return turn
            return GuidedTurn(
                step=step,
                message=guided.prompt_for(step, course_name=cur_name),
                choices=[
                    guided.Choice(label=f"Minggu {w}", value=str(w), kind="week")
                    for w in weeks
                ],
                **ctx,
            )

        if step == guided.STEP_MATERIAL:
            if not materials:
                weeks = await self._store.list_weeks(cur_course or "")
                return GuidedTurn(
                    step=guided.STEP_WEEK,
                    message=guided.empty_message(
                        guided.STEP_MATERIAL, course_name=cur_name, week=cur_week,
                    ),
                    choices=[
                        guided.Choice(label=f"Minggu {w}", value=str(w), kind="week")
                        for w in weeks
                    ],
                    **{**ctx, "week": None},
                )
            return GuidedTurn(
                step=step,
                message=guided.prompt_for(step, course_name=cur_name, week=cur_week),
                choices=[
                    guided.Choice(
                        label=m["source_file"], value=m["source_file"], kind="material",
                    )
                    for m in materials
                ],
                **ctx,
            )

        # STEP_QUESTION — konteks lengkap, tawarkan pertanyaan template materi ini.
        starters: list[str] = []
        if cur_cid and cur_sf:
            starters = await self.starter_questions(cur_cid, cur_sf, model=model)
        return GuidedTurn(
            step=guided.STEP_QUESTION,
            message=guided.prompt_for(guided.STEP_QUESTION, source_file=cur_sf),
            choices=[
                guided.Choice(label=q, value=q, kind="question") for q in starters
            ],
            **ctx,
        )

    async def _ask_course(self, courses: list[dict]) -> GuidedTurn:
        """Langkah pertama: tawarkan daftar mata kuliah (atau beri tahu kalau kosong)."""
        if not courses:
            return GuidedTurn(
                step=guided.STEP_COURSE,
                message=guided.empty_message(guided.STEP_COURSE),
            )
        return GuidedTurn(
            step=guided.STEP_COURSE,
            message=guided.prompt_for(guided.STEP_COURSE),
            choices=[
                guided.Choice(
                    label=c.get("course_name") or c["course_id"],
                    value=c["course_id"],
                    kind="course",
                )
                for c in courses
            ],
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
