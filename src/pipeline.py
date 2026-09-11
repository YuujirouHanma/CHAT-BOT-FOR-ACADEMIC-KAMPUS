"""End-to-end pipeline orchestration.

1. INDEXING: parse → enrich → chunk → embed → store
2. QUERYING: retrieve → format context → generate
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src import evaluation as evaluation_mod
from src import gen_cache, guided, learning_styles
from src import livecode as livecode_mod
from src.catalog import display_name, resolve_course_week
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
from src.tenancy import require_tenant_id
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
    weeks: list[int] = field(default_factory=list)
    content_id: str | None = None
    source_file: str | None = None
    style: str | None = None
    topic: str | None = None


def _umpan_balik_kode(nilai: dict) -> str:
    """Rangkum hasil penilaian kode menjadi satu teks umpan balik.

    Jalur kode menghasilkan temuan statis, catatan keliru, dan petunjuk secara
    terpisah, sementara satu butir evaluasi hanya punya satu ruang umpan balik.
    Ketiganya digabung supaya mahasiswa yang mengerjakan soal koding lewat
    evaluasi tetap menerima koreksi yang sama rincinya dengan yang lewat
    latihan mandiri.
    """
    bagian: list[str] = []
    ringkas = (nilai.get("ringkasan") or "").strip()
    if ringkas:
        bagian.append(ringkas)
    bagian += [
        f"- {t['message']}" for t in nilai.get("temuan", [])
        if t.get("severity") != "info" and t.get("message")
    ]
    bagian += [
        f"- {k['masalah']}" for k in nilai.get("keliru", []) if k.get("masalah")
    ]
    bagian += [f"Petunjuk: {p}" for p in nilai.get("petunjuk", [])]
    return "\n".join(bagian)


def _course_name(courses: list[dict], course_id: str) -> str:
    """Nama tampilan sebuah course dari daftar terindex; fallback ke slug dirapikan."""
    for c in courses:
        if c.get("course_id") == course_id and c.get("course_name"):
            return c["course_name"]
    return display_name(course_id) or course_id


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
        *,
        tenant_id: str,
    ) -> IndexResult:
        tenant_id = require_tenant_id(tenant_id, operation="index_document")
        logger.info(
            "=== Indexing {} (tenant={}, content_id={}) ===",
            file_path.name, tenant_id, content_id,
        )

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
                "tenant_id": tenant_id,
                "course_id": r_course_id,
                "course_name": r_course_name,
                "week": r_week,
            })
            for c in chunks
        ]

        embedded = await self._embedder.embed_chunks(chunks)
        stored = await self._store.upsert_chunks(embedded, tenant_id=tenant_id)
        # Indeks ulang harus MENGGANTI, bukan menambah. Dibersihkan sesudah
        # penulisan berhasil, supaya kegagalan di tengah tidak menghapus materi.
        await self._store.delete_stale_chunks(
            file_path.name, content_id, [c.chunk_id for c in embedded],
            tenant_id=tenant_id,
        )

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
        course_id: str | None = None,
        weeks: list[int] | None = None,
        style: str | None = None,
        *,
        tenant_id: str,
    ) -> QueryResult:
        tenant_id = require_tenant_id(tenant_id, operation="query")
        logger.info("=== Query: '{}' (tenant={}, content_id={}, model={}, level={}) ===",
                    question[:80], tenant_id, content_id, model or "default",
                    level or "standar")
        start = time.monotonic()

        # Stage 1: decompose the question into an enriched, more searchable query.
        dq = await self._generator.decompose_query(question, model=model)
        enriched_query = dq.get("query_diperkaya") or question

        # Stage 2: retrieve using the enriched query.
        results = await self._retriever.retrieve(
            query=enriched_query, content_id=content_id, source_filter=source_filter,
            course_id=course_id, weeks=weeks, tenant_id=tenant_id,
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
                content_id=content_id, session_id=session_id, tenant_id=tenant_id,
            )
            return result

        context = format_retrieval_results(results)
        answer = await self._generator.generate(
            question, context, model=model, level=level, style=style,
        )

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
            content_id=content_id, session_id=session_id, tenant_id=tenant_id,
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
        weeks: list[int] | None = None,
        content_id: str | None = None,
        source_file: str | None = None,
        style: str | None = None,
        awaiting: str | None = None,
        model: str | None = None,
        tenant_id: str,
        allowed_courses: frozenset[str] | None = None,
    ) -> GuidedTurn:
        """Satu putaran guided navigation: mata kuliah → minggu → materi → pertanyaan.

        Menerima konteks yang sudah terkumpul di session, membaca pesan terbaru,
        lalu memutuskan: menuntun ke langkah berikutnya, atau menyerahkan pesan ini
        ke `query()` karena ternyata sebuah pertanyaan.

        Semua pilihan diambil dari yang benar-benar terindex, jadi mahasiswa tidak
        pernah ditawari minggu atau materi yang kosong.
        """
        tenant_id = require_tenant_id(tenant_id, operation="guided_turn")
        courses = await self._store.list_courses(tenant_id=tenant_id)
        # ABAC: kunci yang dibatasi sebagian mata kuliah tidak boleh DITAWARI
        # sisanya. Menyaring di sini, bukan hanya menolak saat dipilih, membuat
        # chatbot tidak pernah menyebut mata kuliah yang ujungnya akan ditolak.
        if allowed_courses is not None:
            courses = [c for c in courses if c["course_id"] in allowed_courses]

        if guided.wants_reset(question):
            return await self._ask_course(courses)

        cur_course, cur_name = course_id, course_name
        cur_weeks = sorted(set(weeks or []))
        cur_cid, cur_sf = content_id, source_file
        # Gaya belajar boleh diganti kapan saja lewat teks bebas ("ganti gaya",
        # "pakai diagram") — bukan hanya pada langkah pemilihannya.
        cur_style = learning_styles.match(question) or style

        # Fase 0: kalau pesan ini menyebut salah satu materi yang sedang
        # ditawarkan, kenali dulu dan buang namanya dari teks. Tanpa ini nama
        # file seperti "Materi SBD TM9.pptx" terbaca sebagai minggu 9.
        text_for_ctx = question
        picked: dict | None = None
        if cur_course and cur_weeks:
            offered = await self._store.list_materials_for_weeks(
                cur_course, cur_weeks, tenant_id=tenant_id,
            )
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
            cur_weeks = []
            cur_cid = cur_sf = None
            picked = None          # daftar materi tadi sudah tidak berlaku
        if refs.weeks and refs.weeks != cur_weeks:
            cur_weeks = refs.weeks
            cur_cid = cur_sf = None
            picked = None
        # Nama tampilan bisa belum terisi walau course_id sudah ada — misalnya
        # klien mengirim course_id eksplisit tanpa course_name.
        if cur_course and not cur_name:
            cur_name = _course_name(courses, cur_course)

        # Fase 2: tetapkan materi.
        materials: list[dict] = []
        if cur_course and cur_weeks:
            materials = await self._store.list_materials_for_weeks(
                cur_course, cur_weeks, tenant_id=tenant_id,
            )
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
            "course_id": cur_course, "course_name": cur_name, "weeks": cur_weeks,
            "content_id": cur_cid, "source_file": cur_sf, "style": cur_style,
        }

        # Pertanyaan sungguhan selalu dijawab — mahasiswa tetap bisa memakai ini
        # seperti chatbot biasa, dengan atau tanpa memilih materi lebih dulu.
        # Klik tombol gaya belajar bukan pertanyaan. Diperiksa LEBIH DULU karena
        # dua labelnya ("Lewat contoh & kode …", "Poin-poin ringkas …") panjang
        # dan berisi banyak kata isi, sehingga lolos sebagai pertanyaan lalu
        # dijawab sungguhan — mahasiswa menunggu dua menit untuk jawaban atas
        # teks tombol yang baru saja ia tekan.
        memilih_gaya = learning_styles.is_choice_label(question) or (
            awaiting == guided.STEP_STYLE and learning_styles.match(question) is not None
        )
        if guided.looks_like_question(question) and not memilih_gaya:
            return GuidedTurn(step=guided.STEP_ANSWER, answer_question=True, **ctx)

        step = guided.next_step(
            course_id=cur_course, weeks=cur_weeks, source_file=cur_sf,
            style=cur_style,
        )
        logger.info(
            "Guided | step={} course={} minggu={} materi={}",
            step, cur_course, cur_weeks, cur_sf,
        )

        if step == guided.STEP_COURSE:
            return await self._ask_course(courses)

        if step == guided.STEP_WEEK:
            tersedia = await self._store.list_weeks(cur_course or "", tenant_id=tenant_id)
            if not tersedia:
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
                    for w in tersedia
                ],
                **ctx,
            )

        if step == guided.STEP_MATERIAL:
            if not materials:
                tersedia = await self._store.list_weeks(
                    cur_course or "", tenant_id=tenant_id,
                )
                return GuidedTurn(
                    step=guided.STEP_WEEK,
                    message=guided.empty_message(
                        guided.STEP_MATERIAL, course_name=cur_name, weeks=cur_weeks,
                    ),
                    choices=[
                        guided.Choice(label=f"Minggu {w}", value=str(w), kind="week")
                        for w in tersedia
                    ],
                    **{**ctx, "weeks": []},
                )
            topik = await self.week_topic(
                cur_course or "", cur_weeks, model=model, tenant_id=tenant_id,
            )
            return GuidedTurn(
                step=step,
                message=guided.prompt_for(
                    step, course_name=cur_name, weeks=cur_weeks, topic=topik,
                ),
                topic=topik or None,
                choices=[
                    guided.Choice(
                        # Saat lebih dari satu minggu dipilih, nama berkas saja
                        # ambigu — minggunya ikut ditampilkan.
                        label=(f"Minggu {m['week']} — {m['source_file']}"
                               if len(cur_weeks) > 1 and m.get("week") is not None
                               else m["source_file"]),
                        value=m["source_file"], kind="material",
                    )
                    for m in materials
                ],
                **ctx,
            )

        if step == guided.STEP_STYLE:
            return GuidedTurn(
                step=step,
                message=guided.prompt_for(step, source_file=cur_sf),
                choices=[
                    guided.Choice(
                        label=learning_styles.choice_label(s), value=s.key, kind="style",
                    )
                    for s in learning_styles.all_styles()
                ],
                **ctx,
            )

        # STEP_QUESTION — konteks lengkap, tawarkan pertanyaan template materi ini.
        spec = learning_styles.resolve(cur_style)
        starters: list[str] = []
        if cur_cid and cur_sf:
            starters = await self.starter_questions(
                cur_cid, cur_sf, model=model, style=cur_style, tenant_id=tenant_id,
            )
        return GuidedTurn(
            step=guided.STEP_QUESTION,
            message=spec.greeting + "\n\n" + guided.prompt_for(
                guided.STEP_QUESTION, source_file=cur_sf,
            ),
            choices=[
                guided.Choice(label=q, value=q, kind="question") for q in starters
            ],
            **ctx,
        )

    async def week_topic(
        self, course_id: str, weeks: list[int], model: str | None = None,
        *, tenant_id: str,
    ) -> str:
        """Satu kalimat topik yang dibahas pada minggu-minggu tertentu.

        Disimpulkan dari isi materi yang terindex, bukan dari nama berkas — nama
        seperti "Materi SBD TM9.pptx" tidak memberi tahu mahasiswa apa pun.
        Di-cache karena isinya hanya berubah bila materi minggu itu berubah.
        """
        tenant_id = require_tenant_id(tenant_id, operation="week_topic")
        if not (course_id and weeks):
            return ""
        kunci = f"{course_id}::{'-'.join(str(w) for w in sorted(set(weeks)))}"
        # Teks diambil lebih dulu meski mungkin berujung cache hit: sidik
        # jarinya ikut menentukan kunci, sehingga entri lama berhenti terpakai
        # begitu materi atau cara perakitannya berubah.
        text = await self._store.get_week_text(course_id, weeks, tenant_id=tenant_id)
        sidik = gen_cache.hash_material(text)
        cached = gen_cache.load(
            "week_topic", kunci, "topic", tenant_id=tenant_id, material_hash=sidik,
        )
        if cached is not None:
            return cached[0] if isinstance(cached, list) and cached else ""

        topik = await self._generator.summarize_week_topic(text, model=model)
        if topik:
            gen_cache.save(
                "week_topic", kunci, "topic", [topik],
                tenant_id=tenant_id, material_hash=sidik,
            )
        return topik

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
        style: str | None = None, *, tenant_id: str,
    ) -> list[str]:
        """Pertanyaan pembuka untuk sebuah materi, disesuaikan gaya belajar.

        Cache dipisah per gaya: pertanyaan untuk gaya "visual" (soal alur dan
        hubungan) tidak cocok dipakai ulang untuk gaya "praktik" (soal cara dan
        penerapan), jadi keduanya tidak boleh berbagi kunci cache.
        """
        tenant_id = require_tenant_id(tenant_id, operation="starter_questions")
        spec = learning_styles.resolve(style)
        kunci = f"{source_file}::{spec.key}"
        text = await self._store.get_material_text(
            content_id, source_file, tenant_id=tenant_id,
        )
        sidik = gen_cache.hash_material(text)
        cached = gen_cache.load(
            "starter", content_id, kunci, tenant_id=tenant_id, material_hash=sidik,
        )
        if cached is not None:
            return cached

        questions = await self._generator.generate_starter_questions(
            text, model=model, style_hint=spec.starter_hint,
        )
        if questions:
            gen_cache.save(
                "starter", content_id, kunci, questions,
                tenant_id=tenant_id, material_hash=sidik,
            )
        return questions

    async def quiz(
        self, content_id: str, source_file: str, model: str | None = None,
        *, tenant_id: str,
    ) -> list[dict]:
        """Multiple-choice quiz for a material. Cached after first generation."""
        tenant_id = require_tenant_id(tenant_id, operation="quiz")
        text = await self._store.get_material_text(
            content_id, source_file, tenant_id=tenant_id,
        )
        sidik = gen_cache.hash_material(text)
        cached = gen_cache.load(
            "quiz", content_id, source_file, tenant_id=tenant_id, material_hash=sidik,
        )
        if cached is not None:
            return cached

        quiz = await self._generator.generate_quiz(text, model=model)
        if quiz:
            gen_cache.save(
                "quiz", content_id, source_file, quiz,
                tenant_id=tenant_id, material_hash=sidik,
            )
        return quiz

    async def evaluation(
        self,
        course_id: str,
        plan: evaluation_mod.EvaluationPlan,
        *,
        tenant_id: str,
        model: str | None = None,
    ) -> list[dict]:
        """Soal evaluasi atas rentang minggu sebuah mata kuliah.

        Di-cache per (tenant, mata kuliah, rentang minggu, komposisi soal).
        Cache di sini bukan sekadar penghematan biaya: satu evaluasi harus SAMA
        untuk seluruh mahasiswa di satu kelas. Tanpa cache, setiap orang
        mengerjakan soal yang berbeda dan nilainya tidak dapat dibandingkan —
        yang mematikan seluruh gunanya sebagai alat ukur penelitian.
        """
        require_tenant_id(tenant_id, operation="evaluation")
        if not course_id:
            return []

        kunci = (
            f"{plan.kind}::{'-'.join(str(w) for w in plan.weeks_covered)}"
            f"::{'-'.join(f'{t}{n}' for t, n in sorted(plan.blueprint.counts.items()))}"
        )
        teks = await self._store.get_week_text(
            course_id, list(plan.weeks_covered), max_chars=12_000,
            tenant_id=tenant_id,
        )
        if not teks:
            logger.warning(
                "Evaluasi {} untuk {} {} dibatalkan: tidak ada materi terindeks",
                plan.kind, course_id, plan.range_text,
            )
            return []

        sidik = gen_cache.hash_material(teks)
        cached = gen_cache.load(
            "evaluation", course_id, kunci, tenant_id=tenant_id, material_hash=sidik,
        )
        if cached is not None:
            return cached

        soal = await self._generator.generate_evaluation(
            teks, plan.label, plan.range_text, plan.blueprint.counts, model=model,
        )
        if soal:
            gen_cache.save(
                "evaluation", course_id, kunci, soal,
                tenant_id=tenant_id, material_hash=sidik,
            )
        return soal

    async def grade_evaluation(
        self,
        items: list[dict],
        answers: list,
        *,
        tenant_id: str,
        model: str | None = None,
    ) -> dict:
        """Nilai satu evaluasi bercampur jenis soal.

        Soal objektif dinilai mesin — pasti, seketika, tanpa biaya. Soal
        `koding` dinilai lewat jalur kode (analisis statis + tinjauan kode),
        bukan lewat pemeriksa prosa: kode yang sama harus mendapat satu standar
        penilaian, dari endpoint mana pun ia dikirim. Sisanya dinilai LLM
        terhadap kunci/rubrik. Butir yang LLM ragu atau gagal dinilai ditandai
        `perlu_tinjauan`, bukan diberi nol diam-diam.
        """
        require_tenant_id(tenant_id, operation="grade_evaluation")
        hasil: list[evaluation_mod.GradedItem] = []

        for i, butir in enumerate(items):
            jawaban = answers[i] if i < len(answers) else None

            if not evaluation_mod.needs_llm_grading(butir):
                hasil.append(evaluation_mod.grade_objective(butir, jawaban, i))
                continue

            if butir.get("type") == "koding":
                hasil.append(
                    await self._grade_koding_item(butir, jawaban, i, model=model),
                )
                continue

            nilai = await self._generator.grade_open_answer(
                question=butir.get("question", ""),
                student_answer=str(jawaban or ""),
                expected=butir.get("expected_answer", "")
                         or butir.get("expected_behavior", ""),
                rubric=butir.get("rubric") or butir.get("key_points") or [],
                model=model,
            )
            ragu = nilai is None or nilai.get("ragu", False)
            hasil.append(evaluation_mod.GradedItem(
                index=i,
                type=butir.get("type", "esai"),
                question=butir.get("question", ""),
                student_answer=jawaban,
                correct_answer=butir.get("expected_answer")
                               or butir.get("expected_behavior") or "",
                # None = belum dapat dipastikan; dosen yang memutuskan.
                is_correct=None if ragu else bool((nilai or {}).get("benar")),
                score=float((nilai or {}).get("skor", 0.0)),
                explanation=butir.get("explanation", ""),
                feedback=(nilai or {}).get(
                    "feedback", "Perlu ditinjau dosen — penilaian otomatis gagal.",
                ),
            ))

        ringkas = evaluation_mod.summarize(hasil)
        ringkas["butir"] = [
            {
                "index": g.index, "type": g.type, "question": g.question,
                "your_answer": g.student_answer, "correct_answer": g.correct_answer,
                "is_correct": g.is_correct, "score": round(g.score, 2),
                "explanation": g.explanation, "feedback": g.feedback,
            }
            for g in hasil
        ]
        return ringkas

    async def _grade_koding_item(
        self,
        item: dict,
        answer: object,
        index: int,
        *,
        model: str | None = None,
    ) -> evaluation_mod.GradedItem:
        """Nilai satu butir `koding` evaluasi lewat jalur penilaian kode.

        Butirnya diubah dulu menjadi latihan livecode supaya yang dinilai
        benar-benar objek yang sama dengan latihan mandiri — termasuk ketika
        soalnya tidak menyertakan `test_cases` maupun `required_function`.
        Spesifikasi yang kosong itu keadaan yang sah, bukan galat: analisis
        statis melewatkan pemeriksaan yang tidak diminta, dan tinjauan kode
        tetap berjalan dengan berbekal perintah soal.
        """
        latihan = livecode_mod.exercise_from_item(item, index=index)
        kode = answer if isinstance(answer, str) else ""
        nilai = await self._review_code(latihan, kode, model=model)
        return evaluation_mod.GradedItem(
            index=index,
            type="koding",
            question=item.get("question", ""),
            student_answer=answer,
            correct_answer=item.get("expected_behavior", ""),
            # None = belum dapat dipastikan; dosen yang memutuskan.
            is_correct=None if nilai["perlu_tinjauan_dosen"] else nilai["lulus"],
            score=nilai["skor"],
            explanation=item.get("explanation", ""),
            feedback=_umpan_balik_kode(nilai),
        )

    async def livecode_exercises(
        self,
        course_id: str,
        weeks: list[int],
        *,
        tenant_id: str,
        count: int = 3,
        model: str | None = None,
    ) -> list[livecode_mod.Exercise]:
        """Latihan koding yang diturunkan dari materi minggu tertentu.

        Memakai jalur pembuatan soal yang sama dengan evaluasi, dengan komposisi
        khusus koding. Menyatukannya disengaja: soal koding di ETS dan latihan
        mandiri adalah hal yang sama bagi mahasiswa, dan membuatnya lewat dua
        jalur berbeda akan melahirkan dua standar soal yang berbeda pula.
        """
        require_tenant_id(tenant_id, operation="livecode_exercises")
        if not (course_id and weeks):
            return []

        plan = evaluation_mod.custom_plan(
            weeks, "kuis", {"koding": max(1, min(count, 10))},
        )
        butir = await self.evaluation(
            course_id, plan, tenant_id=tenant_id, model=model,
        )
        return [
            livecode_mod.exercise_from_item(
                b, course_id=course_id, week=plan.weeks_covered[-1], index=i,
            )
            for i, b in enumerate(butir)
            if b.get("type") == "koding"
        ]

    async def grade_livecode(
        self,
        exercise: livecode_mod.Exercise,
        code: str,
        *,
        tenant_id: str,
        student_id: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Nilai satu kiriman kode dan catat kirimannya.

        Penilaiannya sendiri ada di `_review_code`, yang dipakai bersama butir
        `koding` pada evaluasi. Yang khusus di sini hanya pencatatan kiriman:
        riwayat percobaan itu milik latihan mandiri, bukan bagian dari menilai.
        """
        require_tenant_id(tenant_id, operation="grade_livecode")

        hasil = await self._review_code(exercise, code, model=model)
        hasil["submission_id"] = livecode_mod.record_submission(
            tenant_id=tenant_id, exercise_id=exercise.exercise_id,
            student_id=student_id, code=code, result=hasil, session_id=session_id,
        )
        return hasil

    async def _review_code(
        self,
        exercise: livecode_mod.Exercise,
        code: str,
        *,
        model: str | None = None,
    ) -> dict:
        """Jalur penilaian kode: analisis statis dulu, LLM hanya bila perlu.

        Urutannya menentukan. Galat sintaks dan fungsi yang belum didefinisikan
        sudah pasti tanpa perlu penalaran apa pun — memanggil LLM untuk itu
        membuang biaya dan waktu tunggu mahasiswa, sekaligus berisiko
        menghasilkan diagnosis yang lebih kabur daripada pesan galat Python
        yang sudah menyebut nomor barisnya.

        Satu-satunya jalur menilai kode di seluruh sistem: /livecode/submit dan
        butir `koding` pada /evaluation/submit sama-sama lewat sini, supaya kode
        yang sama tidak dinilai dengan dua standar yang berbeda.
        """
        temuan, layak_llm = livecode_mod.analyze_code(code, exercise)
        memblokir = livecode_mod.has_blocking_error(temuan)

        tinjauan: dict | None = None
        if layak_llm and not memblokir:
            tinjauan = await self._generator.review_code(
                prompt=exercise.prompt,
                code=code,
                expected_behavior=exercise.expected_behavior,
                rubric=list(exercise.rubric),
                test_cases=list(exercise.test_cases),
                model=model,
            )

        ragu = (not memblokir) and (tinjauan is None or tinjauan.get("ragu", False))
        hasil = {
            "exercise_id": exercise.exercise_id,
            # Lulus hanya bila tidak ada galat statis DAN tinjauan menyatakan lulus.
            "lulus": bool(tinjauan and tinjauan.get("lulus")) and not memblokir,
            "skor": 0.0 if memblokir else float((tinjauan or {}).get("skor", 0.0)),
            "temuan": [t.as_dict() for t in temuan],
            "ringkasan": (
                "Kode belum bisa dinilai — perbaiki dulu yang ditandai di atas."
                if memblokir
                else (tinjauan or {}).get(
                    "ringkasan", "Belum dapat ditinjau otomatis.",
                )
            ),
            "benar": (tinjauan or {}).get("benar", []),
            "keliru": (tinjauan or {}).get("keliru", []),
            "petunjuk": (tinjauan or {}).get("petunjuk", []),
            "perlu_tinjauan_dosen": ragu,
        }
        return hasil

    async def grade_quiz(
        self,
        content_id: str,
        source_file: str,
        answers: list[int],
        session_id: str | None = None,
        student_id: str | None = None,
        *,
        tenant_id: str,
    ) -> dict:
        """Grade submitted answers against the material's quiz and log the attempt.

        Grades against the SAME (cached) quiz the student received. Raises
        ValueError if no quiz exists for the material.
        """
        tenant_id = require_tenant_id(tenant_id, operation="grade_quiz")
        quiz = await self.quiz(content_id, source_file, tenant_id=tenant_id)
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
            session_id=session_id, student_id=student_id, tenant_id=tenant_id,
        )
        return {
            "total": total,
            "correct": correct,
            "score": score,
            "attempt_id": attempt_id,
            "results": results,
        }
