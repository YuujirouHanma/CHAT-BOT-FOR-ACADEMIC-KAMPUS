"""Generation LLM wrapper — multi-provider support.

Supports: OpenAI, Groq, HuggingFace Inference API, Ollama/vLLM.
All use the OpenAI-compatible API format. Switch provider by changing
GENERATION_PROVIDER + GENERATION_MODEL + GENERATION_BASE_URL in .env.
"""
from __future__ import annotations

from typing import Any

from openai import APIConnectionError as OAIConnError
from openai import APITimeoutError as OAITimeoutError
from openai import AsyncOpenAI
from openai import InternalServerError as OAIServerError
from openai import RateLimitError as OAIRateLimit
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import model_registry
from src.config import settings
from src.generation.prompts import (
    CODE_REVIEW_SYSTEM_PROMPT,
    DECOMPOSE_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    GRADE_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    STARTER_SYSTEM_PROMPT,
    WEEK_TOPIC_SYSTEM_PROMPT,
    FormattedContext,
    build_evaluation_prompt,
    build_system_prompt,
    build_user_prompt,
    format_history,
    parse_code_review_json,
    parse_decompose_json,
    parse_evaluation_json,
    parse_followup_json,
    parse_grade_json,
    parse_quiz_json,
)
from src.utils.logger import logger

_RETRYABLE = (OAIConnError, OAITimeoutError, OAIServerError, OAIRateLimit)

# Batas token untuk panggilan penunjang (decompose, follow-up, starter questions).
# Keluarannya sendiri pendek — sebuah JSON kecil — tetapi model reasoning seperti
# Qwen 3.7 menghabiskan token untuk penalaran SEBELUM menulis jawaban, dan token
# itu ikut dihitung terhadap max_tokens. Bila jatahnya habis saat masih menalar,
# model mengembalikan konten KOSONG dan fitur gagal diam-diam.
#
# Terukur pada Qwen 3.7 Flash: penalaran 650-1.400 token untuk tugas sekecil ini,
# sementara isi JSON-nya hanya ±100-400 token. Batas 256 dan 1.024 sama-sama
# terbukti tidak cukup — pada 1.024 sempat lolos mepet (penalaran 921, sisa 105)
# lalu gagal begitu penalaran menembus 1.024. Prompt follow-up kini juga memuat
# riwayat percakapan sehingga penalarannya lebih panjang lagi.
# Nilai ini batas atas, bukan target: token yang tidak terpakai tidak ditagih.
_AUX_MAX_TOKENS = 3072

# Kuis butuh jauh lebih besar: keluarannya 5 soal × 4 opsi × penjelasan (±1.000
# token) DITAMBAH token penalaran. Dengan batas lama 1800, penalaran menghabiskan
# jatah sebelum JSON-nya sempat ditulis sehingga model mengembalikan konten kosong
# dan pembuatan kuis selalu gagal pada model reasoning.
_QUIZ_MAX_TOKENS = 4096
_EVAL_MAX_TOKENS = 8192
# Penilaian & tinjauan kode. Terukur pada Qwen 3.7: satu tinjauan kode memakai
# ~2.900 token hanya untuk penalaran sebelum menulis JSON-nya. Dengan batas
# penunjang 3.072, yang tersisa untuk jawabannya tinggal ~150 token — JSON
# terpotong, parse gagal, dan butir itu ditandai "perlu tinjauan dosen" padahal
# kodenya benar. Batas atas, bukan target: sisanya tidak ditagih.
_REVIEW_MAX_TOKENS = 6144


def _usage_summary(response: Any) -> str:
    """Ringkasan pemakaian token (dan biaya bila penyedia mengirimkannya).

    Dicatat di setiap panggilan supaya pemakaian dan biaya dapat dipantau dari
    log — sebelumnya tidak ada cara mengetahui berapa token yang terbakar selain
    membuka dasbor penyedia. `cost` dan `reasoning_tokens` bersifat khusus
    OpenRouter; penyedia lain cukup melaporkan jumlah tokennya saja.
    """
    u = getattr(response, "usage", None)
    if u is None:
        return "usage: n/a"
    bagian = [
        f"in={getattr(u, 'prompt_tokens', '?')}",
        f"out={getattr(u, 'completion_tokens', '?')}",
    ]
    detail = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(detail, "reasoning_tokens", None) if detail else None
    if not reasoning and isinstance(u, dict):  # sebagian SDK mengembalikan dict
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        bagian.append(f"reasoning={reasoning}")
    cost = getattr(u, "cost", None)
    if cost is None and isinstance(u, dict):
        cost = u.get("cost")
    if cost is not None:
        bagian.append(f"cost=${float(cost):.6f}")
    return "usage: " + " ".join(bagian)


def _detect_image_mime(image_base64: str) -> str:
    import base64 as _b64
    try:
        head = _b64.b64decode(image_base64[:24], validate=False)
    except Exception:
        return "image/png"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:4] == b"GIF8":
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


class GenerationError(RuntimeError):
    """Raised when generation fails after retries."""


class LLMGenerator:
    """Multimodal-aware generator with multi-provider support."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        # Default provider client (or an injected one for tests). Extra providers
        # get their client built lazily and cached in _clients when first used.
        self._default_provider = settings.generation_provider
        self._default_client = client or self._build_client(self._default_provider)
        self._clients: dict[str, AsyncOpenAI] = {self._default_provider: self._default_client}

    @staticmethod
    def _build_client(provider: str) -> AsyncOpenAI:
        """Build an OpenAI-compatible client for a provider."""
        logger.info("Building LLM client for provider={}", provider)
        return AsyncOpenAI(
            api_key=settings.api_key_for(provider),
            base_url=settings.base_url_for(provider),
        )

    def _resolve(self, model: str | None) -> tuple[AsyncOpenAI, str]:
        """Map an optional model key to (client, provider model id).

        None/unknown → the default provider+model. A known key builds/reuses a
        client for that model's provider.
        """
        spec = model_registry.get(model)
        if spec is None:
            if model:
                logger.warning("Unknown model '{}', using default", model)
            return self._default_client, settings.generation_model
        client = self._clients.get(spec.provider)
        if client is None:
            client = self._build_client(spec.provider)
            self._clients[spec.provider] = client
        return client, spec.model

    async def generate(
        self,
        question: str,
        context: FormattedContext,
        model: str | None = None,
        level: str | None = None,
        style: str | None = None,
    ) -> str:
        user_prompt = build_user_prompt(question, context)
        user_content = self._build_user_content(user_prompt, context)

        messages: list[Any] = [
            {"role": "system", "content": build_system_prompt(level, style)},
            {"role": "user", "content": user_content},
        ]

        return await self._call_with_retry(messages, model=model)

    async def decompose_query(
        self, question: str, model: str | None = None
    ) -> dict[str, Any]:
        """Stage 1: enrich the raw question with topic/key-concepts before retrieval."""
        messages: list[Any] = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"[PERTANYAAN MAHASISWA]\n{question}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.1, max_tokens=_AUX_MAX_TOKENS
            )
        except Exception as exc:
            logger.warning("Stage 1 decompose failed, falling back to raw question: {}", exc)
            return parse_decompose_json("", question)
        return parse_decompose_json(raw, question)

    async def generate_followup(
        self,
        question: str,
        dq: dict[str, Any],
        answer: str,
        model: str | None = None,
        history: list[dict] | None = None,
    ) -> list[str]:
        """Stage 5: generate follow-up questions as a separate call from the main answer.

        `history` (percakapan sebelumnya di session yang sama) membuat pertanyaan
        lanjutan menyambung alur belajar mahasiswa, bukan mengulang yang sudah
        ditanyakan.
        """
        past = format_history(history)
        prompt = (
            (f"[RIWAYAT PERCAKAPAN]\n{past}\n\n" if past else "")
            + f"[PERTANYAAN AWAL]\n{question}\n\n"
            f"[TOPIK]\n{dq.get('topik_utama', '-')}\n\n"
            f"[KONSEP KUNCI]\n{', '.join(dq.get('konsep_kunci', []))}\n\n"
            f"[JAWABAN FINAL]\n{answer}\n"
        )
        messages: list[Any] = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.5, max_tokens=_AUX_MAX_TOKENS
            )
        except Exception as exc:
            logger.warning("Stage 5 follow-up generation failed: {}", exc)
            return []
        return parse_followup_json(raw)

    async def summarize_week_topic(
        self, material_text: str, model: str | None = None
    ) -> str:
        """Satu kalimat topik yang dibahas pada minggu tertentu.

        Mengembalikan "" bila gagal — pemanggil cukup menyembunyikan barisnya,
        karena topik hanyalah pelengkap, bukan syarat alur belajar berjalan.
        """
        text = (material_text or "").strip()
        if not text:
            return ""
        messages: list[Any] = [
            {"role": "system", "content": WEEK_TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"[ISI MATERI]\n{text}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.2, max_tokens=_AUX_MAX_TOKENS
            )
        except Exception as exc:
            logger.warning("Ringkasan topik minggu gagal: {}", exc)
            return ""
        return raw.strip().strip('"').split("\n")[0][:300]

    async def generate_starter_questions(
        self, material_text: str, model: str | None = None,
        style_hint: str | None = None,
    ) -> list[str]:
        """Generate template opener questions from a material's text.

        Returns [] on failure (caller can fall back to empty). Result is meant
        to be cached by the caller so we don't pay per request.
        """
        text = (material_text or "").strip()
        if not text:
            return []
        # Arahan gaya belajar ikut membentuk pertanyaan pembuka: mahasiswa yang
        # memilih belajar lewat diagram ditawari pertanyaan tentang alur, yang
        # memilih praktik ditawari pertanyaan "bagaimana cara".
        sistem = STARTER_SYSTEM_PROMPT
        if style_hint:
            sistem += f" {style_hint}"
        messages: list[Any] = [
            {"role": "system", "content": sistem},
            {"role": "user", "content": f"[ISI MATERI]\n{text}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.4, max_tokens=_AUX_MAX_TOKENS
            )
        except Exception as exc:
            logger.warning("Starter question generation failed: {}", exc)
            return []
        return parse_followup_json(raw, limit=5)

    async def generate_quiz(
        self, material_text: str, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Generate a multiple-choice quiz from a material's text.

        Returns a list of {question, options[4], answer_index, explanation}.
        Empty on failure. Meant to be cached by the caller.
        """
        text = (material_text or "").strip()
        if not text:
            return []
        messages: list[Any] = [
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
            {"role": "user", "content": f"[ISI MATERI]\n{text}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.3, max_tokens=_QUIZ_MAX_TOKENS
            )
        except Exception as exc:
            logger.warning("Quiz generation failed: {}", exc)
            return []
        return parse_quiz_json(raw)

    async def generate_evaluation(
        self,
        material_text: str,
        kind_label: str,
        range_text: str,
        counts: dict[str, int],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Susun soal evaluasi bercampur jenis atas materi beberapa minggu.

        Mengembalikan daftar butir; kosong bila gagal. Pemanggil yang menyimpan
        hasilnya ke cache — sebuah evaluasi harus SAMA untuk semua mahasiswa di
        satu kelas, dan membuatnya ulang tiap permintaan berarti setiap orang
        mengerjakan soal yang berbeda.
        """
        text = (material_text or "").strip()
        if not text:
            return []
        messages: list[Any] = [
            {"role": "system",
             "content": build_evaluation_prompt(kind_label, range_text, counts)},
            {"role": "user", "content": f"[ISI MATERI {range_text.upper()}]\n{text}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.3, max_tokens=_EVAL_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("Pembuatan evaluasi gagal: {}", exc)
            return []
        return parse_evaluation_json(raw)

    async def grade_open_answer(
        self,
        question: str,
        student_answer: str,
        expected: str = "",
        rubric: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Nilai satu jawaban terbuka terhadap kunci atau rubrik.

        None berarti penilaian tidak dapat dilakukan — panggilan gagal atau
        keluaran tidak terbaca. Pemanggil menandainya perlu tinjauan dosen,
        BUKAN memberinya nilai nol: kegagalan mesin bukan kesalahan mahasiswa,
        dan memberi nol diam-diam adalah kerugian yang tidak terlihat.
        """
        jawaban = (student_answer or "").strip()
        if not jawaban:
            return {"skor": 0.0, "benar": False, "ragu": False,
                    "feedback": "Belum ada jawaban."}

        bagian = [f"[SOAL]\n{question}"]
        if expected:
            bagian.append(f"[KUNCI JAWABAN]\n{expected}")
        if rubric:
            bagian.append("[RUBRIK]\n" + "\n".join(f"- {r}" for r in rubric))
        bagian.append(f"[JAWABAN MAHASISWA]\n{jawaban}")

        messages: list[Any] = [
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(bagian)},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.1, max_tokens=_REVIEW_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("Penilaian jawaban terbuka gagal: {}", exc)
            return None
        return parse_grade_json(raw)

    async def review_code(
        self,
        prompt: str,
        code: str,
        expected_behavior: str = "",
        rubric: list[str] | None = None,
        test_cases: list[dict] | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Tinjau program mahasiswa: benar atau belum, dan di mana kelirunya.

        Kode TIDAK dijalankan di mana pun — model diminta menalar alurnya dengan
        membaca. Kasus uji disertakan sebagai bahan penalaran, bukan dieksekusi.

        None berarti tinjauan gagal; pemanggil menandainya perlu ditinjau dosen.
        """
        if not (code or "").strip():
            return None

        bagian = [f"[SOAL]\n{prompt}"]
        if expected_behavior:
            bagian.append(f"[PERILAKU YANG DIHARAPKAN]\n{expected_behavior}")
        if test_cases:
            contoh = "\n".join(
                f"- masukan: {tc.get('input')!r} -> keluaran: {tc.get('output')!r}"
                for tc in test_cases[:8]
            )
            bagian.append(f"[KASUS UJI YANG HARUS TERPENUHI]\n{contoh}")
        if rubric:
            bagian.append("[RUBRIK]\n" + "\n".join(f"- {r}" for r in rubric))
        bagian.append(f"[KODE MAHASISWA]\n```\n{code}\n```")

        messages: list[Any] = [
            {"role": "system", "content": CODE_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(bagian)},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.1, max_tokens=_REVIEW_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("Tinjauan kode gagal: {}", exc)
            return None
        return parse_code_review_json(raw)

    @staticmethod
    def _build_user_content(
        user_prompt: str, context: FormattedContext
    ) -> list[dict[str, Any]] | str:
        """Build user message. Text-only if no images, multimodal if images present."""
        if not context.image_payloads:
            return user_prompt

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img in context.image_payloads:
            mime = _detect_image_mime(img["image_base64"])
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{img['image_base64']}",
                        "detail": "low",
                    },
                }
            )
        return content

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _call_with_retry(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        client, model_id = self._resolve(model)
        temp = temperature if temperature is not None else settings.generation_temperature
        max_tok = max_tokens if max_tokens is not None else settings.generation_max_tokens
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        except _RETRYABLE:
            raise
        except Exception as exc:
            raise GenerationError(f"LLM call failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise GenerationError("LLM returned empty content")

        logger.info(
            "Generated {} chars via model={} | {}",
            len(content), model_id, _usage_summary(response),
        )
        return content