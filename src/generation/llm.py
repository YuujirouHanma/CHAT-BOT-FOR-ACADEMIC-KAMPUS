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
    DECOMPOSE_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    QUIZ_SYSTEM_PROMPT,
    STARTER_SYSTEM_PROMPT,
    FormattedContext,
    build_system_prompt,
    build_user_prompt,
    parse_decompose_json,
    parse_followup_json,
    parse_quiz_json,
)
from src.utils.logger import logger

_RETRYABLE = (OAIConnError, OAITimeoutError, OAIServerError, OAIRateLimit)


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
    ) -> str:
        user_prompt = build_user_prompt(question, context)
        user_content = self._build_user_content(user_prompt, context)

        messages: list[Any] = [
            {"role": "system", "content": build_system_prompt(level)},
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
                messages, model=model, temperature=0.1, max_tokens=256
            )
        except Exception as exc:
            logger.warning("Stage 1 decompose failed, falling back to raw question: {}", exc)
            return parse_decompose_json("", question)
        return parse_decompose_json(raw, question)

    async def generate_followup(
        self, question: str, dq: dict[str, Any], answer: str, model: str | None = None
    ) -> list[str]:
        """Stage 5: generate follow-up questions as a separate call from the main answer."""
        prompt = (
            f"[PERTANYAAN AWAL]\n{question}\n\n"
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
                messages, model=model, temperature=0.5, max_tokens=256
            )
        except Exception as exc:
            logger.warning("Stage 5 follow-up generation failed: {}", exc)
            return []
        return parse_followup_json(raw)

    async def generate_starter_questions(
        self, material_text: str, model: str | None = None
    ) -> list[str]:
        """Generate template opener questions from a material's text.

        Returns [] on failure (caller can fall back to empty). Result is meant
        to be cached by the caller so we don't pay per request.
        """
        text = (material_text or "").strip()
        if not text:
            return []
        messages: list[Any] = [
            {"role": "system", "content": STARTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"[ISI MATERI]\n{text}"},
        ]
        try:
            raw = await self._call_with_retry(
                messages, model=model, temperature=0.4, max_tokens=384
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
                messages, model=model, temperature=0.3, max_tokens=1800
            )
        except Exception as exc:
            logger.warning("Quiz generation failed: {}", exc)
            return []
        return parse_quiz_json(raw)

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

        logger.info("Generated {} chars via model={}", len(content), model_id)
        return content