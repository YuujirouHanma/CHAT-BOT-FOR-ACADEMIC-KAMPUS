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

from src.config import settings
from src.generation.prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
    FOLLOWUP_SYSTEM_PROMPT,
    STARTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    FormattedContext,
    build_user_prompt,
    parse_decompose_json,
    parse_followup_json,
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
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> AsyncOpenAI:
        """Build OpenAI-compatible client for any provider."""
        logger.info(
            "LLM provider={}, model={}, url={}",
            settings.generation_provider,
            settings.generation_model,
            settings.generation_base_url,
        )
        return AsyncOpenAI(
            api_key=settings.generation_api_key,
            base_url=settings.generation_base_url,
        )

    async def generate(
        self,
        question: str,
        context: FormattedContext,
    ) -> str:
        user_prompt = build_user_prompt(question, context)
        user_content = self._build_user_content(user_prompt, context)

        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await self._call_with_retry(messages)

    async def decompose_query(self, question: str) -> dict[str, Any]:
        """Stage 1: enrich the raw question with topic/key-concepts before retrieval."""
        messages: list[Any] = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"[PERTANYAAN MAHASISWA]\n{question}"},
        ]
        try:
            raw = await self._call_with_retry(messages, temperature=0.1, max_tokens=256)
        except Exception as exc:
            logger.warning("Stage 1 decompose failed, falling back to raw question: {}", exc)
            return parse_decompose_json("", question)
        return parse_decompose_json(raw, question)

    async def generate_followup(
        self, question: str, dq: dict[str, Any], answer: str
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
            raw = await self._call_with_retry(messages, temperature=0.4, max_tokens=256)
        except Exception as exc:
            logger.warning("Stage 5 follow-up generation failed: {}", exc)
            return []
        return parse_followup_json(raw)

    async def generate_starter_questions(self, material_text: str) -> list[str]:
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
            raw = await self._call_with_retry(messages, temperature=0.4, max_tokens=384)
        except Exception as exc:
            logger.warning("Starter question generation failed: {}", exc)
            return []
        return parse_followup_json(raw, limit=5)

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
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=settings.generation_model,
                messages=messages,
                temperature=temperature if temperature is not None else settings.generation_temperature,
                max_tokens=max_tokens if max_tokens is not None else settings.generation_max_tokens,
            )
        except _RETRYABLE:
            raise
        except Exception as exc:
            raise GenerationError(f"LLM call failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise GenerationError("LLM returned empty content")

        logger.info(
            "Generated answer ({} chars) via {}/{}",
            len(content),
            settings.generation_provider,
            settings.generation_model,
        )
        return content