"""Generation LLM wrapper.

Uses an OpenAI-compatible chat completion API. Vision content blocks are
attached when retrieved chunks include images, so the model can re-read
the actual image at answer time rather than relying solely on the
indexing-time description.

Defaults to gpt-4o-mini (multimodal). For self-hosted, swap to Qwen2.5-VL
served via vLLM with an OpenAI-compatible endpoint.
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
    SYSTEM_PROMPT,
    FormattedContext,
    build_user_prompt,
)
from src.indexing.summarizer import _detect_image_mime
from src.utils.logger import logger

_RETRYABLE = (OAIConnError, OAITimeoutError, OAIServerError, OAIRateLimit)


class GenerationError(RuntimeError):
    """Raised when generation fails after retries."""


class LLMGenerator:
    """Multimodal-aware generator. One instance per app run."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )

    async def generate(
        self,
        question: str,
        context: FormattedContext,
    ) -> str:
        """Generate an answer grounded in the retrieved context.

        Args:
            question: User question.
            context: Output of `format_retrieval_results()`.

        Returns:
            The model's answer as a plain string.
        """
        user_prompt = build_user_prompt(question, context)
        user_content = self._build_user_content(user_prompt, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await self._call_with_retry(messages)

    @staticmethod
    def _build_user_content(
        user_prompt: str, context: FormattedContext
    ) -> list[dict[str, Any]]:
        """Build the user message — text first, then any image blocks."""
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
    async def _call_with_retry(self, messages: list[dict[str, Any]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=settings.generation_model,
                messages=messages,
                temperature=settings.generation_temperature,
                max_tokens=settings.generation_max_tokens,
            )
        except _RETRYABLE:
            raise
        except Exception as exc:
            raise GenerationError(f"LLM call failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise GenerationError("LLM returned empty content")

        logger.info(
            "Generated answer ({} chars) using {}",
            len(content),
            settings.generation_model,
        )
        return content