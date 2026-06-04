"""Generation LLM wrapper."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from openai import APIConnectionError as OAIConnError
    from openai import APITimeoutError as OAITimeoutError
    from openai import AsyncOpenAI
    from openai import InternalServerError as OAIServerError
    from openai import RateLimitError as OAIRateLimit
else:
    try:
        from openai import APIConnectionError as OAIConnError
        from openai import APITimeoutError as OAITimeoutError
        from openai import AsyncOpenAI
        from openai import InternalServerError as OAIServerError
        from openai import RateLimitError as OAIRateLimit
    except ImportError:
        class _DummyRetryableError(Exception):
            pass

        OAIConnError = _DummyRetryableError
        OAITimeoutError = _DummyRetryableError
        AsyncOpenAI = Any
        OAIServerError = _DummyRetryableError
        OAIRateLimit = _DummyRetryableError

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
from src.utils.image import detect_image_mime
from src.utils.logger import logger

# Explicit type annotation agar static checker tidak bingung
_RETRYABLE: tuple[type[Exception], ...] = (
    OAIConnError,
    OAITimeoutError,
    OAIServerError,
    OAIRateLimit,
)


class GenerationError(RuntimeError):
    """Raised when generation fails after retries."""


class LLMGenerator:
    """Multimodal-aware generator. One instance per app run."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        if client:
            self._client = client
        elif settings.generation_provider == "groq":
            groq_key: Any = settings.groq_api_key
            raw_key = groq_key.get_secret_value() if hasattr(groq_key, "get_secret_value") else str(groq_key)
            self._client = AsyncOpenAI(
                api_key=raw_key,
                base_url="https://api.groq.com/openai/v1",
            )
            logger.info("LLMGenerator using Groq provider ({})", settings.generation_model)
        else:
            api_key_obj: Any = settings.openai_api_key
            raw_key = api_key_obj.get_secret_value() if hasattr(api_key_obj, "get_secret_value") else str(api_key_obj)
            self._client = AsyncOpenAI(api_key=raw_key)
            logger.info("LLMGenerator using OpenAI provider ({})", settings.generation_model)

    async def generate(
        self,
        question: str,
        context: FormattedContext,
    ) -> str:
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
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img in context.image_payloads:
            mime = detect_image_mime(img["image_base64"])
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
            response = await self._client.chat.completions.create(  # type: ignore[union-attr, call-arg]
                model=settings.generation_model,
                messages=cast(Any, messages),
                temperature=settings.generation_temperature,
                max_tokens=settings.generation_max_tokens,
            )
        except Exception as exc:
            if isinstance(exc, _RETRYABLE):
                raise  # Biarkan tenacity menangani retry
            raise GenerationError(f"LLM call failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise GenerationError("LLM returned empty content")

        logger.info(
            f"Generated answer ({len(content)} chars) using {settings.generation_model}"
        )
        return content