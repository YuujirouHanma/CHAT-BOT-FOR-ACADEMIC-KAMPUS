"""Multimodal summarization for table HTML and image base64 payloads.

Tables go to Groq Llama 3.1 (fast, cheap, good with structured text).
Images go to OpenAI GPT-4o mini vision (best price/quality for academic content).
Text elements pass through unchanged — their content is already embeddable.

Public API:
    summarizer = MultimodalSummarizer()
    enriched = await enrich_elements(parsed_elements, summarizer)

Each enriched ParsedElement is a NEW instance (Pydantic model_copy); inputs
are not mutated. If a single element's summarization fails after retries,
that element is returned without a summary instead of crashing the batch.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

class _DummyRetryableError(Exception):
    pass

if TYPE_CHECKING:
    from groq import AsyncGroq
    from groq import APIConnectionError as GroqConnError
    from groq import APITimeoutError as GroqTimeoutError
    from groq import InternalServerError as GroqServerError
    from groq import RateLimitError as GroqRateLimit
    from openai import AsyncOpenAI
    from openai import APIConnectionError as OAIConnError
    from openai import APITimeoutError as OAITimeoutError
    from openai import InternalServerError as OAIServerError
    from openai import RateLimitError as OAIRateLimit
else:
    try:
        from groq import AsyncGroq
        from groq import APIConnectionError as GroqConnError
        from groq import APITimeoutError as GroqTimeoutError
        from groq import InternalServerError as GroqServerError
        from groq import RateLimitError as GroqRateLimit
    except ImportError:  # pragma: no cover - runtime should have package installed
        AsyncGroq = Any
        GroqConnError = _DummyRetryableError
        GroqTimeoutError = _DummyRetryableError
        GroqServerError = _DummyRetryableError
        GroqRateLimit = _DummyRetryableError

    try:
        from openai import AsyncOpenAI
        from openai import APIConnectionError as OAIConnError
        from openai import APITimeoutError as OAITimeoutError
        from openai import InternalServerError as OAIServerError
        from openai import RateLimitError as OAIRateLimit
    except ImportError:  # pragma: no cover - runtime should have package installed
        AsyncOpenAI = Any
        OAIConnError = _DummyRetryableError
        OAITimeoutError = _DummyRetryableError
        OAIServerError = _DummyRetryableError
        OAIRateLimit = _DummyRetryableError

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.schemas import ElementType, ParsedElement
from src.utils.image import detect_image_mime
from src.utils.logger import logger


def _detect_image_mime(image_base64: str) -> str:
    """Backward-compatible wrapper expected by tests."""
    return detect_image_mime(image_base64)

_TABLE_SUMMARY_PROMPT = """Anda adalah asisten yang meringkas tabel untuk sistem retrieval-augmented generation (RAG) edukasi.

Ringkas tabel HTML berikut dalam 2-4 kalimat bahasa Indonesia. Fokus pada:
- Topik atau judul tabel (jika ada)
- Variabel atau kolom utama yang dibandingkan
- Tren atau perbedaan paling signifikan
- Angka kunci (nilai tertinggi, terendah, atau yang mencolok)

JANGAN salin tabel verbatim. Tulis ringkasan natural yang akan membantu pencarian semantik.

Tabel:
{table_html}

Ringkasan:"""


_IMAGE_DESCRIPTION_PROMPT = """Deskripsikan gambar ini untuk sistem retrieval-augmented generation (RAG) edukasi dalam 3-5 kalimat bahasa Indonesia.

Fokus pada:
- Jenis visual: foto, diagram, grafik (bar/line/scatter/pie), screenshot, ilustrasi
- Topik dan konsep utama yang ditampilkan
- Untuk grafik atau chart: sumbu, satuan, tren, dan nilai-nilai mencolok
- Untuk diagram: komponen utama dan hubungannya
- Teks yang terlihat di dalam gambar (jika ada dan relevan)

Tujuan: deskripsi ini akan dipakai untuk mencari gambar dengan query mahasiswa."""


_GROQ_RETRY_ERRORS: tuple[type[Exception], ...] = (
    GroqConnError,
    GroqTimeoutError,
    GroqServerError,
    GroqRateLimit,
)
_OAI_RETRY_ERRORS: tuple[type[Exception], ...] = (
    OAIConnError,
    OAITimeoutError,
    OAIServerError,
    OAIRateLimit,
)


class SummarizationError(RuntimeError):
    """Raised when a summarization call fails irrecoverably (after retries)."""


class MultimodalSummarizer:
    """Async summarizer wrapping Groq (text/tables) and OpenAI (images).

    Construct once per pipeline run; both SDK clients are reusable across calls.
    """

    def __init__(
        self,
        groq_client: AsyncGroq | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        # Get Groq API key safely
        groq_key_obj: Any = settings.groq_api_key
        if hasattr(groq_key_obj, "get_secret_value"):
            groq_key = groq_key_obj.get_secret_value()
        else:
            groq_key = str(groq_key_obj)
        
        self._groq = groq_client or AsyncGroq(api_key=groq_key)
        
        # Get OpenAI API key safely
        openai_key_obj: Any = settings.openai_api_key
        if hasattr(openai_key_obj, "get_secret_value"):
            openai_key = openai_key_obj.get_secret_value()
        else:
            openai_key = str(openai_key_obj)
        
        self._openai = openai_client or AsyncOpenAI(api_key=openai_key)

    async def summarize_table(self, table_html: str) -> str:
        cleaned = (table_html or "").strip()
        if not cleaned:
            raise ValueError("Empty table HTML")
        return await self._call_groq(
            prompt=_TABLE_SUMMARY_PROMPT.format(table_html=cleaned)
        )

    async def describe_image(self, image_base64: str) -> str:
        cleaned = (image_base64 or "").strip()
        if not cleaned:
            raise ValueError("Empty image base64")
        mime = detect_image_mime(cleaned)
        return await self._call_openai_vision(image_base64=cleaned, mime=mime)

    async def _call_groq(self, prompt: str) -> str:
        response = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_GROQ_RETRY_ERRORS),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._groq.chat.completions.create(  # type: ignore[union-attr,call-arg]
                        model=settings.groq_summary_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=512,
                    )
                except Exception as exc:
                    if isinstance(exc, _GROQ_RETRY_ERRORS):
                        raise  # Biarkan tenacity menangani retry
                    raise SummarizationError(f"Groq call failed: {exc}") from exc

        assert response is not None
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise SummarizationError("Groq returned empty content")
        return content

    async def _call_openai_vision(self, image_base64: str, mime: str) -> str:
        response = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_OAI_RETRY_ERRORS),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._openai.chat.completions.create(  # type: ignore[union-attr,call-arg]
                        model=settings.openai_vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": _IMAGE_DESCRIPTION_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime};base64,{image_base64}",
                                            "detail": "low",
                                        },
                                    },
                                ],
                            }
                        ],
                        temperature=0.2,
                        max_tokens=512,
                    )
                except Exception as exc:
                    if isinstance(exc, _OAI_RETRY_ERRORS):
                        raise  # Biarkan tenacity menangani retry
                    raise SummarizationError(f"OpenAI vision call failed: {exc}") from exc

        assert response is not None
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise SummarizationError("OpenAI returned empty content")
        return content


async def enrich_elements(
    elements: Sequence[ParsedElement],
    summarizer: MultimodalSummarizer | None = None,
    max_concurrency: int = 5,
) -> list[ParsedElement]:
    """Populate `.summary` on TABLE and IMAGE elements.

    TEXT elements pass through unchanged. Per-element failures are logged
    and the element is returned without a summary (its embeddable_text()
    will fall back to .content).

    Args:
        elements: Output of parse_document().
        summarizer: Optional injected summarizer (for testing).
        max_concurrency: Cap on simultaneous in-flight API calls.

    Returns:
        New list of ParsedElement; input is not mutated.
    """
    summarizer = summarizer or MultimodalSummarizer()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _enrich_one(element: ParsedElement) -> ParsedElement:
        if element.element_type == ElementType.TEXT:
            return element

        async with semaphore:
            try:
                if element.element_type == ElementType.TABLE:
                    if not element.raw_html:
                        logger.warning(
                            f"Table {element.element_id} has no HTML, skipping summary"
                        )
                        return element
                    summary = await summarizer.summarize_table(element.raw_html)

                elif element.element_type == ElementType.IMAGE:
                    if not element.image_base64:
                        logger.warning(
                            f"Image {element.element_id} has no base64, skipping summary"
                        )
                        return element
                    summary = await summarizer.describe_image(element.image_base64)

                else:
                    return element

                return element.model_copy(update={"summary": summary})

            except Exception as exc:  # pyright: ignore[reportGeneralTypeIssues]
                logger.error(
                    f"Summarization failed for {element.element_type.value} {element.element_id}: {exc}"
                )
                return element

    results = await asyncio.gather(*(_enrich_one(e) for e in elements))

    enriched_count = sum(1 for e in results if e.summary)
    target_count = sum(
        1 for e in results if e.element_type != ElementType.TEXT
    )
    logger.info(
        f"Enriched {enriched_count}/{target_count} non-text elements with summaries"
    )
    return list(results)