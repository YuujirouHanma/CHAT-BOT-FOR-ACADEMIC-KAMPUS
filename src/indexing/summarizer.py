"""Multimodal summarization for table HTML and image base64 payloads.

Tables and images both go through the single active generation model
(GENERATION_PROVIDER/GENERATION_MODEL in .env) — same client as chat
generation, so the whole pipeline runs on one model.
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
from typing import Any

from openai import APIConnectionError as OAIConnError
from openai import APITimeoutError as OAITimeoutError
from openai import AsyncOpenAI
from openai import InternalServerError as OAIServerError
from openai import RateLimitError as OAIRateLimit
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


_RETRYABLE: tuple[type[Exception], ...] = (
    OAIConnError,
    OAITimeoutError,
    OAIServerError,
    OAIRateLimit,
)


class SummarizationError(RuntimeError):
    """Raised when a summarization call fails irrecoverably (after retries)."""


class MultimodalSummarizer:
    """Async summarizer for tables and images, both via the active generation model.

    Construct once per pipeline run; the SDK client is reusable across calls.
    """

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        self._client = client or AsyncOpenAI(
            api_key=settings.generation_api_key,
            base_url=settings.generation_base_url,
        )

    async def summarize_table(self, table_html: str) -> str:
        cleaned = (table_html or "").strip()
        if not cleaned:
            raise ValueError("Empty table HTML")
        messages: list[Any] = [
            {"role": "user", "content": _TABLE_SUMMARY_PROMPT.format(table_html=cleaned)}
        ]
        return await self._call(messages)

    async def describe_image(self, image_base64: str) -> str:
        cleaned = (image_base64 or "").strip()
        if not cleaned:
            raise ValueError("Empty image base64")
        mime = detect_image_mime(cleaned)
        messages: list[Any] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _IMAGE_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{cleaned}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]
        return await self._call(messages)

    async def _call(self, messages: list[Any]) -> str:
        response = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._client.chat.completions.create(
                        model=settings.generation_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=512,
                    )
                except Exception as exc:
                    if isinstance(exc, _RETRYABLE):
                        raise  # Biarkan tenacity menangani retry
                    raise SummarizationError(f"Summarization call failed: {exc}") from exc

        assert response is not None
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise SummarizationError("Model returned empty content")
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
