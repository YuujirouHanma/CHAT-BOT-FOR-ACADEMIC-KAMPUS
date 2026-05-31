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
import base64 as _b64
from collections.abc import Sequence

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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.schemas import ElementType, ParsedElement
from src.utils.logger import logger

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


_GROQ_RETRY_ERRORS = (GroqConnError, GroqTimeoutError, GroqServerError, GroqRateLimit)
_OAI_RETRY_ERRORS = (OAIConnError, OAITimeoutError, OAIServerError, OAIRateLimit)


class SummarizationError(RuntimeError):
    """Raised when a summarization call fails irrecoverably (after retries)."""


def _detect_image_mime(image_base64: str) -> str:
    """Detect image MIME type from base64-encoded magic bytes.

    Returns image/png as a safe default. OpenAI vision auto-detects, but
    sending the correct MIME prevents ambiguity.
    """
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


class MultimodalSummarizer:
    """Async summarizer wrapping Groq (text/tables) and OpenAI (images).

    Construct once per pipeline run; both SDK clients are reusable across calls.
    """

    def __init__(
        self,
        groq_client: AsyncGroq | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self._groq = groq_client or AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._openai = openai_client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )

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
        mime = _detect_image_mime(cleaned)
        return await self._call_openai_vision(image_base64=cleaned, mime=mime)

    @retry(
        retry=retry_if_exception_type(_GROQ_RETRY_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _call_groq(self, prompt: str) -> str:
        try:
            response = await self._groq.chat.completions.create(
                model=settings.groq_summary_model, #model llama-3.1-8b-instant
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
        except _GROQ_RETRY_ERRORS:
            raise
        except Exception as exc:
            raise SummarizationError(f"Groq call failed: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise SummarizationError("Groq returned empty content")
        return content

    @retry(
        retry=retry_if_exception_type(_OAI_RETRY_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _call_openai_vision(self, image_base64: str, mime: str) -> str:
        try:
            response = await self._openai.chat.completions.create(
                model=settings.openai_vision_model, #model gpt-4o-mini
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
        except _OAI_RETRY_ERRORS:
            raise
        except Exception as exc:
            raise SummarizationError(f"OpenAI vision call failed: {exc}") from exc

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
                            "Table {} has no HTML, skipping summary",
                            element.element_id,
                        )
                        return element
                    summary = await summarizer.summarize_table(element.raw_html)

                elif element.element_type == ElementType.IMAGE:
                    if not element.image_base64:
                        logger.warning(
                            "Image {} has no base64, skipping summary",
                            element.element_id,
                        )
                        return element
                    summary = await summarizer.describe_image(element.image_base64)

                else:
                    return element

                return element.model_copy(update={"summary": summary})

            except Exception as exc:
                logger.error(
                    "Summarization failed for {} {}: {}",
                    element.element_type.value,
                    element.element_id,
                    exc,
                )
                return element

    results = await asyncio.gather(*(_enrich_one(e) for e in elements))

    enriched_count = sum(1 for e in results if e.summary)
    target_count = sum(
        1 for e in results if e.element_type != ElementType.TEXT
    )
    logger.info(
        "Enriched {}/{} non-text elements with summaries",
        enriched_count,
        target_count,
    )
    return list(results)