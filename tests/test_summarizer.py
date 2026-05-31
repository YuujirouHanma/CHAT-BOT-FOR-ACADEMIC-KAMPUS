"""Tests for the multimodal summarizer.

All API calls are mocked. Async tests run via pytest-asyncio.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.indexing.summarizer import (
    MultimodalSummarizer,
    SummarizationError,
    _detect_image_mime,
    enrich_elements,
)
from src.schemas import ElementType, ParsedElement


def _completion_response(content: str) -> SimpleNamespace:
    """Shape a fake OpenAI/Groq chat.completions.create response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_summarizer(
    groq_text: str = "Ringkasan tabel.",
    openai_text: str = "Deskripsi gambar.",
) -> tuple[MultimodalSummarizer, AsyncMock, AsyncMock]:
    """Build a summarizer with both SDK clients fully mocked."""
    groq_create = AsyncMock(return_value=_completion_response(groq_text))
    groq_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=groq_create)
        )
    )

    openai_create = AsyncMock(return_value=_completion_response(openai_text))
    openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=openai_create)
        )
    )

    summarizer = MultimodalSummarizer(
        groq_client=groq_client,  # type: ignore[arg-type]
        openai_client=openai_client,  # type: ignore[arg-type]
    )
    return summarizer, groq_create, openai_create


def _make_element(
    element_type: ElementType,
    *,
    content: str = "",
    raw_html: str | None = None,
    image_base64: str | None = None,
) -> ParsedElement:
    return ParsedElement(
        element_id=f"id-{element_type.value}",
        element_type=element_type,
        content=content,
        raw_html=raw_html,
        image_base64=image_base64,
        source_file="test.pdf",
        page_number=1,
    )


class TestMimeDetection:
    def test_png(self) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        assert _detect_image_mime(base64.b64encode(png_bytes).decode()) == "image/png"

    def test_jpeg(self) -> None:
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 16
        assert _detect_image_mime(base64.b64encode(jpeg_bytes).decode()) == "image/jpeg"

    def test_gif(self) -> None:
        gif_bytes = b"GIF89a" + b"\x00" * 16
        assert _detect_image_mime(base64.b64encode(gif_bytes).decode()) == "image/gif"

    def test_webp(self) -> None:
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        assert _detect_image_mime(base64.b64encode(webp_bytes).decode()) == "image/webp"

    def test_unknown_falls_back_to_png(self) -> None:
        assert _detect_image_mime("invalid===") == "image/png"


class TestSummarizer:
    @pytest.mark.asyncio
    async def test_summarize_table_returns_content(self) -> None:
        summ, groq_mock, _ = _make_summarizer(groq_text="Tabel berisi data nilai.")
        result = await summ.summarize_table("<table><tr><td>A</td></tr></table>")

        assert result == "Tabel berisi data nilai."
        groq_mock.assert_awaited_once()
        kwargs = groq_mock.call_args.kwargs
        assert "<table>" in kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_summarize_table_empty_raises(self) -> None:
        summ, _, _ = _make_summarizer()
        with pytest.raises(ValueError, match="Empty table HTML"):
            await summ.summarize_table("")

    @pytest.mark.asyncio
    async def test_describe_image_returns_content(self) -> None:
        summ, _, openai_mock = _make_summarizer(openai_text="Grafik bar nilai siswa.")
        png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()

        result = await summ.describe_image(png_b64)

        assert result == "Grafik bar nilai siswa."
        openai_mock.assert_awaited_once()
        msg = openai_mock.call_args.kwargs["messages"][0]
        assert msg["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][1]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_describe_image_empty_raises(self) -> None:
        summ, _, _ = _make_summarizer()
        with pytest.raises(ValueError, match="Empty image base64"):
            await summ.describe_image("")

    @pytest.mark.asyncio
    async def test_empty_response_raises_summarization_error(self) -> None:
        summ, groq_mock, _ = _make_summarizer()
        groq_mock.return_value = _completion_response("")

        with pytest.raises(SummarizationError, match="empty content"):
            await summ.summarize_table("<table/>")


class TestEnrichElements:
    @pytest.mark.asyncio
    async def test_text_passes_through_unchanged(self) -> None:
        summ, groq_mock, openai_mock = _make_summarizer()
        text_el = _make_element(ElementType.TEXT, content="Hello world.")

        result = await enrich_elements([text_el], summarizer=summ)

        assert len(result) == 1
        assert result[0].summary is None
        assert result[0].content == "Hello world."
        groq_mock.assert_not_awaited()
        openai_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_table_gets_summary_via_groq(self) -> None:
        summ, groq_mock, _ = _make_summarizer(groq_text="Tabel statistik nilai.")
        table_el = _make_element(
            ElementType.TABLE,
            raw_html="<table><tr><td>X</td></tr></table>",
        )

        result = await enrich_elements([table_el], summarizer=summ)

        assert result[0].summary == "Tabel statistik nilai."
        assert result[0].raw_html == "<table><tr><td>X</td></tr></table>"
        groq_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_image_gets_summary_via_openai(self) -> None:
        summ, _, openai_mock = _make_summarizer(openai_text="Diagram alir proses.")
        png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
        image_el = _make_element(ElementType.IMAGE, image_base64=png_b64)

        result = await enrich_elements([image_el], summarizer=summ)

        assert result[0].summary == "Diagram alir proses."
        assert result[0].image_base64 == png_b64
        openai_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_input_elements_not_mutated(self) -> None:
        summ, _, _ = _make_summarizer(groq_text="ringkasan")
        original = _make_element(ElementType.TABLE, raw_html="<table/>")
        assert original.summary is None

        result = await enrich_elements([original], summarizer=summ)

        assert original.summary is None
        assert result[0].summary == "ringkasan"
        assert result[0].element_id == original.element_id

    @pytest.mark.asyncio
    async def test_table_without_html_skipped_gracefully(self) -> None:
        summ, groq_mock, _ = _make_summarizer()
        table_el = _make_element(ElementType.TABLE, raw_html=None)

        result = await enrich_elements([table_el], summarizer=summ)

        assert result[0].summary is None
        groq_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failure_does_not_break_batch(self) -> None:
        summ, groq_mock, _ = _make_summarizer()
        groq_mock.side_effect = SummarizationError("boom")

        elements = [
            _make_element(ElementType.TEXT, content="ok"),
            _make_element(ElementType.TABLE, raw_html="<table/>"),
            _make_element(ElementType.TEXT, content="also ok"),
        ]

        result = await enrich_elements(elements, summarizer=summ)

        assert len(result) == 3
        assert result[0].content == "ok"
        assert result[1].summary is None
        assert result[2].content == "also ok"

    @pytest.mark.asyncio
    async def test_concurrency_processes_mixed_batch(self) -> None:
        summ, groq_mock, openai_mock = _make_summarizer(
            groq_text="tabel-ringkasan", openai_text="gambar-deskripsi"
        )
        png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()

        elements = [
            _make_element(ElementType.TEXT, content="t1"),
            _make_element(ElementType.TABLE, raw_html="<table>1</table>"),
            _make_element(ElementType.IMAGE, image_base64=png_b64),
            _make_element(ElementType.TABLE, raw_html="<table>2</table>"),
        ]

        result = await enrich_elements(elements, summarizer=summ, max_concurrency=2)

        assert [e.summary for e in result] == [
            None,
            "tabel-ringkasan",
            "gambar-deskripsi",
            "tabel-ringkasan",
        ]
        assert groq_mock.await_count == 2
        assert openai_mock.await_count == 1