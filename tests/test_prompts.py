"""Tests for prompt formatting and context building.

Pure logic, no mocks needed.
"""
from __future__ import annotations

from src.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_retrieval_results,
)


def _result(
    *,
    element_type: str = "text",
    text: str = "",
    source_file: str = "doc.pdf",
    page_number: int | None = 1,
    raw_html: str | None = None,
    image_base64: str | None = None,
) -> dict:
    return {
        "chunk_id": "c1",
        "score": 0.9,
        "rerank_score": 0.95,
        "payload": {
            "text": text,
            "element_type": element_type,
            "source_file": source_file,
            "page_number": page_number,
            "raw_html": raw_html,
            "image_base64": image_base64,
        },
    }


class TestFormatRetrievalResults:
    def test_empty_results_returns_empty_context(self) -> None:
        ctx = format_retrieval_results([])
        assert ctx.text_block == ""
        assert ctx.image_payloads == []

    def test_text_chunk_gets_source_citation(self) -> None:
        results = [_result(text="Hello world.", source_file="lesson.pdf", page_number=3)]
        ctx = format_retrieval_results(results)

        assert "[Sumber 1]" in ctx.text_block
        assert "lesson.pdf" in ctx.text_block
        assert "hal. 3" in ctx.text_block
        assert "Hello world." in ctx.text_block
        assert ctx.image_payloads == []

    def test_table_chunk_includes_raw_html(self) -> None:
        results = [
            _result(
                element_type="table",
                text="Tabel ringkasan",
                raw_html="<table><tr><td>X</td><td>42</td></tr></table>",
            )
        ]
        ctx = format_retrieval_results(results)

        assert "<table>" in ctx.text_block
        assert "42" in ctx.text_block
        assert "TABEL" in ctx.text_block

    def test_image_chunk_emits_payload_and_keeps_description(self) -> None:
        results = [
            _result(
                element_type="image",
                text="Grafik nilai per semester.",
                image_base64="iVBORw0KGgo=",
            )
        ]
        ctx = format_retrieval_results(results)

        assert "Grafik nilai" in ctx.text_block
        assert "GAMBAR" in ctx.text_block
        assert len(ctx.image_payloads) == 1
        assert ctx.image_payloads[0]["source_idx"] == 1
        assert ctx.image_payloads[0]["image_base64"] == "iVBORw0KGgo="

    def test_image_without_base64_no_payload(self) -> None:
        results = [
            _result(
                element_type="image",
                text="Description only.",
                image_base64=None,
            )
        ]
        ctx = format_retrieval_results(results)

        assert "Description only." in ctx.text_block
        assert ctx.image_payloads == []

    def test_sequential_numbering(self) -> None:
        results = [
            _result(text="First."),
            _result(text="Second."),
            _result(element_type="image", text="Third.", image_base64="abc"),
        ]
        ctx = format_retrieval_results(results)

        assert "[Sumber 1]" in ctx.text_block
        assert "[Sumber 2]" in ctx.text_block
        assert "[Sumber 3]" in ctx.text_block
        assert ctx.image_payloads[0]["source_idx"] == 3

    def test_missing_page_number_omitted(self) -> None:
        results = [_result(text="No page.", page_number=None)]
        ctx = format_retrieval_results(results)

        assert "[Sumber 1]" in ctx.text_block
        assert "hal." not in ctx.text_block

    def test_table_without_html_falls_back_to_text(self) -> None:
        results = [
            _result(
                element_type="table",
                text="Plain table summary.",
                raw_html=None,
            )
        ]
        ctx = format_retrieval_results(results)
        assert "Plain table summary." in ctx.text_block


class TestBuildUserPrompt:
    def test_includes_question_and_context(self) -> None:
        ctx = format_retrieval_results(
            [_result(text="Context fact.")]
        )
        prompt = build_user_prompt("Apa itu X?", ctx)

        assert "Apa itu X?" in prompt
        assert "Context fact." in prompt
        assert "KONTEKS" in prompt
        assert "PERTANYAAN" in prompt

    def test_question_is_stripped(self) -> None:
        ctx = format_retrieval_results([_result(text="ctx")])
        prompt = build_user_prompt("   What?   \n", ctx)
        assert "What?" in prompt


class TestSystemPrompt:
    def test_mentions_citation_format(self) -> None:
        assert "[Sumber" in SYSTEM_PROMPT

    def test_mentions_grounding(self) -> None:
        assert "konteks" in SYSTEM_PROMPT.lower()