"""Tests for the LLM generator.

OpenAI client is mocked. We verify message construction (especially
multimodal content blocks for images) and error handling.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.generation.llm import GenerationError, LLMGenerator
from src.generation.prompts import FormattedContext


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_generator(
    response_text: str = "Jawaban model.",
) -> tuple[LLMGenerator, AsyncMock]:
    create_mock = AsyncMock(return_value=_completion(response_text))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    gen = LLMGenerator(client=client)  # type: ignore[arg-type]
    return gen, create_mock


def _png_b64() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()


class TestLLMGenerator:
    @pytest.mark.asyncio
    async def test_returns_content(self) -> None:
        gen, _ = _make_generator(response_text="Halo dunia.")
        ctx = FormattedContext(text_block="ctx", image_payloads=[])

        result = await gen.generate("pertanyaan?", ctx)
        assert result == "Halo dunia."

    @pytest.mark.asyncio
    async def test_text_only_context_no_image_blocks(self) -> None:
        gen, create_mock = _make_generator()
        ctx = FormattedContext(text_block="just text", image_payloads=[])

        await gen.generate("q", ctx)

        messages = create_mock.call_args.kwargs["messages"]
        user_content = messages[1]["content"]
        # Text-only: content is a plain string, not a list
        assert isinstance(user_content, str)
        assert "just text" in user_content

    @pytest.mark.asyncio
    async def test_image_context_adds_image_url_block(self) -> None:
        gen, create_mock = _make_generator()
        ctx = FormattedContext(
            text_block="ctx with image",
            image_payloads=[{"source_idx": 1, "image_base64": _png_b64()}],
        )

        await gen.generate("q", ctx)

        user_content = create_mock.call_args.kwargs["messages"][1]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
        assert "data:image/png;base64," in user_content[1]["image_url"]["url"]
        assert user_content[1]["image_url"]["detail"] == "low"

    @pytest.mark.asyncio
    async def test_multiple_images_added_in_order(self) -> None:
        gen, create_mock = _make_generator()
        ctx = FormattedContext(
            text_block="ctx",
            image_payloads=[
                {"source_idx": 1, "image_base64": _png_b64()},
                {"source_idx": 2, "image_base64": _png_b64()},
                {"source_idx": 3, "image_base64": _png_b64()},
            ],
        )

        await gen.generate("q", ctx)

        user_content = create_mock.call_args.kwargs["messages"][1]["content"]
        assert len(user_content) == 4
        assert sum(1 for c in user_content if c["type"] == "image_url") == 3

    @pytest.mark.asyncio
    async def test_system_prompt_included(self) -> None:
        gen, create_mock = _make_generator()
        ctx = FormattedContext(text_block="ctx", image_payloads=[])

        await gen.generate("q", ctx)

        messages = create_mock.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)
        assert len(messages[0]["content"]) > 0

    @pytest.mark.asyncio
    async def test_empty_response_raises(self) -> None:
        gen, create_mock = _make_generator()
        create_mock.return_value = _completion("")

        ctx = FormattedContext(text_block="ctx", image_payloads=[])
        with pytest.raises(GenerationError, match="empty content"):
            await gen.generate("q", ctx)

    @pytest.mark.asyncio
    async def test_settings_used_for_model_and_temperature(self) -> None:
        from src.config import settings

        gen, create_mock = _make_generator()
        ctx = FormattedContext(text_block="ctx", image_payloads=[])

        await gen.generate("q", ctx)

        kwargs = create_mock.call_args.kwargs
        assert kwargs["model"] == settings.generation_model
        assert kwargs["temperature"] == settings.generation_temperature
        assert kwargs["max_tokens"] == settings.generation_max_tokens


class TestModelSwitching:
    @pytest.mark.asyncio
    async def test_none_uses_default_client_and_model(self) -> None:
        from src.config import settings

        gen, default_mock = _make_generator()
        await gen.generate("q", FormattedContext("ctx", []), model=None)

        assert default_mock.call_args.kwargs["model"] == settings.generation_model

    @pytest.mark.asyncio
    async def test_registry_key_switches_provider_and_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gen, default_mock = _make_generator()
        # A distinct mock client for whatever provider gets built.
        other_create = AsyncMock(return_value=_completion("groq answer"))
        other_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=other_create))
        )
        monkeypatch.setattr(
            LLMGenerator, "_build_client", staticmethod(lambda provider: other_client)
        )

        out = await gen.generate("q", FormattedContext("ctx", []), model="groq-llama-70b")

        assert out == "groq answer"
        assert other_create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"
        default_mock.assert_not_called()  # default client bypassed

    @pytest.mark.asyncio
    async def test_unknown_model_falls_back_to_default(self) -> None:
        from src.config import settings

        gen, default_mock = _make_generator()
        await gen.generate("q", FormattedContext("ctx", []), model="tidak-ada")

        assert default_mock.call_args.kwargs["model"] == settings.generation_model

    @pytest.mark.asyncio
    async def test_generate_quiz_parses_items(self) -> None:
        import json

        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(json.dumps([
            {"question": "Apa itu LIFO?", "options": ["a", "b", "c", "d"],
             "answer_index": 2, "explanation": "stack"},
        ]))

        quiz = await gen.generate_quiz("materi stack")

        assert len(quiz) == 1
        assert quiz[0]["answer_index"] == 2
        assert len(quiz[0]["options"]) == 4