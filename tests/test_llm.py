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


def _completion(
    content: str,
    finish_reason: str | None = None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Respons chat-completion tiruan.

    `finish_reason` dan `usage` opsional agar bentuk lama tetap terpakai sebagai
    kasus "penyedia tidak melaporkan apa pun".
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _usage(
    prompt: int = 100, completion: int = 50, reasoning: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


class _LogPalsu:
    """Perekam log; loguru dipanggil dengan placeholder {} sehingga cukup format()."""

    def __init__(self) -> None:
        self.info_baris: list[str] = []
        self.warning_baris: list[str] = []

    def info(self, msg: str, *args: object) -> None:
        self.info_baris.append(msg.format(*args))

    def warning(self, msg: str, *args: object) -> None:
        self.warning_baris.append(msg.format(*args))


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


class TestAnggaranHabis:
    """Konten kosong karena jatah token habis harus diulang; penolakan tidak.

    Model penalar (Qwen 3.7) memakai 2.900-3.800 token untuk berpikir sebelum
    menulis apa pun. Bila jatahnya habis di situ, penyedia mengembalikan konten
    kosong dengan finish_reason='length' — kegagalan sementara yang dulu justru
    permanen karena GenerationError tidak pernah diulang.
    """

    @pytest.mark.asyncio
    async def test_kosong_karena_length_diulang_dengan_anggaran_lebih_besar(self) -> None:
        gen, create_mock = _make_generator()
        create_mock.side_effect = [
            _completion("", finish_reason="length", usage=_usage(completion=3072, reasoning=3072)),
            _completion("Jawaban pada percobaan kedua."),
        ]

        hasil = await gen.generate("q", FormattedContext("ctx", []))

        assert hasil == "Jawaban pada percobaan kedua."
        assert create_mock.call_count == 2
        anggaran_1 = create_mock.call_args_list[0].kwargs["max_tokens"]
        anggaran_2 = create_mock.call_args_list[1].kwargs["max_tokens"]
        # Mengulang dengan anggaran yang sama akan terpotong di titik yang sama.
        assert anggaran_2 > anggaran_1

    @pytest.mark.asyncio
    async def test_length_terus_menerus_berhenti_pada_batas_percobaan(self) -> None:
        from src.generation.llm import _EMPTY_RETRY_ATTEMPTS

        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(
            "", finish_reason="length", usage=_usage(completion=3072)
        )

        with pytest.raises(GenerationError, match="empty content"):
            await gen.generate("q", FormattedContext("ctx", []))

        # Panggilan mahal tidak boleh diulang tanpa batas.
        assert create_mock.call_count == _EMPTY_RETRY_ATTEMPTS

    @pytest.mark.asyncio
    async def test_anggaran_tidak_pernah_melewati_plafon(self) -> None:
        from src.generation.llm import _MAX_TOKENS_CEILING

        gen, create_mock = _make_generator()
        create_mock.return_value = _completion("", finish_reason="length")

        with pytest.raises(GenerationError):
            await gen._call_with_retry(
                [{"role": "user", "content": "q"}], max_tokens=_MAX_TOKENS_CEILING
            )

        # Sudah mentok di plafon: menaikkan lagi mustahil, jadi tidak ada ulangan.
        assert create_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_completion_tokens_menyentuh_batas_dianggap_terpotong(self) -> None:
        """Penyedia yang tidak mengirim finish_reason tetap terdeteksi."""
        gen, create_mock = _make_generator()
        create_mock.side_effect = [
            _completion("", usage=_usage(completion=3072)),
            _completion("Jawaban."),
        ]

        hasil = await gen.generate("q", FormattedContext("ctx", []))

        assert hasil == "Jawaban."
        assert create_mock.call_count == 2


class TestKegagalanPermanenTidakDiulang:
    @pytest.mark.asyncio
    async def test_kosong_dengan_finish_stop_gagal_sekali_saja(self) -> None:
        """Model selesai bicara tetapi tidak menulis apa pun — penolakan, bukan potong."""
        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(
            "", finish_reason="stop", usage=_usage(completion=12)
        )

        with pytest.raises(GenerationError, match="empty content"):
            await gen.generate("q", FormattedContext("ctx", []))

        assert create_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_kosong_karena_content_filter_tidak_diulang(self) -> None:
        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(
            "", finish_reason="content_filter", usage=_usage(completion=5)
        )

        with pytest.raises(GenerationError):
            await gen.generate("q", FormattedContext("ctx", []))

        assert create_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_galat_non_retryable_tidak_diulang(self) -> None:
        gen, create_mock = _make_generator()
        create_mock.side_effect = ValueError("400 bad request")

        with pytest.raises(GenerationError, match="LLM call failed"):
            await gen.generate("q", FormattedContext("ctx", []))

        assert create_mock.call_count == 1


class TestInstrumentasiFinishReason:
    @pytest.mark.asyncio
    async def test_pemotongan_dicatat_sebagai_peringatan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jawaban terpotong tetap dikembalikan, tetapi tidak boleh lewat diam-diam."""
        log = _LogPalsu()
        monkeypatch.setattr("src.generation.llm.logger", log)
        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(
            "Jawaban yang terpoto", finish_reason="length",
            usage=_usage(completion=3072, reasoning=2900),
        )

        await gen.generate("q", FormattedContext("ctx", []))

        gabung = " | ".join(log.warning_baris)
        assert "DIPOTONG" in gabung
        assert "max_tokens=3072" in gabung
        assert "finish=length" in gabung

    @pytest.mark.asyncio
    async def test_log_sukses_memuat_finish_reason_dan_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _LogPalsu()
        monkeypatch.setattr("src.generation.llm.logger", log)
        gen, create_mock = _make_generator()
        create_mock.return_value = _completion(
            "Jawaban utuh.", finish_reason="stop",
            usage=_usage(prompt=800, completion=420, reasoning=310),
        )

        await gen.generate("q", FormattedContext("ctx", []))

        baris = " | ".join(log.info_baris)
        assert "finish=stop" in baris
        assert "out=420" in baris
        assert "reasoning=310" in baris
        assert not log.warning_baris

