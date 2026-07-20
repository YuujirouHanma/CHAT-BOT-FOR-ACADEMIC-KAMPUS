"""Tests for video/audio transcription.

ffmpeg and the Whisper API are mocked — we verify routing (which files count as
media), transcript assembly, and that every failure mode degrades to an empty
result instead of raising (an upload must never crash because of transcription).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion import transcriber
from src.schemas import ElementType


def _media_file(tmp_path: Path, name: str = "kuliah.mp4") -> Path:
    f = tmp_path / name
    f.write_bytes(b"fake media bytes")
    return f


class TestIsMedia:
    @pytest.mark.parametrize("name", ["a.mp4", "a.mkv", "a.webm", "a.mp3", "a.wav", "a.m4a"])
    def test_media_extensions(self, name: str) -> None:
        assert transcriber.is_media(Path(name)) is True

    @pytest.mark.parametrize("name", ["a.pdf", "a.pptx", "a.docx", "a.txt", "a.png"])
    def test_non_media_extensions(self, name: str) -> None:
        assert transcriber.is_media(Path(name)) is False

    def test_case_insensitive(self) -> None:
        assert transcriber.is_media(Path("REKAMAN.MP4")) is True


class TestTranscribeMediaGuards:
    @pytest.mark.asyncio
    async def test_disabled_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber.settings, "transcription_provider", "disabled")
        assert await transcriber.transcribe_media(_media_file(tmp_path)) == ""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber.settings, "transcription_provider", "groq")
        monkeypatch.setattr(
            type(transcriber.settings), "api_key_for", lambda self, provider: ""
        )
        assert await transcriber.transcribe_media(_media_file(tmp_path)) == ""

    @pytest.mark.asyncio
    async def test_missing_ffmpeg_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(transcriber.settings, "transcription_provider", "groq")
        monkeypatch.setattr(
            type(transcriber.settings), "api_key_for", lambda self, provider: "key"
        )
        monkeypatch.setattr(transcriber, "_ffmpeg_available", lambda: False)
        assert await transcriber.transcribe_media(_media_file(tmp_path)) == ""


class TestTranscribeMediaHappyPath:
    @staticmethod
    def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(transcriber.settings, "transcription_provider", "groq")
        monkeypatch.setattr(
            type(transcriber.settings), "api_key_for", lambda self, provider: "key"
        )
        monkeypatch.setattr(transcriber, "_ffmpeg_available", lambda: True)

    @pytest.mark.asyncio
    async def test_joins_segment_transcripts_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable(monkeypatch)
        monkeypatch.setattr(
            transcriber, "_extract_audio_segments",
            lambda src, out: [Path("seg_000.mp3"), Path("seg_001.mp3")],
        )
        texts = {"seg_000.mp3": "bagian satu", "seg_001.mp3": "bagian dua"}

        async def fake_transcribe(client, segment):
            return texts[segment.name]

        monkeypatch.setattr(transcriber, "_transcribe_segment", fake_transcribe)

        assert await transcriber.transcribe_media(_media_file(tmp_path)) == (
            "bagian satu\nbagian dua"
        )

    @pytest.mark.asyncio
    async def test_failing_segment_does_not_lose_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable(monkeypatch)
        monkeypatch.setattr(
            transcriber, "_extract_audio_segments",
            lambda src, out: [Path("seg_000.mp3"), Path("seg_001.mp3")],
        )

        async def flaky(client, segment):
            if segment.name == "seg_000.mp3":
                raise RuntimeError("API 500")
            return "bagian dua"

        monkeypatch.setattr(transcriber, "_transcribe_segment", flaky)

        assert await transcriber.transcribe_media(_media_file(tmp_path)) == "bagian dua"

    @pytest.mark.asyncio
    async def test_no_segments_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._enable(monkeypatch)
        monkeypatch.setattr(transcriber, "_extract_audio_segments", lambda src, out: [])
        assert await transcriber.transcribe_media(_media_file(tmp_path)) == ""


class TestParseMedia:
    @pytest.mark.asyncio
    async def test_builds_single_text_element(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_transcribe(path):
            return "isi rekaman kuliah"

        monkeypatch.setattr(transcriber, "transcribe_media", fake_transcribe)
        f = _media_file(tmp_path)

        elements = await transcriber.parse_media(f, content_id="kka-minggu-14")

        assert len(elements) == 1
        el = elements[0]
        assert el.element_type == ElementType.TEXT
        assert el.content == "isi rekaman kuliah"
        assert el.source_file == f.name
        assert el.content_id == "kka-minggu-14"

    @pytest.mark.asyncio
    async def test_empty_transcript_yields_no_elements(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_transcribe(path):
            return "   "

        monkeypatch.setattr(transcriber, "transcribe_media", fake_transcribe)
        assert await transcriber.parse_media(_media_file(tmp_path)) == []
