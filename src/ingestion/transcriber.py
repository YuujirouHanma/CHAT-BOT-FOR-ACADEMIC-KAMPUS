"""Transcribe video/audio materials so their content becomes searchable.

Lecture recordings can't be fed to an LLM directly — the practical path is
audio → text, then index the transcript like any other document.

Pipeline:
    video/audio → ffmpeg (strip video, mono 16kHz, split into segments)
                → Whisper via an OpenAI-compatible API (Groq / OpenAI)
                → one TEXT ParsedElement carrying the full transcript

Everything degrades gracefully: if transcription is disabled, ffmpeg is missing,
or the provider key is unset, we log a clear reason and return no elements —
the upload still succeeds, the file just isn't searchable.

Requires `ffmpeg` on PATH (see Dockerfile).
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from src.config import settings
from src.ingestion.validators import validate_file
from src.schemas import ElementType, ParsedElement
from src.utils.logger import logger


def is_media(file_path: Path) -> bool:
    """True for video or audio files (which need transcription, not parsing)."""
    ext = file_path.suffix.lower().lstrip(".")
    return ext in settings.video_extensions_set or ext in settings.audio_extensions_set


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio_segments(src: Path, out_dir: Path) -> list[Path]:
    """Strip video, downmix to mono 16kHz low-bitrate mp3, split into segments.

    Segmenting keeps every chunk well under provider upload limits and lets a
    long lecture transcribe incrementally. Returns segment files in order.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", str(src),
        "-vn",                  # drop any video stream
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz is what speech models expect
        "-b:a", "32k",
        "-f", "segment",
        "-segment_time", str(settings.transcription_segment_seconds),
        str(out_dir / "seg_%03d.mp3"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg failed for {}: {}", src.name, (proc.stderr or "")[-300:]
        )
        return []
    return sorted(out_dir.glob("seg_*.mp3"))


async def _transcribe_segment(client: AsyncOpenAI, segment: Path) -> str:
    response = await client.audio.transcriptions.create(
        model=settings.transcription_model,
        file=(segment.name, segment.read_bytes()),
        language=settings.transcription_language,
    )
    return (getattr(response, "text", "") or "").strip()


async def transcribe_media(file_path: Path) -> str:
    """Return the transcript of a media file, or "" when unavailable."""
    provider = settings.transcription_provider
    if provider == "disabled":
        logger.info("Transcription disabled; skipping {}", file_path.name)
        return ""
    if not settings.api_key_for(provider):
        logger.warning(
            "No API key for transcription provider '{}'; skipping {}",
            provider, file_path.name,
        )
        return ""
    if not _ffmpeg_available():
        logger.warning(
            "ffmpeg not found — cannot transcribe {}. Install ffmpeg in the "
            "runtime image to make video/audio searchable.", file_path.name,
        )
        return ""

    client = AsyncOpenAI(
        api_key=settings.api_key_for(provider),
        base_url=settings.base_url_for(provider),
    )

    with tempfile.TemporaryDirectory() as tmp:
        segments = await asyncio.to_thread(
            _extract_audio_segments, file_path, Path(tmp)
        )
        if not segments:
            logger.warning("No audio extracted from {}", file_path.name)
            return ""

        parts: list[str] = []
        for segment in segments:
            try:
                text = await _transcribe_segment(client, segment)
            except Exception as exc:
                # One bad segment shouldn't lose the whole lecture.
                logger.warning(
                    "Transcription failed for {} segment {}: {}",
                    file_path.name, segment.name, exc,
                )
                continue
            if text:
                parts.append(text)
        segment_count = len(segments)

    transcript = "\n".join(parts).strip()
    logger.info(
        "Transcribed {}: {} segment(s) → {} chars",
        file_path.name, segment_count, len(transcript),
    )
    return transcript


async def parse_media(
    file_path: Path, content_id: str | None = None
) -> list[ParsedElement]:
    """Transcribe a media file into a single TEXT element for the index.

    Returns [] when there is nothing to index (transcription unavailable or
    the recording produced no speech) — the caller treats it like an empty doc.
    """
    validate_file(file_path)
    logger.info("Transcribing media: {}", file_path.name)

    transcript = (await transcribe_media(file_path) or "").strip()
    if not transcript:
        return []

    return [
        ParsedElement(
            element_id=str(uuid.uuid4()),
            element_type=ElementType.TEXT,
            content=transcript,
            source_file=file_path.name,
            content_id=content_id,
        )
    ]
