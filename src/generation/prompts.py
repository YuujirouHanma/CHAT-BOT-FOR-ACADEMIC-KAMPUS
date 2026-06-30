"""Prompt templates and context formatting for generation.

Critical design choice: when a retrieved chunk has a `raw_html` table or
`image_base64`, we inject the RAW DATA into the LLM context, not the summary.
The summary was only used for retrieval; the actual answer should be grounded
in the original data.

Each retrieved chunk becomes a `[Sumber N]` block the LLM can cite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.schemas import ElementType

SYSTEM_PROMPT = """Anda adalah asisten pembelajaran untuk mahasiswa. Tugas Anda:

1. Menjawab pertanyaan HANYA berdasarkan konteks materi yang diberikan.
2. Jika konteks tidak cukup, katakan dengan jujur: "Materi yang tersedia tidak mencakup informasi tersebut."
3. Untuk pertanyaan yang melibatkan tabel atau gambar, baca data mentah yang disertakan dan kutip angka secara presisi.
4. Sebutkan sumber di setiap klaim penting dengan format [Sumber N], di mana N adalah nomor sumber yang Anda kutip.
5. Gunakan bahasa Indonesia yang jelas dan akademis. Jangan mengulang pertanyaan.
6. Jika ada angka, formula, atau definisi penting, sajikan dengan tepat."""


USER_PROMPT_TEMPLATE = """KONTEKS MATERI:
{context}

PERTANYAAN MAHASISWA:
{question}

JAWABAN (sertakan [Sumber N] untuk setiap klaim):"""


# --- Stage 1: Query decomposition (enrich query before retrieval) ---
DECOMPOSE_SYSTEM_PROMPT = (
    "Anda adalah asisten akademik. Diberikan sebuah pertanyaan dari mahasiswa, "
    "uraikan pertanyaan tersebut menjadi komponen terstruktur berikut (jawab dalam JSON):\n"
    "{\n"
    '  "topik_utama": "<topik atau materi yang paling relevan>",\n'
    '  "konsep_kunci": ["<konsep 1>", "<konsep 2>", ...],\n'
    '  "tipe_pertanyaan": "<definisi | penjelasan | contoh | perhitungan | perbandingan | lainnya>",\n'
    '  "query_diperkaya": "<versi pertanyaan yang lebih eksplisit dan informatif untuk pencarian materi>"\n'
    "}\n"
    "Jawab HANYA dengan JSON valid. Jangan coba menjawab pertanyaannya."
)


# --- Stage 5: Follow-up question generation (separate from the main answer) ---
FOLLOWUP_SYSTEM_PROMPT = (
    "Anda adalah tutor yang membuat pertanyaan lanjutan untuk membantu mahasiswa belajar. "
    "Buat tepat 3 pertanyaan lanjutan yang relevan dengan pertanyaan awal dan jawaban final. "
    "Pertanyaan harus singkat, natural, dan mendorong pemahaman lebih dalam. "
    "Jawab HANYA dalam JSON valid berbentuk array string, tanpa markdown, tanpa penjelasan. "
    'Contoh: ["Pertanyaan 1?", "Pertanyaan 2?", "Pertanyaan 3?"]'
)


@dataclass(frozen=True)
class FormattedContext:
    """Result of formatting retrieval results into LLM-ready content.

    Attributes:
        text_block: The textual context to put in the user message.
        image_payloads: List of {data, mime, source_idx} for images that
            should be sent as separate vision content blocks.
    """

    text_block: str
    image_payloads: list[dict[str, Any]]


def format_retrieval_results(results: list[dict]) -> FormattedContext:
    """Convert reranker output into formatted LLM context.

    For TEXT chunks: include the chunk text inline.
    For TABLE chunks: include raw_html so the LLM reads actual numbers.
    For IMAGE chunks: include the description inline AND emit a separate
        image payload that the caller can attach as a vision content block.

    Each source gets a 1-indexed citation tag [Sumber N].
    """
    text_parts: list[str] = []
    image_payloads: list[dict[str, Any]] = []

    for idx, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        element_type = payload.get("element_type", "text")
        source_file = payload.get("source_file", "unknown")
        page = payload.get("page_number")
        page_str = f", hal. {page}" if page is not None else ""
        header = f"[Sumber {idx}] {source_file}{page_str}"

        if element_type == ElementType.TABLE.value and payload.get("raw_html"):
            block = (
                f"{header} (TABEL — data asli HTML, baca angka langsung dari sini):\n"
                f"{payload['raw_html']}"
            )
        elif element_type == ElementType.IMAGE.value:
            description = payload.get("text", "")
            block = (
                f"{header} (GAMBAR — deskripsi dan gambar asli disertakan):\n"
                f"Deskripsi: {description}"
            )
            if payload.get("image_base64"):
                image_payloads.append(
                    {
                        "source_idx": idx,
                        "image_base64": payload["image_base64"],
                    }
                )
        else:
            text = payload.get("text", "")
            block = f"{header}:\n{text}"

        text_parts.append(block)

    return FormattedContext(
        text_block="\n\n---\n\n".join(text_parts),
        image_payloads=image_payloads,
    )


def build_user_prompt(question: str, context: FormattedContext) -> str:
    return USER_PROMPT_TEMPLATE.format(
        context=context.text_block,
        question=question.strip(),
    )


def parse_decompose_json(raw: str, fallback_question: str) -> dict[str, Any]:
    """Parse Stage 1 decomposition output. Falls back to the raw question on failure."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "topik_utama": fallback_question,
        "konsep_kunci": [fallback_question],
        "tipe_pertanyaan": "lainnya",
        "query_diperkaya": fallback_question,
    }


def parse_followup_json(raw: str) -> list[str]:
    """Parse Stage 5 follow-up output into a list of up to 3 questions."""
    clean = raw.replace("```json", "").replace("```", "").strip()

    suggestions: list[str] = []
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            suggestions = [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    suggestions = [str(x).strip() for x in data if str(x).strip()]
            except Exception:
                pass

    if not suggestions:
        suggestions = [
            re.sub(r"^\s*[\-\d\.\)\]]+\s*", "", line).strip()
            for line in clean.splitlines()
            if line.strip()
        ]

    return [s for s in suggestions if s][:3]