"""Prompt templates and context formatting for generation.

Critical design choice: when a retrieved chunk has a `raw_html` table or
`image_base64`, we inject the RAW DATA into the LLM context, not the summary.
The summary was only used for retrieval; the actual answer should be grounded
in the original data.

Each retrieved chunk becomes a `[Sumber N]` block the LLM can cite.
"""
from __future__ import annotations

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