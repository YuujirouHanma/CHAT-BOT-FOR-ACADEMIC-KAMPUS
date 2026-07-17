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


# Level gaya jawaban — ditambahkan ke SYSTEM_PROMPT. Untuk mahasiswa pelosok yang
# bingung, "sederhana" menjelaskan seperti ke pemula; "detail" untuk yang mau mendalam.
ANSWER_LEVEL_INSTRUCTIONS = {
    "sederhana": (
        "\n\nGAYA JAWABAN: Jelaskan dengan bahasa SANGAT SEDERHANA, seolah menjelaskan "
        "ke siswa SMA atau orang yang baru pertama belajar. Hindari istilah teknis; jika "
        "terpaksa memakainya, jelaskan artinya dengan kata sehari-hari. Pakai kalimat "
        "pendek dan analogi yang mudah dibayangkan. Tetap sertakan [Sumber N]."
    ),
    "standar": "",
    "detail": (
        "\n\nGAYA JAWABAN: Jelaskan secara MENDALAM untuk mahasiswa tingkat lanjut. "
        "Sertakan detail penting, istilah teknis yang tepat beserta nuansanya, dan bila "
        "relevan tunjukkan keterkaitan antar konsep. Tetap sertakan [Sumber N]."
    ),
}


def build_system_prompt(level: str | None = None) -> str:
    """SYSTEM_PROMPT plus an optional answer-level style instruction.

    Unknown/None level → standard prompt (no extra instruction)."""
    return SYSTEM_PROMPT + ANSWER_LEVEL_INSTRUCTIONS.get(level or "standar", "")


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


# --- Starter questions: template questions generated from a material's content ---
STARTER_SYSTEM_PROMPT = (
    "Anda adalah tutor yang membuat pertanyaan pembuka untuk mahasiswa yang "
    "BARU membuka sebuah materi kuliah dan belum tahu harus bertanya apa. "
    "Berdasarkan isi materi yang diberikan, buat tepat 5 pertanyaan pembuka yang: "
    "(1) mencakup konsep-konsep utama materi, (2) sederhana dan mengundang, "
    "cocok untuk mahasiswa yang baru belajar, (3) bisa dijawab dari materi itu. "
    "Jawab HANYA dalam JSON valid berbentuk array string, tanpa markdown, tanpa "
    'penjelasan. Contoh: ["Apa itu ...?", "Bagaimana cara ...?", "Mengapa ...?"]'
)


# --- Stage 5: Follow-up question generation (separate from the main answer) ---
# General-purpose: works for ANY subject. Enforces 3 DIFFERENT question TYPES so
# the recommendations are varied instead of three near-duplicates.
FOLLOWUP_SYSTEM_PROMPT = (
    "Anda adalah tutor yang membuat pertanyaan lanjutan untuk membantu mahasiswa belajar, "
    "untuk materi kuliah APA PUN (berlaku umum, bukan satu bidang tertentu). "
    "Buat tepat 3 pertanyaan lanjutan yang relevan dengan pertanyaan awal dan jawaban final. "
    "WAJIB: ketiganya harus dari TIPE yang BERBEDA — pilih 3 tipe berbeda dari daftar ini: "
    "definisi/konsep, contoh/penerapan, perbandingan/perbedaan, sebab-akibat/alasan, "
    "langkah/proses, atau analisis/evaluasi. "
    "Pertanyaan harus singkat, natural, dan mendorong pemahaman lebih dalam. "
    "Jawab HANYA dalam JSON valid berbentuk array string, tanpa markdown, tanpa penjelasan. "
    'Contoh (tipe berbeda): ["Apa yang dimaksud dengan ...?", '
    '"Bagaimana penerapan ... dalam kasus nyata?", "Apa perbedaan ... dan ...?"]'
)


# --- Quiz: multiple-choice questions generated from a material's content ---
QUIZ_SYSTEM_PROMPT = (
    "Anda adalah pembuat soal kuis untuk mahasiswa, untuk materi kuliah APA PUN "
    "(berlaku umum). Berdasarkan isi materi yang diberikan, buat 5 soal PILIHAN GANDA yang: "
    "(1) menguji pemahaman konsep utama materi, (2) punya TEPAT 4 opsi jawaban, "
    "(3) hanya SATU jawaban benar, (4) sertakan penjelasan singkat mengapa jawaban itu benar. "
    "Jawab HANYA dalam JSON valid berbentuk array objek, tanpa markdown, tanpa teks lain. "
    "Setiap objek berbentuk: "
    '{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "..."} '
    "di mana answer_index adalah indeks (0-3) dari opsi yang benar."
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


def parse_followup_json(raw: str, limit: int = 3) -> list[str]:
    """Parse a JSON array of question strings, capped at `limit` items."""
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

    return [s for s in suggestions if s][:limit]


def parse_quiz_json(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array of multiple-choice quiz items.

    Keeps only well-formed items: non-empty question, exactly 4 string options,
    and an answer_index within range. Returns [] if nothing valid is found.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    data: Any = None
    try:
        data = json.loads(clean)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if not isinstance(data, list):
        return []

    quiz: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = item.get("options")
        answer_index = item.get("answer_index")
        if not question or not isinstance(options, list) or len(options) != 4:
            continue
        if not isinstance(answer_index, int) or not (0 <= answer_index < 4):
            continue
        quiz.append({
            "question": question,
            "options": [str(o) for o in options],
            "answer_index": answer_index,
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return quiz