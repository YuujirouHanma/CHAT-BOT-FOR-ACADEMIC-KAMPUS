"""Personalized guided-learning prompt templates.

This module provides:
- gaya belajar: dibaca dari src/learning_styles.py (satu daftar bersama)
- kedalaman penjelasan: dibaca dari src/generation/prompts.py (satu tabel bersama)
- TEMPLATE_REGISTRY         : 6 universal template categories
- build_guided_system_prompt(): builds a fully personalized system prompt
- build_guided_user_prompt() : builds the user-facing prompt from template + vars
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Learning Style Modifiers
# ---------------------------------------------------------------------------
# Tabel gaya belajar TIDAK lagi ditulis di sini. Dulu ada dua daftar untuk
# hal yang sama: daftar ini memakai model VARK, satu lagi di
# src/learning_styles.py memakai nama cara penyampaian. Mahasiswa mendapat
# cara penjelasan berbeda hanya karena masuk lewat pintu berbeda.
#
# Keduanya kini satu. Nama VARK tetap dikenali lewat ALIAS_GAYA, jadi profil
# mahasiswa yang sudah tersimpan tidak perlu dimigrasi.
from src import learning_styles

# Tabel kedalaman TIDAK lagi ditulis di sini. Dulu ada dua tabel yang
# meminta hal yang sama dengan nama berbeda, sehingga mahasiswa mendapat
# kedalaman berlainan hanya karena masuk lewat Guided Learning atau lewat
# tanya-jawab biasa. Sekarang keduanya membaca src/generation/prompts.py,
# yang juga mengenali nama Inggris di sini lewat ALIAS_KEDALAMAN.
from src.generation.prompts import instruksi_kedalaman

# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    template_id: str
    label: str
    icon: str
    description: str
    # Pola pertanyaan dengan placeholder {topic} dan {class_name}
    question_pattern: str
    # Tag kategori untuk filtering
    tags: list[str] = field(default_factory=list)
    # Apakah user perlu mengisi topik secara eksplisit?
    requires_topic_input: bool = True


TEMPLATE_REGISTRY: list[PromptTemplate] = [
    PromptTemplate(
        template_id="understand_concept",
        label="Pahami Konsep",
        icon="🔍",
        description="Minta penjelasan mendalam tentang konsep utama dari materi minggu ini",
        question_pattern=(
            "Jelaskan konsep '{topic}' dari materi {class_name} minggu ini secara menyeluruh. "
            "Pastikan saya benar-benar memahami apa itu, mengapa penting, dan bagaimana cara kerjanya."
        ),
        tags=["konsep", "pemahaman", "dasar"],
        requires_topic_input=True,
    ),
    PromptTemplate(
        template_id="real_world_example",
        label="Contoh Nyata",
        icon="🧪",
        description="Lihat bagaimana konsep materi diterapkan di kehidupan nyata atau dunia industri",
        question_pattern=(
            "Berikan contoh nyata dan konkret dari penerapan '{topic}' dalam kehidupan sehari-hari "
            "atau dunia industri. Sertakan minimal 2 skenario yang berbeda dan jelaskan mengapa "
            "contoh tersebut relevan dengan konsep di materi {class_name}."
        ),
        tags=["aplikasi", "contoh", "industri"],
        requires_topic_input=True,
    ),
    PromptTemplate(
        template_id="connect_concepts",
        label="Hubungkan Konsep",
        icon="🔗",
        description="Pahami hubungan dan keterkaitan antar konsep dalam materi minggu ini",
        question_pattern=(
            "Dalam materi {class_name} minggu ini, bagaimana hubungan antara '{topic}' "
            "dengan konsep-konsep lain yang ada? Jelaskan ketergantungan, persamaan, "
            "dan perbedaan yang perlu saya pahami."
        ),
        tags=["relasi", "koneksi", "peta konsep"],
        requires_topic_input=True,
    ),
    PromptTemplate(
        template_id="quiz_me",
        label="Kuis Pemahaman",
        icon="❓",
        description="Uji pemahaman kamu dengan soal latihan yang dibuat dari materi aktif",
        question_pattern=(
            "Buat 5 pertanyaan latihan tentang '{topic}' dari materi {class_name} minggu ini. "
            "Variasikan tingkat kesulitan (mudah, sedang, sulit). Sertakan kunci jawaban "
            "singkat di bawah setiap pertanyaan."
        ),
        tags=["latihan", "kuis", "evaluasi"],
        requires_topic_input=True,
    ),
    PromptTemplate(
        template_id="structured_summary",
        label="Ringkasan Terstruktur",
        icon="📊",
        description="Dapatkan ringkasan terstruktur dari seluruh materi minggu ini",
        question_pattern=(
            "Buat ringkasan komprehensif dari materi {class_name} minggu ini, "
            "khususnya tentang '{topic}'. Gunakan format terstruktur: poin-poin utama, "
            "konsep kunci, dan hal-hal yang paling penting untuk diingat."
        ),
        tags=["ringkasan", "review", "rekap"],
        requires_topic_input=False,
    ),
    PromptTemplate(
        template_id="step_by_step",
        label="Panduan Langkah",
        icon="🚀",
        description="Dapatkan panduan langkah demi langkah untuk memahami atau mengerjakan sesuatu",
        question_pattern=(
            "Jelaskan langkah demi langkah bagaimana cara '{topic}' bekerja atau dilakukan "
            "berdasarkan materi {class_name} minggu ini. Mulai dari langkah paling awal "
            "hingga akhir, dengan penjelasan singkat di setiap langkah."
        ),
        tags=["prosedur", "langkah", "tutorial"],
        requires_topic_input=True,
    ),
]

# Mapping template_id → PromptTemplate untuk lookup cepat
TEMPLATE_MAP: dict[str, PromptTemplate] = {t.template_id: t for t in TEMPLATE_REGISTRY}


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------

GUIDED_SYSTEM_PROMPT_BASE = """Anda adalah Sevi — AI tutor personal untuk {student_name} dalam mata kuliah {class_name}.

Materi yang sedang dipelajari minggu ini: [{section_name}]
{material_context}

ATURAN MENJAWAB:
1. Fokus HANYA pada materi minggu ini ({section_name}) dalam konteks mata kuliah {class_name}.
2. Jawab HANYA berdasarkan konteks materi yang diberikan oleh sistem retrieval.
3. Jika informasi tidak ada di materi, katakan: "Topik ini tidak tercakup dalam materi minggu ini."
4. Sebutkan sumber dengan format [Sumber N] untuk setiap klaim penting.
5. Sapa user sebagai "{student_name}" di awal respons pertama saja.

GAYA PENJELASAN (disesuaikan untuk {student_name}):
{style_instruction}

KEDALAMAN PENJELASAN:
{depth_instruction}"""


def build_guided_system_prompt(
    *,
    student_name: str,
    class_name: str,
    section_name: str,
    learning_style: str = "reading",
    explanation_depth: str = "detailed",
    material_topics: list[str] | None = None,
) -> str:
    """Build a fully personalized system prompt for guided mode.

    Args:
        student_name: First name / nickname of the student.
        class_name: Name of the course/class.
        section_name: Name of the active week/section.
        learning_style: kunci gaya belajar; nama VARK lama ikut dikenali.
        explanation_depth: 'simple', 'detailed', atau 'academic'.
        material_topics: Optional list of topic hints extracted from indexed content.

    Returns:
        Formatted system prompt string.
    """
    # .strip() karena instruksi bersama diawali baris kosong untuk ditempel
    # ke SYSTEM_PROMPT, sementara di sini ia disisipkan ke dalam kalimat.
    style_instruction = learning_styles.resolve(learning_style).system_suffix.strip()
    # .strip() karena instruksi bersama diawali baris kosong untuk
    # ditempel ke SYSTEM_PROMPT, sementara di sini ia disisipkan ke
    # dalam kalimat.
    depth_instruction = instruksi_kedalaman(explanation_depth).strip()

    if material_topics:
        topic_list = ", ".join(material_topics[:8])
        material_context = f"Topik-topik dalam materi ini mencakup: {topic_list}."
    else:
        material_context = "Jawab berdasarkan materi yang tersedia di sistem retrieval."

    return GUIDED_SYSTEM_PROMPT_BASE.format(
        student_name=student_name or "Mahasiswa",
        class_name=class_name,
        section_name=section_name,
        material_context=material_context,
        style_instruction=style_instruction,
        depth_instruction=depth_instruction,
    )


def build_guided_user_prompt(
    *,
    template_id: str,
    topic: str,
    class_name: str,
    context_block: str,
) -> str:
    """Build the user-facing prompt from a selected template.

    Args:
        template_id: ID of the template from TEMPLATE_REGISTRY.
        topic: The specific topic the student wants to explore.
        class_name: Name of the course.
        context_block: The RAG-retrieved context text.

    Returns:
        Formatted user prompt string.
    """
    template = TEMPLATE_MAP.get(template_id)
    if not template:
        # Fallback ke pertanyaan bebas
        question = topic
    else:
        question = template.question_pattern.format(
            topic=topic,
            class_name=class_name,
        )

    return (
        f"KONTEKS MATERI MINGGU INI:\n{context_block}\n\n"
        f"PERTANYAAN:\n{question}\n\n"
        f"JAWABAN (gunakan [Sumber N] untuk setiap klaim):"
    )


def get_templates_for_response() -> list[dict[str, Any]]:
    """Return all templates as serializable dicts for API responses."""
    return [
        {
            "template_id": t.template_id,
            "label": t.label,
            "icon": t.icon,
            "description": t.description,
            "question_pattern": t.question_pattern,
            "tags": t.tags,
            "requires_topic_input": t.requires_topic_input,
        }
        for t in TEMPLATE_REGISTRY
    ]
