"""API request and response schemas.

Validasi di sini adalah ambang pertama sistem: seluruh masukan dari luar
dianggap tidak tepercaya sampai lolos skema. Pola (`pattern`) dipakai — bukan
sekadar panjang maksimum — pada field yang nilainya akan menjadi bagian dari
nama berkas atau filter penyimpanan, sehingga bentuk yang berbahaya ditolak di
ambang, jauh sebelum menyentuh disk atau Qdrant.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- pola bersama ---
# Identitas yang ikut menjadi nama berkas / kunci filter: tanpa garis miring,
# tanpa titik ganda, tanpa spasi.
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
# Nama berkas materi. Titik diizinkan (ekstensi), tetapi pemisah path tidak.
_FILENAME_PATTERN = r"^[^/\\\x00]{1,255}$"
_TENANT_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,62}$"


class TenantScopedRequest(BaseModel):
    """Body yang boleh menyertakan `tenant_id` untuk dicocokkan.

    PENTING bagi tim pemanggil: field ini TIDAK menentukan data siapa yang
    diakses. Cakupan selalu diturunkan dari kunci API. Nilai di sini hanya
    dicocokkan, dan ketidakcocokan ditolak dengan 403 — gunanya menangkap kunci
    yang tertukar di sisi klien sedini mungkin, bukan memilih tenant.
    """
    model_config = ConfigDict(extra="forbid")

    tenant_id: str | None = Field(default=None, pattern=_TENANT_PATTERN)


# --- Chat ---
class QueryRequest(TenantScopedRequest):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    content_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    source_filter: str | None = Field(default=None, max_length=255, pattern=_FILENAME_PATTERN)
    model: str | None = Field(default=None, max_length=64)  # registry key; None = default
    level: Literal["sederhana", "standar", "detail"] | None = None  # kedalaman jawaban
    # Gaya belajar: mengganti system prompt (CARA menjawab) dan alat yang
    # dipakai LLM. Lihat GET /learning-styles. Berbeda dari `level`.
    style: str | None = Field(default=None, max_length=32)
    # Guided navigation: chatbot menuntun mahasiswa memilih mata kuliah → minggu →
    # materi lewat pilihan yang bisa diklik. Matikan (False) kalau klien ingin
    # perilaku tanya-jawab murni tanpa tanya-balik.
    guided: bool = True
    # Diisi klien saat mahasiswa mengklik tombol pilihan. Opsional — mengirim
    # label tombol sebagai `question` biasa juga sudah dikenali.
    course_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    week: int | None = Field(default=None, ge=1, le=52)
    # Identitas mahasiswa dari aplikasi pemanggil. Dipakai menyaring daftar
    # riwayat percakapan miliknya sendiri. Data pribadi — disimpan dalam bentuk
    # tersandi, tidak pernah apa adanya (lihat src/security/crypto.py).
    student_id: str | None = Field(default=None, max_length=128)
    # Beberapa minggu sekaligus, mis. saat mahasiswa menyiapkan ujian. Bila diisi,
    # nilainya menang atas `week`. `week` dipertahankan agar klien lama tetap jalan.
    weeks: list[int] | None = Field(default=None, max_length=52)


class SourceInfo(BaseModel):
    index: int
    source_file: str | None = None
    page_number: int | None = None
    element_type: str | None = None
    content_id: str | None = None
    rerank_score: float | None = None


class Attachment(BaseModel):
    """Berkas turunan dari sebuah jawaban, mis. notebook dari gaya belajar praktik.

    Dibentuk deterministik dari isi jawaban — tanpa panggilan LLM tambahan —
    sehingga isinya dijamin sama dengan yang dibaca mahasiswa di layar.
    """
    kind: Literal["notebook"]
    filename: str
    content: str          # isi berkas apa adanya (JSON .ipynb)
    mime: str = "application/x-ipynb+json"


class ChoiceInfo(BaseModel):
    """Satu pilihan yang bisa diklik mahasiswa di chat."""
    label: str
    value: str
    kind: Literal["course", "week", "material", "style", "question", "quiz"]


class ChatContext(BaseModel):
    """Konteks yang sedang aktif — dipakai klien menampilkan breadcrumb."""
    course_id: str | None = None
    course_name: str | None = None
    weeks: list[int] = Field(default_factory=list)
    # Minggu pertama dari `weeks`, dipertahankan agar klien yang sudah membaca
    # `week` (satu nilai) tidak rusak saat mahasiswa memilih beberapa minggu.
    week: int | None = None
    content_id: str | None = None
    source_file: str | None = None
    style: str | None = None
    # Topik yang dibahas pada minggu terpilih, disimpulkan dari materi terindex.
    topic: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    recommendations: list[str]
    session_id: str
    interaction_id: str | None = None
    # "answer" = `answer` adalah jawaban atas pertanyaan.
    # "choices" = `answer` adalah pertanyaan balik chatbot, dan `choices` berisi
    # pilihan yang harus ditampilkan sebagai tombol.
    mode: Literal["answer", "choices"] = "answer"
    step: Literal[
        "course", "week", "material", "style", "question", "answer", "quiz",
    ] = "answer"
    choices: list[ChoiceInfo] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    context: ChatContext = Field(default_factory=ChatContext)


class FeedbackRequest(TenantScopedRequest):
    interaction_id: str = Field(max_length=128, pattern=_ID_PATTERN)
    rating: Literal["membantu", "cukup", "tidak_membantu"]
    issues: list[str] = Field(default_factory=list, max_length=20)
    comment: str | None = Field(default=None, max_length=1000)


# --- Upload / Indexing ---
class IndexResponse(BaseModel):
    source_file: str
    elements_parsed: int
    chunks_created: int
    points_stored: int
    content_id: str | None


# --- Browse ---
class ContentListResponse(BaseModel):
    contents: list[str]


class FileInfo(BaseModel):
    filename: str
    content_id: str
    size_bytes: int
    size_display: str
    category: str          # document, video, audio, image, other
    is_indexable: bool
    indexed: bool


class FileListResponse(BaseModel):
    content_id: str
    files: list[FileInfo]
    total_files: int
    total_indexable: int
    total_indexed: int


# --- Batch ---
class BatchIndexRequest(TenantScopedRequest):
    content_id: str = Field(pattern=_ID_PATTERN)


class BatchFileResult(BaseModel):
    filename: str
    content_id: str
    chunks_created: int
    points_stored: int
    status: str


class BatchIndexSummary(BaseModel):
    total_files: int
    indexed: int
    skipped: int
    errors: list[str]
    results: list[BatchFileResult]


# --- Catalog (guided navigation: mata kuliah → minggu → materi) ---
class CourseInfo(BaseModel):
    course_id: str
    course_name: str


class CourseListResponse(BaseModel):
    courses: list[CourseInfo]


class WeekListResponse(BaseModel):
    course_id: str
    weeks: list[int]


class MaterialInfo(BaseModel):
    source_file: str
    content_id: str | None = None


class MaterialListResponse(BaseModel):
    course_id: str
    week: int
    materials: list[MaterialInfo]


class StarterQuestionsResponse(BaseModel):
    content_id: str
    source_file: str
    questions: list[str]


# --- Quiz ---
class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str = ""


class QuizResponse(BaseModel):
    content_id: str
    source_file: str
    questions: list[QuizQuestion]


class QuizSubmitRequest(TenantScopedRequest):
    answers: list[int] = Field(
        max_length=100,
        description="Indeks opsi yang dipilih per soal (urut sesuai soal)",
    )
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    student_id: str | None = Field(default=None, max_length=128)


class QuizResultItem(BaseModel):
    question: str
    options: list[str]
    your_answer: int | None = None
    correct_answer: int
    is_correct: bool
    explanation: str = ""


class QuizSubmitResponse(BaseModel):
    content_id: str
    source_file: str
    total: int
    correct: int
    score: float                 # 0-100
    attempt_id: str
    results: list[QuizResultItem]


# --- Models (untuk UI switching) ---
class ModelInfo(BaseModel):
    key: str
    label: str
    provider: str
    vision: bool


class ModelListResponse(BaseModel):
    default: str
    models: list[ModelInfo]


# --- Riwayat percakapan ---
class TranscriptMessage(BaseModel):
    """Satu pesan seperti yang tampil di layar, termasuk langkah navigasi."""
    role: Literal["user", "assistant"]
    content: str
    at: str | None = None
    mode: Literal["answer", "choices"] | None = None
    step: str | None = None
    choices: list[ChoiceInfo] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)
    interaction_id: str | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    student_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int = 0
    context: ChatContext = Field(default_factory=ChatContext)


class ConversationListResponse(BaseModel):
    total: int
    conversations: list[ConversationSummary]


class ConversationDetail(ConversationSummary):
    transcript: list[TranscriptMessage] = Field(default_factory=list)


# --- Gaya belajar ---
class LearningStyleInfo(BaseModel):
    key: str
    label: str
    description: str
    produces_notebook: bool = False


class LearningStyleListResponse(BaseModel):
    default: str
    styles: list[LearningStyleInfo]


# --- Error ---
class ErrorResponse(BaseModel):
    detail: str
