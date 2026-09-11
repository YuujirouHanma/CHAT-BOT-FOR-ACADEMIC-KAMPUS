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
    kind: Literal[
        "course", "week", "material", "style", "question", "quiz", "back",
    ]


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


# --- Validasi dosen atas jawaban AI ---
class ValidationAnswerInfo(BaseModel):
    """Satu jawaban AI beserta status validasinya."""
    interaction_id: str
    at: str | None = None
    question: str = ""
    answer: str = ""
    sources: list[SourceInfo] = Field(default_factory=list)
    content_id: str | None = None
    course_id: str | None = None
    session_id: str | None = None
    # Kosong berarti belum dinilai siapa pun.
    verdict: Literal["sesuai", "perlu_perbaikan", "tidak_sesuai"] | None = None
    catatan: str = ""
    dosen_id: str = ""


class ValidationListResponse(BaseModel):
    total: int
    answers: list[ValidationAnswerInfo]


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["sesuai", "perlu_perbaikan", "tidak_sesuai"]
    # Wajib bila putusannya bukan "sesuai" — divalidasi di lapisan domain.
    # Menandai salah tanpa menjelaskan salahnya di mana tidak menolong siapa pun.
    catatan: str | None = Field(default=None, max_length=4000)
    dosen_id: str | None = Field(default=None, max_length=128)
    course_id: str | None = Field(default=None, max_length=128)
    content_id: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=64)


class ValidationVerdictResponse(BaseModel):
    interaction_id: str
    verdict: str
    catatan: str = ""
    at: str


class ValidationStatsResponse(BaseModel):
    total_jawaban: int
    sudah_dinilai: int
    belum_dinilai: int
    rincian: dict[str, int]
    # None (bukan 0) bila belum ada yang dinilai — "belum diukur" tidak boleh
    # terbaca sebagai "akurasinya nol".
    akurasi: float | None = None


# --- Evaluasi berkala (kuis / ETS / EAS) ---
class EvaluationPlanInfo(BaseModel):
    """Rencana sebuah evaluasi: kapan, jenisnya apa, mencakup minggu mana."""
    week: int
    kind: str
    label: str
    weeks_covered: list[int]
    range_text: str
    counts: dict[str, int]
    total: int


class EvaluationScheduleResponse(BaseModel):
    evaluation_weeks: list[int]
    plans: list[EvaluationPlanInfo]


class _EvaluationTarget(BaseModel):
    """Bagian permintaan yang menentukan evaluasi mana yang dimaksud."""
    model_config = ConfigDict(extra="forbid")

    course_id: str = Field(min_length=1, max_length=128)
    # Salah satu wajib: `week` memakai jadwal bawaan, `weeks` rentang bebas.
    week: int | None = Field(default=None, ge=1, le=52)
    weeks: list[int] | None = Field(default=None, max_length=52)
    kind: Literal["kuis", "ets", "eas"] | None = None
    # Komposisi soal khusus; kosong = memakai komposisi bawaan jenis tersebut.
    counts: dict[str, int] | None = None
    model: str | None = Field(default=None, max_length=64)
    tenant_id: str | None = Field(default=None, max_length=64)


class EvaluationQuestionsRequest(_EvaluationTarget):
    pass


class EvaluationItemInfo(BaseModel):
    """Satu soal seperti yang dilihat mahasiswa.

    Tanpa kunci jawaban maupun rubrik — keduanya hanya ada di sisi server.
    """
    index: int
    type: str
    question: str
    options: list[str] = Field(default_factory=list)
    starter_code: str = ""


class EvaluationQuestionsResponse(BaseModel):
    course_id: str
    plan: EvaluationPlanInfo
    items: list[EvaluationItemInfo]


class EvaluationSubmitRequest(_EvaluationTarget):
    # Urut sesuai soal. Objektif berupa indeks opsi (int); isian/esai/koding
    # berupa teks. None berarti tidak dijawab.
    #
    # `bool` WAJIB berdiri di depan `int` pada union ini. Soal benar/salah
    # secara alami dikirim klien sebagai `true`/`false`, dan tanpa `bool` di
    # depan pydantic memaksanya menjadi int di batas skema — `true` menjadi 1,
    # yang di urutan opsi ["Benar", "Salah"] justru berarti "Salah". Penilaian
    # jadi terbalik sebelum jawabannya sempat sampai ke `grade_objective`.
    answers: list[bool | int | str | None] = Field(
        default_factory=list, max_length=40,
    )


class GradedItemInfo(BaseModel):
    index: int
    type: str
    question: str
    your_answer: int | str | None = None
    correct_answer: int | str | None = None
    # None = belum dapat dipastikan mesin; menunggu tinjauan dosen.
    is_correct: bool | None = None
    score: float
    explanation: str = ""
    feedback: str = ""


class EvaluationSubmitResponse(BaseModel):
    course_id: str
    plan: EvaluationPlanInfo
    total: int
    benar: int
    skor: float                      # 0-100
    per_jenis: dict[str, dict] = Field(default_factory=dict)
    perlu_tinjauan: int = 0
    items: list[GradedItemInfo]


# --- Livecode: latihan koding ---
class _LivecodeTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    course_id: str = Field(min_length=1, max_length=128)
    weeks: list[int] = Field(min_length=1, max_length=52)
    model: str | None = Field(default=None, max_length=64)
    tenant_id: str | None = Field(default=None, max_length=64)


class LivecodeExercisesRequest(_LivecodeTarget):
    count: int = Field(default=3, ge=1, le=10)


class LivecodeExerciseInfo(BaseModel):
    """Latihan seperti yang dilihat mahasiswa.

    Tanpa rubrik dan tanpa keluaran kasus uji — kalau ikut dikirim, mahasiswa
    dapat menuliskan jawabannya langsung tanpa menulis programnya.
    """
    exercise_id: str
    title: str
    prompt: str
    language: str = "python"
    starter_code: str = ""
    expected_behavior: str = ""
    required_function: str = ""
    forbidden_names: list[str] = Field(default_factory=list)
    example_inputs: list = Field(default_factory=list)


class LivecodeExercisesResponse(BaseModel):
    course_id: str
    exercises: list[LivecodeExerciseInfo]


class LivecodeSubmitRequest(_LivecodeTarget):
    exercise_id: str = Field(min_length=1, max_length=256)
    code: str = Field(max_length=20_000)
    student_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    count: int = Field(default=3, ge=1, le=10)


class CodeFindingInfo(BaseModel):
    severity: Literal["galat", "peringatan", "info"]
    message: str
    line: int | None = None


class CodeErrorInfo(BaseModel):
    baris: int | None = None
    masalah: str
    akibat: str = ""


class LivecodeSubmitResponse(BaseModel):
    exercise_id: str
    submission_id: str = ""
    lulus: bool
    skor: float
    ringkasan: str = ""
    # Temuan analisis statis (sintaks, fungsi wajib, konstruksi terlarang).
    temuan: list[CodeFindingInfo] = Field(default_factory=list)
    # Disebut lebih dulu: mahasiswa pemula yang hanya menerima daftar kesalahan
    # cenderung berhenti mencoba.
    benar: list[str] = Field(default_factory=list)
    keliru: list[CodeErrorInfo] = Field(default_factory=list)
    # Mengarahkan, bukan memberi kode jadi.
    petunjuk: list[str] = Field(default_factory=list)
    perlu_tinjauan_dosen: bool = False


class LivecodeSubmissionInfo(BaseModel):
    submission_id: str
    at: str | None = None
    exercise_id: str
    student_id: str | None = None
    lulus: bool | None = None
    skor: float | None = None
    code: str = ""


class LivecodeSubmissionListResponse(BaseModel):
    total: int
    submissions: list[LivecodeSubmissionInfo]


class LivecodeStatsResponse(BaseModel):
    exercise_id: str
    total_kiriman: int
    lulus: int
    rasio_lulus: float | None = None


# --- Admin: pengelolaan tenant ---
class AdminTenantSummary(BaseModel):
    tenant_id: str
    name: str = ""
    status: str = "active"
    created_at: str | None = None
    active_keys: int = 0


class AdminTenantKeyInfo(BaseModel):
    """Metadata sebuah kunci. Rahasianya tidak pernah dapat ditampilkan lagi."""
    key_id: str
    label: str = ""
    scopes: list[str] = Field(default_factory=list)
    allowed_courses: list[str] | None = None
    created_at: str | None = None
    revoked_at: str | None = None
    last_used_at: str | None = None


class AdminTenantDetail(AdminTenantSummary):
    quota: dict[str, int] = Field(default_factory=dict)
    keys: list[AdminTenantKeyInfo] = Field(default_factory=list)


class AdminTenantListResponse(BaseModel):
    total: int
    tenants: list[AdminTenantSummary]


class AdminTenantCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=2, max_length=63)
    name: str | None = Field(default=None, max_length=200)
    quota: dict[str, int] | None = None


class AdminIssueKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, max_length=200)
    # Kosong = hak pembaca (aplikasi mahasiswa).
    scopes: list[str] | None = None
    # ABAC: bila diisi, kunci ini hanya boleh menyentuh mata kuliah tersebut.
    allowed_courses: list[str] | None = None


class AdminIssuedKeyResponse(BaseModel):
    tenant_id: str
    key_id: str
    # Muncul SEKALI seumur hidup kunci.
    api_key: str
    peringatan: str


class AdminStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["active", "suspended"]


class AdminPurgeResponse(BaseModel):
    tenant_id: str
    conversations_deleted: int
    catatan: str


# --- Error ---
class ErrorResponse(BaseModel):
    detail: str
