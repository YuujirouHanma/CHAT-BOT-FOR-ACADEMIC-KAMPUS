"""API request and response schemas.

Validasi di sini adalah ambang pertama sistem: seluruh masukan dari luar
dianggap tidak tepercaya sampai lolos skema. Pola (`pattern`) dipakai — bukan
sekadar panjang maksimum — pada field yang nilainya akan menjadi bagian dari
nama berkas atau filter penyimpanan, sehingga bentuk yang berbahaya ditolak di
ambang, jauh sebelum menyentuh disk atau Qdrant.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import model_registry

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
    # `protected_namespaces` dikosongkan karena portal kampus menyebut modelnya
    # `model_provider`/`model_name`, sementara pydantic memesan awalan `model_`
    # untuk medannya sendiri. Tanpa ini setiap impor memuntahkan peringatan —
    # dan peringatan yang selalu muncul adalah peringatan yang berhenti dibaca.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    content_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    source_filter: str | None = Field(default=None, max_length=255, pattern=_FILENAME_PATTERN)
    model: str | None = Field(default=None, max_length=64)  # registry key; None = default
    # Nama Inggris karena itu yang dipakai portal kampus pada tombolnya.
    # Nama lama (sederhana/standar/detail) masih dikenali di dalam, tetapi
    # tidak lagi diiklankan: "standar" dan "sederhana" meminta hal yang sama.
    # Tiga yang pertama dipakai sekarang; empat sisanya nama lama yang masih
    # mungkin tersimpan di sesi. Semua diselesaikan lewat ALIAS_KEDALAMAN.
    level: Literal[
        "simple", "detailed", "academic",
        "sederhana", "standar", "detail", "akademis",
    ] | None = None  # kedalaman jawaban
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

    # --- nama-nama yang dipakai portal kampus --------------------------------
    # Lima medan di bawah ini adalah kosakata LMS untuk hal yang namanya sudah
    # ada di atas. Diterima, bukan ditolak, karena `extra="forbid"` membuat
    # setiap permintaan dari portal dibalas 422 — bukan "sebagian fitur tidak
    # jalan", melainkan chat mati total. Kanonisasinya terjadi di metode di
    # bawah, sehingga sisa kode hanya mengenal satu nama untuk satu hal.
    section_ids: list[int] | None = Field(
        default=None, max_length=64,
        description="Nama lain `weeks`, kosakata LMS yang dipakai portal kampus.",
    )
    model_provider: str | None = Field(default=None, max_length=32)
    model_name: str | None = Field(default=None, max_length=255)
    student_name: str | None = Field(default=None, max_length=128)
    user_id: str | int | None = Field(
        default=None, description="Nama lain `student_id` di portal kampus.",
    )

    def active_weeks(self) -> list[int]:
        """Minggu yang berlaku, menyatukan tiga bentuk yang dikirim klien.

        `weeks` (terpandu) dan `week` (klien lama) memakai kosakata katalog,
        sementara `section_ids` memakai kosakata LMS — ketiganya menunjuk hal
        yang sama, dan satu-satunya tempat mereka disatukan adalah di sini.
        """
        if self.weeks:
            return sorted({int(w) for w in self.weeks})
        if self.week:
            return [self.week]
        return sorted({int(w) for w in self.section_ids or []})

    def kunci_model(self) -> str | None:
        """Satu kunci registry dari dua cara klien menyebut model.

        Portal mengirim penyedia dan nama terpisah, dan `model_name` berisi id
        milik PENYEDIA ("qwen/qwen3.7-plus") karena itu yang tersimpan di tabel
        `generation_models`. Registry memakai kunci pendeknya sendiri
        ("qwen3.7-plus"). Penerjemahannya ada di `model_registry.kunci_untuk`.

        Sebelumnya id penyedia diteruskan apa adanya sebagai kunci, sehingga
        pencariannya gagal dan permintaan dilayani model BAWAAN tanpa satu pun
        tanda ke pemanggil. `model` menang bila keduanya terkirim.

        Bila penerjemahan gagal, nilai asli diteruskan — bukan None — supaya
        peringatan "Unknown model" di `LLMGenerator._resolve` tetap menyebut apa
        yang sebenarnya diminta pemanggil.
        """
        if self.model:
            return self.model
        nama = (self.model_name or "").strip()
        if not nama:
            return None
        return model_registry.kunci_untuk(self.model_provider, nama) or nama

    def identitas_mahasiswa(self) -> str | None:
        """Pengenal mahasiswa, apa pun nama medan yang dipakai pemanggil.

        Dipakai menyaring daftar riwayat percakapan miliknya sendiri. Tanpa
        penyatuan ini, portal yang mengirim `user_id` menghasilkan percakapan
        tanpa pemilik — dan daftar riwayat setiap mahasiswa jadi kosong.
        """
        if self.student_id:
            return self.student_id
        return str(self.user_id) if self.user_id not in (None, "") else None


class SourceInfo(BaseModel):
    index: int
    source_file: str | None = None
    page_number: int | None = None
    element_type: str | None = None
    content_id: str | None = None
    rerank_score: float | None = None
    # Isi potongan yang dikutip, apa adanya dari indeks.
    #
    # Ada karena .pptx dan .docx tidak punya penampil bawaan peramban: panel
    # sitasi hanya bisa menyebutkan "halaman 7, buka berkasnya sendiri", dan
    # mahasiswa yang ingin memeriksa satu kalimat harus mengunduh satu dek
    # enam puluh slide. Dengan medan ini panelnya dapat menampilkan bagian yang
    # BENAR-BENAR dikutip tanpa penampil pihak ketiga — artinya tanpa mengirim
    # materi kelas ke layanan luar.
    #
    # TIDAK dicatat ke log HITL; lihat `_sumber_tanpa_teks` di src/pipeline.py.
    text: str | None = None


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
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    course_id: str | None = Field(default=None, max_length=128)
    # Salah satu wajib: `week` memakai jadwal bawaan, `weeks` rentang bebas.
    week: int | None = Field(default=None, ge=1, le=52)
    weeks: list[int] | None = Field(default=None, max_length=52)
    kind: Literal["kuis", "ets", "eas"] | None = None
    # Komposisi soal khusus; kosong = memakai komposisi bawaan jenis tersebut.
    counts: dict[str, int] | None = None
    model: str | None = Field(default=None, max_length=64)
    tenant_id: str | None = Field(default=None, max_length=64)

    # --- nama lain yang dipakai portal kampus --------------------------------
    # Kosakata LMS untuk hal yang namanya sudah ada di atas. Diterima, bukan
    # ditolak: `extra="forbid"` membuat permintaan dari portal dibalas 422, dan
    # di layar mahasiswa itu muncul sebagai "layanan tidak tersedia".
    content_id: str | None = Field(default=None, max_length=128)
    section_ids: list[int] | None = Field(default=None, max_length=52)
    model_provider: str | None = Field(default=None, max_length=32)
    model_name: str | None = Field(default=None, max_length=255)

    def mata_kuliah(self) -> str:
        """Mata kuliah yang dimaksud, apa pun nama medan yang dipakai pemanggil."""
        return self.course_id or self.content_id or ""

    def minggu(self) -> list[int]:
        """Minggu yang berlaku, menyatukan kedua kosakata."""
        if self.weeks:
            return sorted({int(w) for w in self.weeks})
        return sorted({int(w) for w in self.section_ids or []})

    def kunci_model(self) -> str | None:
        """Satu kunci registry dari dua cara klien menyebut model.

        Lihat `QueryRequest.kunci_model` untuk alasan penerjemahannya.
        """
        if self.model:
            return self.model
        nama = (self.model_name or "").strip()
        if not nama:
            return None
        return model_registry.kunci_untuk(self.model_provider, nama) or nama


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
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    course_id: str | None = Field(default=None, max_length=128)
    weeks: list[int] | None = Field(default=None, max_length=52)
    model: str | None = Field(default=None, max_length=64)
    tenant_id: str | None = Field(default=None, max_length=64)

    # --- nama lain yang dipakai portal kampus --------------------------------
    # Kosakata LMS untuk hal yang namanya sudah ada di atas. Diterima, bukan
    # ditolak: `extra="forbid"` membuat permintaan dari portal dibalas 422, dan
    # di layar mahasiswa itu muncul sebagai "layanan tidak tersedia".
    content_id: str | None = Field(default=None, max_length=128)
    section_ids: list[int] | None = Field(default=None, max_length=52)
    model_provider: str | None = Field(default=None, max_length=32)
    model_name: str | None = Field(default=None, max_length=255)

    def mata_kuliah(self) -> str:
        """Mata kuliah yang dimaksud, apa pun nama medan yang dipakai pemanggil."""
        return self.course_id or self.content_id or ""

    def minggu(self) -> list[int]:
        """Minggu yang berlaku, menyatukan kedua kosakata."""
        if self.weeks:
            return sorted({int(w) for w in self.weeks})
        return sorted({int(w) for w in self.section_ids or []})

    def kunci_model(self) -> str | None:
        """Satu kunci registry dari dua cara klien menyebut model.

        Lihat `QueryRequest.kunci_model` untuk alasan penerjemahannya.
        """
        if self.model:
            return self.model
        nama = (self.model_name or "").strip()
        if not nama:
            return None
        return model_registry.kunci_untuk(self.model_provider, nama) or nama


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

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _bersihkan_session_id(nilai: str | None) -> str | None:
    if nilai is None or not nilai.strip():
        return None
    nilai = nilai.strip()
    if not _SESSION_ID_RE.match(nilai):
        raise ValueError("session_id tidak valid")
    return nilai


# --- Guided Learning, latihan, dan konsep ----------------------------------
# Dipindahkan apa adanya dari layanan tim back-end, bukan ditulis ulang.
# Bentuk permintaan dan balasan harus sama persis di kedua pemasangan: satu
# medan yang berbeda namanya sudah cukup membuat portal kampus yang sama
# tidak dapat memanggil keduanya.
class UserLearningProfile(BaseModel):
    """Learning preferences captured during onboarding."""
    student_name: str = Field(default="Mahasiswa", max_length=64)
    # Kunci gaya belajar bersama. Nama VARK lama (auditory, reading,
    # kinesthetic) masih dikenali di dalam, tetapi tidak lagi diiklankan.
    # Enam kunci pertama adalah yang dipakai sekarang. Empat terakhir nama VARK
    # lama; tetap DITERIMA karena profil mahasiswa yang sudah tersimpan di basis
    # data portal masih memakainya, dan menolaknya berarti setiap pengguna lama
    # dibalas 422. Keduanya diselesaikan ke satu gaya lewat ALIAS_GAYA.
    learning_style: Literal[
        "narrative", "visual", "practical", "structured", "concise", "socratic",
        "auditory", "reading", "kinesthetic",
    ] = "narrative"
    explanation_depth: Literal["simple", "detailed", "academic"] = "detailed"


class GuidedQueryRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    """Request for the guided /guided/ask endpoint."""
    template_id: str = Field(max_length=64)
    topic: str = Field(min_length=1, max_length=500,
                       description="Topik spesifik yang ingin dipelajari")
    content_id: str | None = Field(default=None, max_length=128)
    section_id: int | None = None
    section_ids: list[int] | None = Field(
        default=None, max_length=64,
        description="Minggu (section) yang diaktifkan; kosong = seluruh kelas",
    )
    session_id: str | None = None
    model_provider: Literal["openai", "groq", "huggingface", "ollama", "openrouter"] | None = None
    model_name: str | None = Field(default=None, max_length=255)
    user_profile: UserLearningProfile = Field(default_factory=UserLearningProfile)
    class_name: str = Field(default="", max_length=128)
    section_name: str = Field(default="Minggu Aktif", max_length=128)
    user_id: int | None = None

    _session_id_sah = field_validator("session_id")(_bersihkan_session_id)

    def active_section_ids(self) -> list[int] | None:
        """Minggu yang berlaku, menyatukan bentuk lama dan baru.

        ``section_id`` tunggal sudah dikirim frontend sejak sebelum fitur ini
        ada (dan dulu diabaikan backend). Ia tetap dihormati sebagai cadangan
        supaya klien versi lama tidak tiba-tiba kehilangan pembatasan minggu.
        """
        if self.section_ids:
            return self.section_ids
        if self.section_id:
            return [self.section_id]
        return None


class TemplateInfo(BaseModel):
    """Single prompt template metadata."""
    template_id: str
    label: str
    icon: str
    description: str
    question_pattern: str
    tags: list[str]
    requires_topic_input: bool


class GuidedTemplatesResponse(BaseModel):
    """Response for GET /guided/templates."""
    templates: list[TemplateInfo]
    section_name: str
    class_name: str
    material_topics: list[str]


class GuidedQueryResponse(BaseModel):
    """Response for POST /guided/ask."""
    answer: str
    sources: list[SourceInfo]
    recommendations: list[str]
    session_id: str
    interaction_id: str | None = None
    template_id: str
    personalized_label: str  # e.g. "Penjelasan untuk Andi (visual learner)"


# --- Chat history (HITL log review) ---


class ExerciseRequest(BaseModel):
    """Request for POST /guided/exercise."""
    topic: str = Field(min_length=1, max_length=500,
                       description="Topik yang ingin dilatih")
    content_id: str | None = None
    class_name: str = Field(default="", max_length=128)
    section_ids: list[int] | None = Field(
        default=None, max_length=64,
        description="Minggu (section) yang diaktifkan; kosong = seluruh kelas",
    )


class ExerciseResponse(BaseModel):
    """Response for POST /guided/exercise.

    ``content`` adalah isi berkas siap unduh — notebook (.ipynb) atau lembar
    kerja (.md). Frontend cukup membungkusnya jadi Blob; formatnya ditentukan
    model dari sifat materinya, jadi tidak diasumsikan di sisi klien.
    """
    format: str            # 'notebook' | 'lembar_kerja'
    filename: str
    mime_type: str
    title: str
    intro: str
    task_count: int
    content: str
    sources: list[SourceInfo]


class ConceptSourceInfo(BaseModel):
    """Di mana sebuah konsep diajarkan, dalam satuan berkas dan rentang halaman."""
    source_file: str
    halaman_awal: int | None = None
    halaman_akhir: int | None = None


class ConceptInfo(BaseModel):
    nama: str
    deskripsi: str = ""
    urutan: int = 0
    jumlah_potongan: int = 0
    sumber: list[ConceptSourceInfo] = []


class FileCoverage(BaseModel):
    """Berapa banyak potongan sebuah berkas yang masuk ke setidaknya satu konsep.

    Berkas dengan ``tercakup`` nol berarti dianggap tidak relevan oleh ekstraksi.
    Itu bisa benar (berkas nyasar) atau salah (materi yang terlewat), dan hanya
    dosen yang bisa membedakannya — karena itu angkanya ikut dikirim.

    Tiga angka, bukan dua, karena ada dua penyusutan berurutan dan keduanya
    perlu dibedakan: dari ``total`` potongan sebuah berkas, hanya ``dibaca``
    yang disodorkan ke model (penyusunan peta dibatasi anggaran konteks), dan
    dari yang dibaca itu hanya ``tercakup`` yang akhirnya dikutip sebuah konsep.
    Menggabungkan keduanya membuat materi yang tidak pernah dibaca tidak dapat
    dibedakan dari materi yang dibaca lalu dinilai tidak relevan.
    """
    source_file: str
    total: int
    dibaca: int
    tercakup: int


class ConceptPreviewResponse(BaseModel):
    """Response for GET /concepts/preview."""
    content_id: str
    section_id: int | None = None
    jumlah_chunk: int = 0
    chunk_dipakai: int = 0
    concepts: list[ConceptInfo] = []
    cakupan: list[FileCoverage] = []


class ConceptQuestionSourceInput(BaseModel):
    """Sumber konsep sebagaimana tersimpan di ``concept_source``."""
    source_file: str
    halaman_awal: int | None = None
    halaman_akhir: int | None = None


class ConceptQuestionConceptInput(BaseModel):
    """Satu konsep yang sudah tersimpan, dikirim frontend untuk dibuatkan soal.

    ``concept_id`` ikut dikirim dan dikembalikan apa adanya supaya frontend bisa
    menautkan soal ke barisnya tanpa mencocokkan nama — nama konsep bisa diedit
    dosen kapan saja, id-nya tidak.
    """
    concept_id: int | None = None
    nama: str = Field(..., min_length=1, max_length=200)
    deskripsi: str = Field(default="", max_length=600)
    sumber: list[ConceptQuestionSourceInput] = []


class ConceptQuestionsRequest(BaseModel):
    """Request for POST /concepts/questions."""
    content_id: str = Field(..., min_length=1)
    section_id: int | None = None
    class_name: str = Field(default="Mata Kuliah", max_length=160)
    section_name: str = Field(default="Minggu Ini", max_length=160)
    concepts: list[ConceptQuestionConceptInput] = Field(..., min_length=1, max_length=12)
    # Batas atas 10: soal per konsep bertumpu pada materi yang sama, dan uji
    # yang lebih panjang justru dikerjakan asal-asalan — lihat komentar
    # SOAL_PER_KONSEP_MAKS di src/indexing/questions.py.
    jumlah_soal: int = Field(default=2, ge=1, le=10)


class QuestionOptionInfo(BaseModel):
    teks: str
    benar: bool = False


class ConceptQuestionInfo(BaseModel):
    """Satu soal pilihan ganda beserta kunci dan pembahasannya.

    Kunci ikut dikirim karena penerimanya adalah dosen yang harus menilai
    apakah soalnya benar. Jalur mahasiswa tidak memakai endpoint ini.
    """
    concept_id: int | None = None
    nama_konsep: str = ""
    tipe: str = "ingatan"
    pertanyaan: str
    opsi: list[QuestionOptionInfo] = []
    pembahasan: str = ""
    urutan: int = 0


class ConceptQuestionsResponse(BaseModel):
    """Response for POST /concepts/questions."""
    content_id: str
    section_id: int | None = None
    questions: list[ConceptQuestionInfo] = []
    konsep_gagal: list[str] = []


# --- Unggah latar belakang, berkas terindeks, riwayat ----------------------
class UploadJobResponse(BaseModel):
    """Response for POST /documents/upload/background — indexing belum selesai."""
    job_id: str
    status: str


class UploadJobStatusResponse(BaseModel):
    """Response for GET /documents/upload/background/{job_id}."""
    job_id: str
    status: str  # pending | running | done | error
    result: IndexResponse | None = None
    error: str | None = None


# --- Browse ---


class ChatHistoryItem(BaseModel):
    interaction_id: str | None = None
    timestamp: str | None = None
    session_id: str | None = None
    content_id: str | None = None
    student_name: str | None = None
    user_id: int | None = None
    question: str
    answer: str
    topik_utama: str | None = None
    tipe_pertanyaan: str | None = None
    sources: list[SourceInfo] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    elapsed_seconds: float | None = None


class ChatHistoryResponse(BaseModel):
    items: list[ChatHistoryItem]
    total: int
    limit: int
    offset: int


class HitungNama(BaseModel):
    """Satu nama beserta berapa kali ia muncul (topik, berkas, jenis)."""
    nama: str
    jumlah: int


class PertanyaanBerulang(BaseModel):
    """Pertanyaan yang ditanyakan oleh beberapa MAHASISWA BERBEDA.

    `mahasiswa` menghitung orang, bukan pengulangan: satu mahasiswa yang
    bertanya lima kali sedang kesulitan sendiri, sedangkan lima mahasiswa yang
    bertanya hal sama menunjuk ke materinya.
    """
    pertanyaan: str
    mahasiswa: int


class ChatHistorySummaryResponse(BaseModel):
    """Ringkasan riwayat chat satu kelas untuk dosen.

    Semua angka di sini dihitung di kode (src/hitl/ringkasan.py); `ringkasan`
    adalah satu-satunya medan yang ditulis model. Bila pemanggilan model gagal,
    `ringkasan` kosong dan sisanya tetap terkirim — dosen kehilangan narasinya,
    bukan datanya.
    """
    ringkasan: str = ""
    total_interaksi: int = 0
    # Berapa yang BENAR-BENAR tersimpan, sebelum dibatasi `maks`. Dikirim supaya
    # dosen tahu ringkasannya menutup sebagian atau seluruh riwayat.
    total_tersimpan: int = 0
    mahasiswa_unik: int = 0
    sejak: str | None = None
    hingga: str | None = None
    tak_terjawab: int = 0
    topik_teratas: list[HitungNama] = Field(default_factory=list)
    materi_teratas: list[HitungNama] = Field(default_factory=list)
    tipe_pertanyaan: list[HitungNama] = Field(default_factory=list)
    contoh_tak_terjawab: list[str] = Field(default_factory=list)
    pertanyaan_berulang: list[PertanyaanBerulang] = Field(default_factory=list)
    dari_cache: bool = False


class IndexedFileInfo(BaseModel):
    """Satu berkas sebagaimana benar-benar tersimpan di indeks."""
    content_id: str | None = None
    section_id: int | None = None
    source_file: str
    chunks: int = 0


class IndexedFilesResponse(BaseModel):
    """Response for GET /documents/indexed."""
    files: list[IndexedFileInfo] = []
    total_files: int = 0
    total_chunks: int = 0
