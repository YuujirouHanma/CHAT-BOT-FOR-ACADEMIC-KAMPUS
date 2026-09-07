"""Session percakapan: konteks belajar, riwayat, dan penyimpanannya.

Session hidup di memori selama dipakai, dan setiap perubahan ditulis ke disk
lewat `conversation_store`. Karena itu `session_id` sekaligus menjadi identitas
percakapan yang dapat dibuka kembali berhari-hari kemudian — klien tidak perlu
mengenal konsep kedua.

Setiap session terikat pada satu tenant. `session_id` adalah UUID yang dikirim
bolak-balik oleh klien, jadi ia harus diperlakukan sebagai nilai yang bisa
ditebak atau dicuri: tanpa ikatan tenant, siapa pun yang memegang sebuah
`session_id` dapat memuat percakapan itu lewat kunci API tenant mana pun. Cache
di memori pun dikunci per tenant, bukan hanya per `session_id`.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.storage import conversation_store
from src.tenancy import require_tenant_id

TTL_SECONDS = 3600


@dataclass
class Session:
    session_id: str
    # Pemilik percakapan ini. Wajib — tanpa nilai bawaan, supaya membuat session
    # tanpa tenant menjadi galat di titik pemanggilan, bukan kebocoran diam-diam.
    tenant_id: str
    content_id: str | None = None
    source_filter: str | None = None
    # Konteks guided navigation (mata kuliah → minggu → materi). Disimpan terpisah
    # dari content_id karena mahasiswa bisa memilih mata kuliah dulu, jauh sebelum
    # sebuah materi (dan content_id-nya) ditentukan.
    course_id: str | None = None
    course_name: str | None = None
    # Bisa lebih dari satu: mahasiswa yang menyiapkan ujian kerap belajar
    # beberapa minggu sekaligus ("minggu 3 dan 4").
    weeks: list[int] = field(default_factory=list)
    # Topik yang dibahas pada minggu terpilih, disimpulkan dari isi materi.
    topic: str | None = None
    # Gaya belajar pilihan mahasiswa; menentukan system prompt LLM.
    style: str | None = None
    # Langkah yang sedang ditanyakan chatbot. Nilainya melonggarkan penafsiran
    # pesan berikutnya: saat menanyakan minggu, "3" boleh berarti minggu 3.
    awaiting: str | None = None
    # Kuis yang sedang dikerjakan di dalam chat: {questions, answers, index}.
    # Disimpan di session karena kuis berlangsung lintas beberapa pesan.
    quiz: dict | None = None
    # Konteks untuk LLM: hanya tanya-jawab sungguhan, tanpa klik menu.
    history: list[dict] = field(default_factory=list)
    # Untuk ditampilkan ulang: SEMUA yang muncul di layar, termasuk navigasi.
    transcript: list[dict] = field(default_factory=list)
    student_id: str | None = None
    title: str = ""
    created_at: str = field(default_factory=conversation_store.now_iso)
    updated_at: str = field(default_factory=conversation_store.now_iso)
    last_active: float = field(default_factory=time.monotonic)

    # --- riwayat yang ditampilkan ---
    def record(self, role: str, content: str, **extra: object) -> None:
        """Catat satu pesan ke transkrip tampilan.

        Judul percakapan diambil dari pesan berisi pertama mahasiswa — sapaan
        seperti "halo" dilewati karena tidak memberi tahu isi percakapan.
        """
        entry: dict = {"role": role, "content": content,
                       "at": conversation_store.now_iso()}
        entry.update({k: v for k, v in extra.items() if v not in (None, [], "")})
        self.transcript.append(entry)
        if role == "user" and not self.title and len(content.strip()) > 4:
            self.title = conversation_store.make_title(content)
        self.updated_at = conversation_store.now_iso()
        self.touch()

    def to_record(self) -> dict:
        """Bentuk yang disimpan ke disk."""
        return {
            "conversation_id": self.session_id,
            "tenant_id": self.tenant_id,
            "student_id": self.student_id,
            "title": self.title or "Percakapan baru",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context": {
                "course_id": self.course_id,
                "course_name": self.course_name,
                "weeks": self.weeks,
                "content_id": self.content_id,
                "source_file": self.source_filter,
                "style": self.style,
                "topic": self.topic,
            },
            "transcript": self.transcript,
            "history": self.history,
        }

    @classmethod
    def from_record(cls, d: dict, *, tenant_id: str) -> Session:
        """Bangun ulang session dari berkas tersimpan.

        Kuis yang sedang berjalan sengaja TIDAK dipulihkan: melanjutkan kuis
        berhari-hari kemudian di tengah soal lebih membingungkan daripada
        memulainya lagi.

        `tenant_id` diambil dari kredensial pemanggil, bukan dari isi berkas —
        berkas hanya berhak menentukan isinya sendiri, tidak menentukan siapa
        yang boleh membacanya.
        """
        ctx = d.get("context") or {}
        return cls(
            session_id=d.get("conversation_id", str(uuid.uuid4())),
            tenant_id=require_tenant_id(tenant_id, operation="Session.from_record"),
            student_id=d.get("student_id"),
            title=d.get("title", ""),
            created_at=d.get("created_at") or conversation_store.now_iso(),
            updated_at=d.get("updated_at") or conversation_store.now_iso(),
            course_id=ctx.get("course_id"),
            course_name=ctx.get("course_name"),
            weeks=list(ctx.get("weeks") or []),
            content_id=ctx.get("content_id"),
            source_filter=ctx.get("source_file"),
            style=ctx.get("style"),
            topic=ctx.get("topic"),
            transcript=list(d.get("transcript") or []),
            history=list(d.get("history") or []),
        )

    def persist(self) -> None:
        conversation_store.save(self.to_record(), tenant_id=self.tenant_id)

    def start_quiz(self, questions: list[dict]) -> None:
        self.quiz = {"questions": questions, "answers": [], "index": 0}
        self.touch()

    def answer_quiz(self, option_index: int) -> None:
        if not self.quiz:
            return
        self.quiz["answers"].append(option_index)
        self.quiz["index"] += 1
        self.touch()

    @property
    def quiz_done(self) -> bool:
        return bool(self.quiz) and self.quiz["index"] >= len(self.quiz["questions"])

    def current_quiz_question(self) -> dict | None:
        if not self.quiz or self.quiz_done:
            return None
        return self.quiz["questions"][self.quiz["index"]]

    def stop_quiz(self) -> None:
        self.quiz = None
        self.touch()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_active) > TTL_SECONDS

    def set_guided(
        self,
        course_id: str | None = None,
        course_name: str | None = None,
        weeks: list[int] | None = None,
        content_id: str | None = None,
        source_filter: str | None = None,
    ) -> None:
        """Terapkan konteks guided. Pilihan yang lebih tinggi mereset yang di bawahnya.

        Ganti mata kuliah → minggu dan materi ikut dibuang; ganti minggu → materi
        dibuang. Tanpa ini mahasiswa yang pindah mata kuliah masih terkunci pada
        materi lama dan jawabannya jadi ngawur.
        """
        if course_id is not None and course_id != self.course_id:
            self.course_id = course_id
            self.course_name = course_name
            self.weeks = []
            self.content_id = None
            self.source_filter = None
        elif course_name and self.course_name is None:
            self.course_name = course_name

        if weeks and sorted(weeks) != self.weeks:
            self.weeks = sorted(set(weeks))
            self.content_id = None
            self.source_filter = None

        if content_id is not None:
            self.content_id = content_id
        if source_filter is not None:
            self.source_filter = source_filter
        self.touch()

    def apply_guided(
        self,
        course_id: str | None,
        course_name: str | None,
        weeks: list[int] | None,
        content_id: str | None,
        source_filter: str | None,
    ) -> None:
        """Timpa konteks apa adanya — `None` berarti DIKOSONGKAN.

        Bedanya dengan `set_guided`: di sini pemanggil (hasil resolusi guided)
        adalah sumber kebenaran, jadi None harus menghapus. `set_guided` dipakai
        untuk input klien yang parsial, di mana None berarti "tidak disebut".
        Tanpa pemisahan ini, "ganti mata kuliah" tidak pernah benar-benar reset.
        """
        self.course_id = course_id
        self.course_name = course_name
        self.weeks = sorted(set(weeks)) if weeks else []
        self.content_id = content_id
        self.source_filter = source_filter
        self.touch()

    def reset_guided(self) -> None:
        """Kembali ke langkah awal tanpa kehilangan riwayat percakapan."""
        self.course_id = None
        self.course_name = None
        self.weeks = []
        self.topic = None
        self.content_id = None
        self.source_filter = None
        self.awaiting = None
        self.touch()

    def step_back_to(self, step: str) -> None:
        """Mundur ke satu langkah tertentu, membuang HANYA pilihan sesudahnya.

        Berbeda dari `reset_guided()` yang mengosongkan semuanya. Mahasiswa yang
        salah memilih minggu ingin mengganti minggunya saja; memaksanya memilih
        ulang mata kuliah membuat satu salah klik terasa seperti hukuman.

        Riwayat percakapan tidak disentuh — yang dibatalkan adalah pilihan
        navigasinya, bukan apa yang sudah dipelajari.
        """
        if step == "course":
            self.course_id = None
            self.course_name = None
        if step in ("course", "week"):
            self.weeks = []
            self.topic = None
        if step in ("course", "week", "material"):
            self.content_id = None
            self.source_filter = None
        if step in ("course", "week", "material", "style"):
            self.style = None
        # Kuis yang sedang berjalan ikut dihentikan: soalnya melekat pada materi
        # yang barusan ditinggalkan, jadi melanjutkannya tidak lagi masuk akal.
        self.quiz = None
        self.awaiting = step
        self.touch()

    def add_turn(self, role: str, content: str, max_turns: int = 10) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]
        self.touch()


class SessionStore:
    """Session aktif di memori, dengan disk sebagai sumber jangka panjang.

    Kedaluwarsa di memori hanya membebaskan RAM — percakapannya tetap ada di
    disk, dan permintaan berikutnya dengan `session_id` yang sama akan
    memuatnya kembali. Itulah yang membuat riwayat dapat dibuka lagi.

    Kunci cache adalah `tenant_id` + `session_id`, bukan `session_id` saja.
    Dengan kunci tunggal, tenant B yang mengirim `session_id` milik tenant A
    akan mendapat session itu langsung dari memori — melewati pemisahan
    direktori di disk yang seharusnya menahannya.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], Session] = {}

    def create(self, *, tenant_id: str, student_id: str | None = None) -> Session:
        tenant_id = require_tenant_id(tenant_id, operation="SessionStore.create")
        s = Session(
            session_id=str(uuid.uuid4()), tenant_id=tenant_id, student_id=student_id,
        )
        self._sessions[(tenant_id, s.session_id)] = s
        return s

    def get(self, session_id: str, *, tenant_id: str) -> Session | None:
        """Session dari memori, atau dimuat ulang dari disk bila sudah lewat TTL."""
        tenant_id = require_tenant_id(tenant_id, operation="SessionStore.get")
        kunci = (tenant_id, session_id)

        s = self._sessions.get(kunci)
        if s is not None and not s.is_expired():
            s.touch()
            return s
        if s is not None:
            del self._sessions[kunci]

        record = conversation_store.load(session_id, tenant_id=tenant_id)
        if record is None:
            return None
        pulih = Session.from_record(record, tenant_id=tenant_id)
        self._sessions[(tenant_id, pulih.session_id)] = pulih
        return pulih

    def get_or_create(
        self,
        session_id: str | None,
        *,
        tenant_id: str,
        student_id: str | None = None,
    ) -> Session:
        """Ambil session yang ada, atau buat baru.

        `session_id` yang tidak dikenali menghasilkan session BARU, bukan galat:
        dari sisi tenant lain, sebuah id milik orang lain memang tidak pernah ada.
        Membedakan "tidak ada" dari "ada tetapi bukan milikmu" akan memberi tahu
        penebak bahwa sebuah id itu nyata.
        """
        if session_id:
            s = self.get(session_id, tenant_id=tenant_id)
            if s:
                if student_id and not s.student_id:
                    s.student_id = student_id
                return s
        return self.create(tenant_id=tenant_id, student_id=student_id)

    def drop(self, session_id: str, *, tenant_id: str) -> None:
        """Buang dari memori. Penghapusan berkasnya urusan pemanggil."""
        self._sessions.pop((tenant_id, session_id), None)


session_store = SessionStore()
