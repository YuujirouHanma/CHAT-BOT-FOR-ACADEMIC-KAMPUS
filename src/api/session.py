"""In-memory session manager for chat context."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

TTL_SECONDS = 3600


@dataclass
class Session:
    session_id: str
    content_id: str | None = None
    source_filter: str | None = None
    # Konteks guided navigation (mata kuliah → minggu → materi). Disimpan terpisah
    # dari content_id karena mahasiswa bisa memilih mata kuliah dulu, jauh sebelum
    # sebuah materi (dan content_id-nya) ditentukan.
    course_id: str | None = None
    course_name: str | None = None
    week: int | None = None
    # Langkah yang sedang ditanyakan chatbot. Nilainya melonggarkan penafsiran
    # pesan berikutnya: saat menanyakan minggu, "3" boleh berarti minggu 3.
    awaiting: str | None = None
    # Kuis yang sedang dikerjakan di dalam chat: {questions, answers, index}.
    # Disimpan di session karena kuis berlangsung lintas beberapa pesan.
    quiz: dict | None = None
    history: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)

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
        week: int | None = None,
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
            self.week = None
            self.content_id = None
            self.source_filter = None
        elif course_name and self.course_name is None:
            self.course_name = course_name

        if week is not None and week != self.week:
            self.week = week
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
        week: int | None,
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
        self.week = week
        self.content_id = content_id
        self.source_filter = source_filter
        self.touch()

    def reset_guided(self) -> None:
        """Kembali ke langkah awal tanpa kehilangan riwayat percakapan."""
        self.course_id = None
        self.course_name = None
        self.week = None
        self.content_id = None
        self.source_filter = None
        self.awaiting = None
        self.touch()

    def add_turn(self, role: str, content: str, max_turns: int = 10) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]
        self.touch()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        s = Session(session_id=str(uuid.uuid4()))
        self._sessions[s.session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        if s.is_expired():
            del self._sessions[session_id]
            return None
        s.touch()
        return s

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id:
            s = self.get(session_id)
            if s:
                return s
        return self.create()


session_store = SessionStore()
