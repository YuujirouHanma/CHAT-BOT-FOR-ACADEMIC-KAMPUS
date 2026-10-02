"""Penyimpan sesi obrolan.

Dua backend, dipilih lewat ``settings.session_backend``:

``memory``
    Dict di dalam proses. Cukup untuk pengembangan dan pengujian, tidak butuh
    layanan tambahan. Ini bawaannya, mengikuti pola ``QDRANT_MODE=local``.

``redis``
    Dipakai saat produksi. WAJIB begitu uvicorn berjalan dengan lebih dari satu
    pekerja: tiap pekerja adalah proses terpisah dengan dict sendiri, jadi
    dengan backend memori mahasiswa akan kehilangan konteks percakapan secara
    acak tergantung pekerja mana yang kebetulan menerima permintaannya —
    bug yang sangat sulit dilacak karena tampak seperti "AI kadang lupa".

Kedaluwarsa ditangani Redis lewat EXPIRE, jadi sesi terbengkalai hilang sendiri.
Backend memori tidak punya pembersih aktif dan hanya membuang sesi kedaluwarsa
saat diakses — dapat diterima karena ia memang bukan untuk produksi.

DUA LAPIS, DUA UMUR
-------------------
Backend di atas memegang sesi yang SEDANG HIDUP (TTL satu jam) dan dibagi antar
pekerja. Di bawahnya, `conversation_store` menyimpan percakapan itu ke disk
tanpa batas waktu. Keduanya bukan salinan satu sama lain melainkan dua umur yang
berbeda: begitu sesi hangus dari lapis pertama, `get()` membangunnya kembali
dari disk. Itulah yang membuat mahasiswa dapat melanjutkan belajar keesokan
harinya dengan `session_id` yang sama, dan yang membuat GET /conversations punya
isi untuk ditampilkan.

Karena itu `session_id` sekaligus menjadi identitas percakapan yang bisa dibuka
kembali — klien tidak perlu mengenal konsep kedua.

Berbeda dari versi sebelumnya, perubahan pada Session TIDAK otomatis tersimpan.
Pemanggil wajib memanggil ``await session_store.save(session)`` setelah selesai
mengubahnya. Menyembunyikan penyimpanan di balik mutasi hanya mungkin selama
penyimpanannya ada di memori proses yang sama.

MULTI-KAMPUS: setiap sesi terikat pada satu kampus, dan kuncinya di kedua backend
adalah (tenant_id, session_id) — bukan session_id saja. `session_id` bolak-balik
lewat klien, jadi diperlakukan sebagai nilai yang bisa bocor: mengirim
`session_id` milik kampus lain menghasilkan sesi BARU, bukan riwayat orang lain.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import settings
from src.storage import conversation_store
from src.tenancy import require_tenant_id
from src.utils.logger import logger

TTL_SECONDS = 3600

_REDIS_PREFIX = "ragacademic:sess:"


@dataclass
class Session:
    session_id: str
    # Wajib, tanpa nilai bawaan: sesi tanpa kampus adalah galat di titik
    # pembuatannya, bukan kebocoran diam-diam. Sesi lama di Redis yang belum
    # punya field ini gagal dibaca dan diganti sesi baru (lihat from_json).
    tenant_id: str
    content_id: str | None = None
    source_filter: str | None = None
    # --- konteks navigasi terpandu (mata kuliah -> minggu -> materi) ---
    # Disimpan terpisah dari content_id karena mahasiswa bisa memilih mata
    # kuliah lebih dulu, jauh sebelum sebuah materi (dan content_id-nya) ada.
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
    # Disimpan di sesi karena kuis berlangsung lintas beberapa pesan.
    quiz: dict | None = None
    # Konteks untuk LLM: hanya tanya-jawab sungguhan, tanpa klik menu.
    history: list[dict] = field(default_factory=list)
    # Untuk ditampilkan ulang: SEMUA yang muncul di layar, termasuk navigasi.
    transcript: list[dict] = field(default_factory=list)
    student_id: str | None = None
    title: str = ""
    created_at: str = field(default_factory=conversation_store.now_iso)
    updated_at: str = field(default_factory=conversation_store.now_iso)
    # Waktu dinding, bukan time.monotonic(): monotonic hanya bermakna di dalam
    # satu proses, sedangkan sesi ini dibaca dan ditulis oleh beberapa pekerja.
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > TTL_SECONDS

    def set_context(
        self, content_id: str | None = None,
        source_filter: str | None = None,
    ) -> None:
        if content_id != self.content_id:
            self.source_filter = None
        if content_id is not None:
            self.content_id = content_id
        if source_filter is not None:
            self.source_filter = source_filter
        self.touch()

    def add_turn(self, role: str, content: str, max_turns: int = 10) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > max_turns * 2:
            self.history = self.history[-(max_turns * 2):]
        self.touch()

    # --- riwayat yang ditampilkan ---------------------------------------------
    def record(self, role: str, content: str, **extra: object) -> None:
        """Catat satu pesan ke transkrip tampilan.

        `transcript` dan `history` sengaja berbeda isinya: transkrip memuat SEMUA
        yang tampil di layar (termasuk langkah memilih menu) supaya percakapan
        dapat ditampilkan ulang apa adanya, sedangkan `history` hanya tanya-jawab
        sungguhan karena ia dikirim ke LLM — klik menu di sana cuma memboroskan
        token dan mengaburkan alur belajar yang ingin disambung.

        Judul percakapan diambil dari pesan berisi pertama mahasiswa; sapaan
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

    # --- bentuk tersimpan di disk ---------------------------------------------
    def to_record(self) -> dict:
        """Bentuk yang disimpan `conversation_store`."""
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
        """Bangun ulang sesi dari percakapan tersimpan.

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

    @property
    def ada_isinya(self) -> bool:
        """Apakah sesi ini sudah memuat sesuatu milik mahasiswa.

        Empat hal itu persis yang membuat sebuah percakapan layak muncul lagi:
        yang tampil di layar, yang dikirim ke LLM, kuis yang sedang dikerjakan,
        dan mata kuliah yang sudah dipilih.
        """
        return bool(self.transcript or self.history or self.quiz or self.course_id)

    def persist(self) -> None:
        """Tulis percakapan ini ke penyimpanan jangka panjang.

        Sesi yang masih kosong sengaja TIDAK ditulis. Rute tutoring berbasis
        template membuat sesi lalu menyimpannya tanpa pernah mengisi transkrip;
        tanpa penjagaan ini setiap kunjungan meninggalkan satu entri "Percakapan
        baru" yang kosong di daftar riwayat mahasiswa.
        """
        if not self.ada_isinya:
            return
        conversation_store.save(self.to_record(), tenant_id=self.tenant_id)

    # --- kuis di dalam chat ----------------------------------------------------
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

    # --- konteks navigasi terpandu ---------------------------------------------
    def set_guided(
        self,
        course_id: str | None = None,
        course_name: str | None = None,
        weeks: list[int] | None = None,
        content_id: str | None = None,
        source_filter: str | None = None,
    ) -> None:
        """Terapkan konteks terpandu. Pilihan lebih tinggi mereset yang di bawah.

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

        Bedanya dengan `set_guided`: di sini pemanggil (hasil resolusi terpandu)
        adalah sumber kebenaran, jadi None harus menghapus. `set_guided` dipakai
        untuk masukan klien yang parsial, di mana None berarti "tidak disebut".
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
        """Mundur ke satu langkah, membuang HANYA pilihan sesudahnya.

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

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str | bytes) -> Session | None:
        try:
            data: dict[str, Any] = json.loads(raw)
            return Session(**data)
        except (ValueError, TypeError) as exc:
            # Sesi rusak atau berasal dari versi skema lain. Mengembalikan None
            # membuat pemanggil membuat sesi baru — kehilangan riwayat satu
            # percakapan jauh lebih ringan daripada menggagalkan permintaannya.
            logger.warning("Sesi tidak bisa dibaca, dibuat ulang: {}", exc)
            return None


def _pulihkan_dari_disk(session_id: str, tenant_id: str) -> Session | None:
    """Bangun ulang sesi yang sudah hangus dari percakapan tersimpan.

    Dipakai kedua backend lewat satu fungsi, bukan disalin ke masing-masing:
    kalau memori dan Redis punya jalur pemulihan sendiri-sendiri, keduanya akan
    menyimpang begitu bentuk rekaman berubah, dan bedanya baru ketahuan di
    produksi — satu-satunya tempat backend Redis benar-benar dipakai.

    Kedaluwarsa di lapis pertama hanya membebaskan RAM (atau kunci Redis);
    percakapannya tetap ada di disk. Inilah yang membuat riwayat dapat dibuka
    kembali berhari-hari kemudian dengan `session_id` yang sama.
    """
    record = conversation_store.load(session_id, tenant_id=tenant_id)
    if record is None:
        return None
    return Session.from_record(record, tenant_id=tenant_id)


class MemorySessionStore:
    """Dict dalam proses. Hanya benar bila proses backend cuma satu."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], Session] = {}

    async def get(self, session_id: str, *, tenant_id: str) -> Session | None:
        tenant_id = require_tenant_id(tenant_id, operation="session.get")
        kunci = (tenant_id, session_id)
        s = self._sessions.get(kunci)
        if s is not None and not s.is_expired():
            s.touch()
            return s
        if s is not None:
            del self._sessions[kunci]

        pulih = _pulihkan_dari_disk(session_id, tenant_id)
        if pulih is not None:
            self._sessions[(tenant_id, pulih.session_id)] = pulih
        return pulih

    async def get_or_create(
        self, session_id: str | None, *, tenant_id: str,
        student_id: str | None = None,
    ) -> Session:
        """Ambil sesi yang ada, atau buat baru.

        `session_id` yang tidak dikenali menghasilkan sesi BARU, bukan galat:
        dari sisi kampus lain, sebuah id milik orang lain memang tidak pernah
        ada. Membedakan "tidak ada" dari "ada tetapi bukan milikmu" akan memberi
        tahu penebak bahwa sebuah id itu nyata.
        """
        tenant_id = require_tenant_id(tenant_id, operation="session.get_or_create")
        if session_id:
            s = await self.get(session_id, tenant_id=tenant_id)
            if s:
                if student_id and not s.student_id:
                    s.student_id = student_id
                return s
        s = Session(
            session_id=str(uuid.uuid4()), tenant_id=tenant_id, student_id=student_id,
        )
        self._sessions[(tenant_id, s.session_id)] = s
        return s

    async def save(self, session: Session) -> None:
        self._sessions[(session.tenant_id, session.session_id)] = session
        session.persist()

    async def drop(self, session_id: str, *, tenant_id: str) -> None:
        """Buang sesi dari penyimpan. Penghapusan berkasnya urusan pemanggil.

        Dipakai DELETE /conversations/{id}: tanpa membuang salinan sesi yang
        masih hidup, permintaan berikutnya dengan `session_id` yang sama akan
        menuliskannya kembali ke disk — dan percakapan yang sudah dihapus
        mahasiswa muncul lagi.
        """
        self._sessions.pop((require_tenant_id(tenant_id, operation="session.drop"),
                            session_id), None)


class RedisSessionStore:
    """Sesi bersama antar pekerja, dengan kedaluwarsa ditangani Redis."""

    def __init__(self, url: str) -> None:
        # Diimpor di sini, bukan di puncak berkas, supaya pemasangan tanpa Redis
        # (pengembangan lokal, pengujian) tidak wajib punya paketnya.
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, decode_responses=True)

    @staticmethod
    def _key(tenant_id: str, session_id: str) -> str:
        return f"{_REDIS_PREFIX}{require_tenant_id(tenant_id, operation='session')}:{session_id}"

    async def get(self, session_id: str, *, tenant_id: str) -> Session | None:
        tenant_id = require_tenant_id(tenant_id, operation="session.get")
        raw = await self._redis.get(self._key(tenant_id, session_id))
        s = Session.from_json(raw) if raw is not None else None
        # Lapis kedua: kuncinya sudah memuat tenant, tetapi isi yang menyebut
        # kampus lain tetap tidak boleh dipakai.
        if s is not None and s.tenant_id == tenant_id:
            return s
        # Kunci sudah kedaluwarsa (atau isinya ditolak): percakapannya sendiri
        # masih ada di disk, jadi sesi dibangun ulang dari sana.
        return _pulihkan_dari_disk(session_id, tenant_id)

    async def get_or_create(
        self, session_id: str | None, *, tenant_id: str,
        student_id: str | None = None,
    ) -> Session:
        """Padanan Redis — lihat MemorySessionStore.get_or_create."""
        tenant_id = require_tenant_id(tenant_id, operation="session.get_or_create")
        if session_id:
            s = await self.get(session_id, tenant_id=tenant_id)
            if s:
                if student_id and not s.student_id:
                    s.student_id = student_id
                s.touch()
                return s
        s = Session(
            session_id=str(uuid.uuid4()), tenant_id=tenant_id, student_id=student_id,
        )
        await self.save(s)
        return s

    async def save(self, session: Session) -> None:
        # TTL disegarkan setiap penyimpanan, jadi sesi yang aktif tidak pernah
        # kedaluwarsa di tengah percakapan sementara yang terbengkalai hilang
        # sendiri tanpa perlu proses pembersih.
        await self._redis.set(
            self._key(session.tenant_id, session.session_id),
            session.to_json(),
            ex=TTL_SECONDS,
        )
        session.persist()

    async def drop(self, session_id: str, *, tenant_id: str) -> None:
        """Padanan Redis dari MemorySessionStore.drop — lihat alasannya di sana."""
        await self._redis.delete(self._key(tenant_id, session_id))


def _build_store() -> MemorySessionStore | RedisSessionStore:
    if settings.session_backend == "redis":
        logger.info("Session store: REDIS di {}", settings.redis_url)
        return RedisSessionStore(settings.redis_url)

    logger.info("Session store: MEMORI (hanya benar untuk satu proses backend)")
    return MemorySessionStore()


session_store = _build_store()
