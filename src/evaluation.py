"""Evaluasi berkala: kuis, ETS, dan EAS atas RENTANG minggu.

Bedanya dengan kuis materi yang sudah ada
-----------------------------------------
`pipeline.quiz()` membuat kuis dari SATU materi. Evaluasi di sini mencakup
beberapa minggu sekaligus — minggu 4 menguji minggu 1-4, minggu 8 menguji
minggu 1-8, dan seterusnya. Itu bukan sekadar kuis yang diperbesar: soal yang
menuntut mahasiswa menghubungkan materi minggu 2 dengan minggu 5 tidak mungkin
lahir dari satu berkas materi saja.

Kenapa lebih dari satu jenis soal
---------------------------------
Pilihan ganda hanya mengukur PENGENALAN — mahasiswa cukup mengenali jawaban
benar di antara empat pilihan. Ia tidak mengukur apakah mahasiswa bisa
MEMPRODUKSI jawaban itu sendiri.

Perbedaan itu menentukan untuk penelitian ini. Kalau tujuannya mencari gaya
belajar mana yang paling cocok, dan satu-satunya alat ukur adalah pilihan ganda,
maka yang terukur hanyalah gaya mana yang paling membantu mengenali jawaban —
bukan gaya mana yang paling membantu memahami. Karena itu isian singkat, esai,
dan soal koding disediakan sejak awal, bukan ditambahkan belakangan.

Jadwal adalah milik tiap kampus
-------------------------------
`DEFAULT_SCHEDULE` hanyalah BAWAAN. Kalender akademik berbeda antar kampus, jadi
setiap fungsi jadwal di sini menerima `EvaluationSchedule` milik tenant sebagai
argumen alih-alih membaca satu jadwal global — tenant yang tidak menentukan
sendiri tetap mendapat bawaan.

Penilaian
---------
- `pilihan_ganda`, `benar_salah` → dinilai mesin, pasti dan gratis.
- `isian_singkat`, `esai`, `koding` → dinilai LLM terhadap kunci/rubrik.

Pemisahan itu disengaja: soal yang bisa dinilai pasti TIDAK BOLEH diserahkan ke
LLM. Menyerahkannya berarti menukar kepastian dengan biaya, kelambatan, dan
kemungkinan salah nilai — tanpa satu pun keuntungan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

QuestionType = Literal[
    "pilihan_ganda", "benar_salah", "isian_singkat", "esai", "koding",
]

QUESTION_TYPES: Final[tuple[str, ...]] = (
    "pilihan_ganda", "benar_salah", "isian_singkat", "esai", "koding",
)

# Jenis yang jawabannya tunggal dan pasti, sehingga dapat dinilai tanpa LLM.
AUTO_GRADED: Final[frozenset[str]] = frozenset({"pilihan_ganda", "benar_salah"})

EvaluationKind = Literal["kuis", "ets", "eas"]

# Label yang dilihat mahasiswa. ETS/EAS memakai istilah yang sudah dikenal di
# kampus Indonesia, bukan terjemahan harfiah "ujian tengah semester".
KIND_LABELS: Final[dict[str, str]] = {
    "kuis": "Kuis",
    "ets": "Evaluasi Tengah Semester",
    "eas": "Evaluasi Akhir Semester",
}


@dataclass(frozen=True)
class Blueprint:
    """Komposisi soal sebuah evaluasi: berapa butir per jenis.

    Disimpan sebagai data, bukan ditanam di dalam prompt, supaya dosen dapat
    mengubah komposisinya tanpa menyentuh kode — dan supaya komposisi yang
    dipakai pada sebuah penelitian dapat dilaporkan apa adanya.
    """
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def types(self) -> list[str]:
        return [t for t in QUESTION_TYPES if self.counts.get(t, 0) > 0]

    def validate(self) -> None:
        asing = set(self.counts) - set(QUESTION_TYPES)
        if asing:
            raise ValueError(f"Jenis soal tidak dikenal: {sorted(asing)}")
        if any(n < 0 for n in self.counts.values()):
            raise ValueError("Jumlah soal tidak boleh negatif")
        if self.total == 0:
            raise ValueError("Evaluasi harus memuat setidaknya satu soal")
        if self.total > 40:
            # Batas atas praktis: melampaui ini, keluaran LLM mulai terpotong
            # dan mahasiswa kelelahan sebelum soal terakhir.
            raise ValueError("Evaluasi maksimal 40 soal")


# Komposisi bawaan per jenis evaluasi. Kuis mingguan sengaja ringan; ETS/EAS
# memuat soal produksi (isian, esai, koding) karena di situlah pemahaman yang
# sesungguhnya terlihat.
DEFAULT_BLUEPRINTS: Final[dict[str, Blueprint]] = {
    "kuis": Blueprint({"pilihan_ganda": 5, "benar_salah": 2, "isian_singkat": 1}),
    "ets": Blueprint({
        "pilihan_ganda": 8, "benar_salah": 3, "isian_singkat": 3,
        "esai": 2, "koding": 1,
    }),
    "eas": Blueprint({
        "pilihan_ganda": 8, "benar_salah": 3, "isian_singkat": 3,
        "esai": 3, "koding": 2,
    }),
}


@dataclass(frozen=True)
class EvaluationPlan:
    """Evaluasi apa yang dijalankan pada sebuah minggu, dan mencakup minggu mana."""
    week: int
    kind: str
    weeks_covered: tuple[int, ...]
    blueprint: Blueprint

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, "Evaluasi")

    @property
    def range_text(self) -> str:
        if not self.weeks_covered:
            return ""
        awal, akhir = self.weeks_covered[0], self.weeks_covered[-1]
        return f"minggu {awal}" if awal == akhir else f"minggu {awal}-{akhir}"


def _plan(week: int, kind: str, awal: int, akhir: int) -> EvaluationPlan:
    return EvaluationPlan(
        week=week, kind=kind,
        weeks_covered=tuple(range(awal, akhir + 1)),
        blueprint=DEFAULT_BLUEPRINTS[kind],
    )


# Jadwal bawaan satu semester. Kuis menguji blok terdekat; ETS dan EAS bersifat
# kumulatif atas separuh semester — itulah bedanya dengan kuis, dan alasan
# rentangnya tidak sekadar empat minggu terakhir.
DEFAULT_SCHEDULE: Final[dict[int, EvaluationPlan]] = {
    4: _plan(4, "kuis", 1, 4),
    8: _plan(8, "ets", 1, 8),
    12: _plan(12, "kuis", 9, 12),
    16: _plan(16, "eas", 9, 16),
}

EVALUATION_WEEKS: Final[tuple[int, ...]] = tuple(sorted(DEFAULT_SCHEDULE))

# Batas atas minggu yang masih masuk akal untuk satu semester. Disamakan dengan
# batas `week` di skema API, supaya jadwal yang tersimpan tidak pernah memuat
# minggu yang justru ditolak saat klien memintanya.
MAX_WEEK: Final[int] = 52


def _week_number(nilai: Any, medan: str) -> int:
    """Baca satu nomor minggu dari JSON, menolak yang di luar nalar.

    `bool` ditolak terpisah karena di Python ia subkelas `int`, sehingga tanpa
    pemeriksaan ini `true` akan lolos sebagai minggu 1.
    """
    if isinstance(nilai, bool) or not isinstance(nilai, int):
        raise ValueError(f"{medan} harus bilangan bulat; diterima {nilai!r}")
    if not 1 <= nilai <= MAX_WEEK:
        raise ValueError(f"{medan} harus antara 1 dan {MAX_WEEK}; diterima {nilai}")
    return nilai


def _plan_from_json(entri: Any) -> EvaluationPlan:
    """Satu baris jadwal: minggu, jenis, rentang yang diuji, komposisi soal."""
    if not isinstance(entri, dict):
        raise ValueError(f"Setiap entri jadwal harus berupa objek; diterima {entri!r}")

    week = _week_number(entri.get("week"), "'week'")
    kind = entri.get("kind", "kuis")
    if kind not in KIND_LABELS:
        raise ValueError(f"Jenis evaluasi tidak dikenal: {kind!r}")

    covers = entri.get("covers") or [1, week]
    if not isinstance(covers, list | tuple) or len(covers) != 2:
        raise ValueError("'covers' harus berbentuk [minggu_awal, minggu_akhir]")
    awal = _week_number(covers[0], "minggu awal 'covers'")
    akhir = _week_number(covers[1], "minggu akhir 'covers'")
    if awal > akhir:
        raise ValueError(f"Rentang 'covers' terbalik: minggu {awal}-{akhir}")
    if akhir > week:
        # Evaluasi tidak mungkin menguji minggu yang belum berlangsung; kalau
        # dibiarkan, mahasiswa mendapat soal dari materi yang belum diajarkan.
        raise ValueError(
            f"Evaluasi minggu {week} tidak dapat mencakup sampai minggu {akhir}"
        )

    counts = entri.get("counts")
    if counts is None:
        bp = DEFAULT_BLUEPRINTS[kind]
    else:
        if not isinstance(counts, dict) or any(
            isinstance(n, bool) or not isinstance(n, int) for n in counts.values()
        ):
            raise ValueError("'counts' harus berupa objek {jenis_soal: jumlah}")
        bp = Blueprint(dict(counts))
    bp.validate()

    return EvaluationPlan(
        week=week, kind=kind,
        weeks_covered=tuple(range(awal, akhir + 1)),
        blueprint=bp,
    )


@dataclass(frozen=True)
class EvaluationSchedule:
    """Kalender evaluasi milik SATU kampus.

    Kalender akademik adalah milik tiap kampus: ada yang bersemester 14 minggu,
    ada yang menaruh ETS di minggu 7. Selama jadwalnya tertanam di kode, kampus
    seperti itu mendapat rencana evaluasi yang salah TANPA pesan apa pun —
    sebagai data per tenant, ia menjadi setelan yang dapat diperiksa dan diubah.
    """
    plans: dict[int, EvaluationPlan]

    @property
    def weeks(self) -> tuple[int, ...]:
        return tuple(sorted(self.plans))

    @classmethod
    def from_json(cls, raw: Any) -> EvaluationSchedule:
        """Muat jadwal dari bentuk JSON; galat bila tidak sah.

        Sengaja divalidasi saat DIMUAT, bukan saat dipakai: jadwal cacat yang
        lolos ke runtime baru ketahuan setelah mahasiswa mengerjakan evaluasi
        dengan rentang minggu yang keliru.
        """
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                "Jadwal evaluasi harus berupa daftar tidak kosong; hapus medannya "
                "bila kampus ini memakai jadwal bawaan"
            )
        plans: dict[int, EvaluationPlan] = {}
        for entri in raw:
            plan = _plan_from_json(entri)
            if plan.week in plans:
                raise ValueError(f"Minggu {plan.week} muncul lebih dari sekali")
            plans[plan.week] = plan
        return cls(plans)

    def as_json(self) -> list[dict[str, Any]]:
        """Bentuk yang ditulis ke berkas tenant."""
        hasil: list[dict[str, Any]] = []
        for w in self.weeks:
            p = self.plans[w]
            entri: dict[str, Any] = {
                "week": p.week,
                "kind": p.kind,
                "covers": [p.weeks_covered[0], p.weeks_covered[-1]],
            }
            # Komposisi hanya ditulis bila berbeda dari bawaan jenisnya, supaya
            # berkas tenant menyorot apa yang memang sengaja dibedakan.
            if p.blueprint.counts != DEFAULT_BLUEPRINTS[p.kind].counts:
                entri["counts"] = dict(p.blueprint.counts)
            hasil.append(entri)
        return hasil


def plans_for(schedule: EvaluationSchedule | None) -> dict[int, EvaluationPlan]:
    """Jadwal milik tenant bila ada, jadwal bawaan bila tenant tidak menentukan."""
    return DEFAULT_SCHEDULE if schedule is None else schedule.plans


def is_evaluation_week(
    week: int | None, schedule: EvaluationSchedule | None = None,
) -> bool:
    return week in plans_for(schedule)


def plan_for_week(
    week: int, schedule: EvaluationSchedule | None = None,
) -> EvaluationPlan | None:
    """Rencana evaluasi untuk sebuah minggu; None bila minggu biasa."""
    return plans_for(schedule).get(week)


def evaluation_weeks(schedule: EvaluationSchedule | None = None) -> tuple[int, ...]:
    """Minggu evaluasi milik tenant; `EVALUATION_WEEKS` adalah bawaannya."""
    return tuple(sorted(plans_for(schedule)))


def custom_plan(
    weeks: list[int],
    kind: str = "kuis",
    counts: dict[str, int] | None = None,
) -> EvaluationPlan:
    """Evaluasi atas rentang minggu bebas — dipakai saat dosen menentukan sendiri.

    Ada karena jadwal bawaan tidak akan cocok untuk semua kelas: ada yang ingin
    menguji minggu 4-8, ada yang hanya 5-7.
    """
    if not weeks:
        raise ValueError("Rentang minggu tidak boleh kosong")
    if kind not in KIND_LABELS:
        raise ValueError(f"Jenis evaluasi tidak dikenal: {kind!r}")
    bp = Blueprint(dict(counts)) if counts else DEFAULT_BLUEPRINTS[kind]
    bp.validate()
    urut = tuple(sorted(set(weeks)))
    return EvaluationPlan(
        week=urut[-1], kind=kind, weeks_covered=urut, blueprint=bp,
    )


# --- Penilaian ------------------------------------------------------------------

@dataclass(frozen=True)
class GradedItem:
    """Hasil penilaian satu butir soal."""
    index: int
    type: str
    question: str
    student_answer: Any
    correct_answer: Any
    is_correct: bool | None      # None = perlu dinilai LLM/dosen
    score: float                 # 0.0-1.0
    explanation: str = ""
    feedback: str = ""


# Bentuk teks jawaban benar/salah yang diterima, dipetakan ke indeks opsi
# ["Benar", "Salah"]. Klien JSON mengirim jawaban ini dalam bentuk yang
# berbeda-beda; yang dinilai tetap indeksnya.
_BENAR_SALAH_TEKS: Final[dict[str, int]] = {
    "benar": 0, "true": 0, "b": 0,
    "salah": 1, "false": 1, "s": 1,
}


def _indeks_pilihan(answer: Any) -> int | None:
    """Indeks opsi yang dipilih pada soal berpilihan; None bila bukan indeks.

    `bool` ditolak eksplisit: di Python ia subkelas `int`, sehingga tanpa
    pemeriksaan ini `True` lolos sebagai indeks 1 dan `False` sebagai indeks 0.
    """
    if isinstance(answer, bool) or not isinstance(answer, int):
        return None
    return answer


def _indeks_benar_salah(answer: Any) -> int | None:
    """Indeks opsi untuk soal benar/salah, mengikuti urutan ["Benar", "Salah"].

    Bentuk jawaban yang dianggap sah — sengaja lebih luas daripada pilihan
    ganda, karena "benar/salah" secara alami dikirim sebagai boolean, bukan
    sebagai nomor opsi:

    - boolean JSON: `true` → 0 (Benar), `false` → 1 (Salah);
    - indeks opsi: 0 atau 1, sama seperti pilihan ganda. Indeks di luar itu
      tidak sah karena soal ini hanya punya dua opsi;
    - teks: "benar"/"true"/"b" → 0 dan "salah"/"false"/"s" → 1, tanpa
      membedakan huruf besar-kecil.

    Selain itu dihitung tidak terjawab (None), termasuk teks "0"/"1" — angka
    yang datang sebagai teks lebih mungkin berasal dari klien yang keliru
    daripada dari mahasiswa yang memilih.
    """
    if isinstance(answer, bool):
        return 0 if answer else 1
    if isinstance(answer, int):
        return answer if answer in (0, 1) else None
    if isinstance(answer, str):
        return _BENAR_SALAH_TEKS.get(answer.strip().lower())
    return None


def grade_objective(item: dict[str, Any], answer: Any, index: int) -> GradedItem:
    """Nilai satu soal objektif tanpa memanggil LLM.

    Tiap jenis menafsirkan jawabannya sendiri: `pilihan_ganda` hanya menerima
    indeks opsi, sedangkan `benar_salah` juga menerima boolean dan teks
    (lihat `_indeks_benar_salah`). Menyeragamkannya justru salah — boolean
    `true` bukan "opsi nomor 1", melainkan "Benar", yaitu opsi nomor 0.

    Jawaban yang tidak dikenali dihitung SALAH di sini — berbeda dari kuis di
    dalam chat, yang menanyakan ulang. Pada evaluasi resmi, menanyakan ulang
    berarti memberi kesempatan menebak berkali-kali.
    """
    tipe = item.get("type", "pilihan_ganda")
    benar = item.get("answer_index")
    dipilih = (
        _indeks_benar_salah(answer) if tipe == "benar_salah"
        else _indeks_pilihan(answer)
    )
    tepat = dipilih is not None and dipilih == benar
    return GradedItem(
        index=index,
        type=tipe,
        question=item.get("question", ""),
        student_answer=dipilih,
        correct_answer=benar,
        is_correct=tepat,
        score=1.0 if tepat else 0.0,
        explanation=item.get("explanation", ""),
    )


def needs_llm_grading(item: dict[str, Any]) -> bool:
    return item.get("type", "pilihan_ganda") not in AUTO_GRADED


def summarize(graded: list[GradedItem]) -> dict[str, Any]:
    """Ringkasan skor, keseluruhan dan per jenis soal.

    Rincian per jenis adalah bagian yang paling berguna untuk penelitian: nilai
    total yang sama bisa berarti dua hal sangat berbeda — mahasiswa yang kuat di
    pilihan ganda tetapi lemah di esai belum tentu memahami materinya.
    """
    if not graded:
        return {
            "total": 0, "skor": 0.0, "benar": 0,
            "per_jenis": {}, "perlu_tinjauan": 0,
        }

    total_skor = sum(g.score for g in graded)
    per_jenis: dict[str, dict[str, Any]] = {}
    for g in graded:
        d = per_jenis.setdefault(g.type, {"jumlah": 0, "skor": 0.0})
        d["jumlah"] += 1
        d["skor"] += g.score
    for d in per_jenis.values():
        d["persen"] = round(100.0 * d["skor"] / d["jumlah"], 1) if d["jumlah"] else 0.0
        d["skor"] = round(d["skor"], 2)

    return {
        "total": len(graded),
        "benar": sum(1 for g in graded if g.is_correct),
        "skor": round(100.0 * total_skor / len(graded), 1),
        "per_jenis": per_jenis,
        # Butir yang LLM tidak yakin menilainya — ditandai agar dosen meninjau,
        # bukan didiamkan sebagai nilai yang seolah pasti.
        "perlu_tinjauan": sum(1 for g in graded if g.is_correct is None),
    }
