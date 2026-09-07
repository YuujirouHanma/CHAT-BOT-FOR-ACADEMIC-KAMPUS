"""Latihan koding: mahasiswa menulis program, AI menilai dan mengoreksi.

KODE MAHASISWA TIDAK PERNAH DIJALANKAN DI SERVER
------------------------------------------------
Ini keputusan paling penting di berkas ini, dan sengaja ditulis paling atas.

Menjalankan kode yang dikirim orang lain di server sama saja menyerahkan mesin
itu kepada pengirimnya: `os.system`, membaca `.env`, menghubungi jaringan
internal, atau sekadar `while True` yang menghabiskan CPU. Untuk sebuah alat
belajar yang dipakai puluhan mahasiswa sekaligus, risikonya tidak sebanding
dengan manfaatnya.

Karena itu penilaian di sini bertumpu pada dua lapis yang keduanya AMAN:

1. **Analisis statis** — `ast.parse` membaca struktur kode tanpa
   mengeksekusinya. Dari situ didapat: galat sintaks beserta nomor barisnya,
   apakah fungsi yang diminta benar-benar didefinisikan, apakah konstruksi yang
   dilarang dipakai, dan apakah kirimannya sekadar tempelan kosong. Semuanya
   pasti, gratis, dan seketika.
2. **Tinjauan LLM** — hanya untuk yang menuntut penilaian: apakah logikanya
   benar, dan di mana letak salahnya.

Kalau nanti eksekusi sungguhan dibutuhkan (mis. untuk mencocokkan keluaran
dengan kasus uji), jalurnya BUKAN menambahkan `exec()` di sini, melainkan salah
satu dari: Pyodide di peramban mahasiswa (server tidak menanggung risiko sama
sekali), atau kontainer sekali-pakai tanpa jaringan dengan batas CPU/memori.
Lihat docs/SECURITY.md.

Umpan balik memberi PETUNJUK lebih dulu
---------------------------------------
Menyodorkan kode yang benar begitu mahasiswa salah memang menyelesaikan
soalnya, tetapi menghapus proses belajarnya. Jadi umpan balik disusun
bertingkat: apa yang sudah benar, lalu di mana yang keliru, lalu petunjuk —
dan koreksi utuh hanya ketika mahasiswa memintanya.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from src.config import PROJECT_ROOT
from src.security import redact
from src.tenancy import require_tenant_id, storage_prefix
from src.utils.logger import logger

SUBMISSION_DIR = PROJECT_ROOT / "data" / "livecode"
SUBMISSION_LOG: Final = "submissions.jsonl"

Language = Literal["python"]
SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("python",)

# Batas ukuran kiriman. Jauh di atas latihan pemrograman dasar mana pun, tetapi
# cukup rendah untuk menahan kiriman yang sengaja dibuat besar agar membebani
# analisis maupun panggilan LLM sesudahnya.
MAX_CODE_CHARS: Final = 20_000

Severity = Literal["galat", "peringatan", "info"]


@dataclass(frozen=True)
class CodeFinding:
    """Satu temuan dari analisis statis."""
    severity: Severity
    message: str
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "message": self.message, "line": self.line}


@dataclass(frozen=True)
class Exercise:
    """Satu latihan koding.

    `test_cases` disimpan sebagai pasangan masukan-keluaran yang DIHARAPKAN.
    Selama eksekusi belum ada, pasangan itu dipakai sebagai bahan penalaran LLM
    — bukan dijalankan. Begitu eksekusi tersedia nanti, bentuk datanya sudah siap
    dipakai apa adanya.
    """
    exercise_id: str
    title: str
    prompt: str
    language: str = "python"
    starter_code: str = ""
    expected_behavior: str = ""
    # Nama fungsi yang wajib didefinisikan mahasiswa; kosong = bebas.
    required_function: str = ""
    # Konstruksi yang dilarang, mis. ["sum", "sorted"] untuk latihan yang justru
    # meminta mahasiswa menulis sendiri logikanya.
    forbidden_names: tuple[str, ...] = ()
    test_cases: tuple[dict[str, Any], ...] = ()
    rubric: tuple[str, ...] = ()
    course_id: str | None = None
    week: int | None = None

    def as_public(self) -> dict[str, Any]:
        """Bentuk yang boleh dilihat mahasiswa.

        Tanpa rubrik dan tanpa keluaran yang diharapkan pada kasus uji — kalau
        ikut dikirim, mahasiswa dapat menuliskan jawabannya langsung tanpa
        menulis programnya.
        """
        return {
            "exercise_id": self.exercise_id,
            "title": self.title,
            "prompt": self.prompt,
            "language": self.language,
            "starter_code": self.starter_code,
            "expected_behavior": self.expected_behavior,
            "required_function": self.required_function,
            "forbidden_names": list(self.forbidden_names),
            # Hanya masukannya yang ditampilkan, sebagai contoh pemakaian.
            "example_inputs": [
                tc.get("input") for tc in self.test_cases[:3] if "input" in tc
            ],
        }


# --- Analisis statis (tanpa eksekusi) -------------------------------------------

def _find_defined_names(tree: ast.AST) -> set[str]:
    """Nama fungsi dan kelas yang didefinisikan di dalam kode."""
    nama: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            nama.add(node.name)
    return nama


def _find_used_names(tree: ast.AST) -> set[str]:
    """Nama yang DIPANGGIL atau dirujuk — dipakai memeriksa larangan."""
    dipakai: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            dipakai.add(node.id)
        elif isinstance(node, ast.Attribute):
            dipakai.add(node.attr)
    return dipakai


def _is_trivial(tree: ast.AST, code: str) -> bool:
    """Apakah kiriman ini sebenarnya belum berisi apa-apa.

    Menangkap kiriman yang hanya berupa `pass`, komentar, atau kode awal yang
    dikembalikan tanpa diubah. Tanpa ini, LLM akan diminta menilai kode kosong —
    memboroskan panggilan dan menghasilkan umpan balik yang membingungkan.
    """
    if not code.strip():
        return True
    bermakna = [
        n for n in ast.walk(tree)
        if not isinstance(n, ast.Module | ast.Pass | ast.Expr | ast.Constant | ast.Load)
    ]
    return not bermakna


def analyze_code(
    code: str,
    exercise: Exercise,
) -> tuple[list[CodeFinding], bool]:
    """Periksa kode secara statis. Mengembalikan (temuan, layak_dinilai_llm).

    `layak_dinilai_llm` bernilai False bila kode bahkan tidak dapat diurai atau
    masih kosong — pada keadaan itu memanggil LLM hanya membuang biaya, karena
    tidak ada logika yang bisa ditinjau.
    """
    temuan: list[CodeFinding] = []

    if len(code) > MAX_CODE_CHARS:
        temuan.append(CodeFinding(
            "galat", f"Kode terlalu panjang (maksimal {MAX_CODE_CHARS:,} karakter).",
        ))
        return temuan, False

    if exercise.language not in SUPPORTED_LANGUAGES:
        temuan.append(CodeFinding(
            "info",
            f"Bahasa '{exercise.language}' belum dianalisis otomatis; "
            "penilaian sepenuhnya oleh peninjau.",
        ))
        return temuan, True

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Nomor baris disertakan supaya mahasiswa langsung tahu di mana melihat.
        temuan.append(CodeFinding(
            "galat",
            f"Kode belum bisa dibaca Python: {exc.msg}.",
            line=exc.lineno,
        ))
        return temuan, False
    except (ValueError, RecursionError) as exc:
        temuan.append(CodeFinding("galat", f"Kode tidak dapat diurai: {exc}"))
        return temuan, False

    if _is_trivial(tree, code):
        temuan.append(CodeFinding(
            "galat", "Belum ada kode yang ditulis — masih kosong atau hanya `pass`.",
        ))
        return temuan, False

    if exercise.required_function:
        if exercise.required_function not in _find_defined_names(tree):
            temuan.append(CodeFinding(
                "galat",
                f"Fungsi `{exercise.required_function}` belum didefinisikan. "
                "Periksa lagi nama fungsinya — harus persis sama.",
            ))

    if exercise.forbidden_names:
        dipakai = _find_used_names(tree)
        terlarang = sorted(set(exercise.forbidden_names) & dipakai)
        if terlarang:
            temuan.append(CodeFinding(
                "peringatan",
                "Latihan ini meminta kamu menulis sendiri logikanya, tetapi kode "
                f"memakai: {', '.join('`' + t + '`' for t in terlarang)}.",
            ))

    return temuan, True


def has_blocking_error(findings: list[CodeFinding]) -> bool:
    return any(f.severity == "galat" for f in findings)


# --- Penyimpanan kiriman --------------------------------------------------------

def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _submission_path(tenant_id: str) -> Path:
    return SUBMISSION_DIR / storage_prefix(tenant_id) / SUBMISSION_LOG


def record_submission(
    *,
    tenant_id: str,
    exercise_id: str,
    student_id: str | None,
    code: str,
    result: dict[str, Any],
    session_id: str | None = None,
) -> str:
    """Simpan satu kiriman beserta hasil penilaiannya.

    Disimpan SELURUHNYA, termasuk kiriman yang salah. Justru percobaan yang gagal
    — dan berapa kali mahasiswa mencoba sebelum berhasil — yang menjadi data
    paling berharga untuk mengukur apakah sebuah gaya belajar benar-benar
    membantu. Menyimpan hanya yang berhasil akan menghapus persis bagian itu.
    """
    tenant_id = require_tenant_id(tenant_id, operation="record_submission")
    submission_id = f"lc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(code)) % 10**6:06d}"
    catatan = {
        "submission_id": submission_id,
        "at": _now(),
        "tenant_id": tenant_id,
        "exercise_id": exercise_id,
        "student_id": student_id,
        "session_id": session_id,
        "code": code[:MAX_CODE_CHARS],
        "lulus": result.get("lulus"),
        "skor": result.get("skor"),
        "temuan": result.get("temuan"),
    }
    path = _submission_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redact.scrub(catatan), ensure_ascii=False) + "\n")
    except OSError as exc:
        # Kegagalan menyimpan tidak boleh membatalkan umpan balik yang sudah
        # dihasilkan — dari sudut pandang mahasiswa, kehilangan catatan jauh
        # lebih ringan daripada kehilangan koreksinya.
        logger.warning("Gagal menyimpan kiriman livecode: {}", exc)
    return submission_id


def list_submissions(
    *,
    tenant_id: str,
    student_id: str | None = None,
    exercise_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Riwayat kiriman, terbaru lebih dulu."""
    tenant_id = require_tenant_id(tenant_id, operation="list_submissions")
    path = _submission_path(tenant_id)
    if not path.exists():
        return []

    hasil: list[dict[str, Any]] = []
    try:
        baris = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Gagal membaca kiriman livecode: {}", exc)
        return []

    for b in reversed(baris):
        b = b.strip()
        if not b:
            continue
        try:
            d = json.loads(b)
        except json.JSONDecodeError:
            continue
        if student_id and d.get("student_id") != redact.pseudonymize(student_id):
            continue
        if exercise_id and d.get("exercise_id") != exercise_id:
            continue
        hasil.append(d)
        if len(hasil) >= limit:
            break
    return hasil


def attempt_stats(*, tenant_id: str, exercise_id: str) -> dict[str, Any]:
    """Statistik percobaan sebuah latihan — bahan analisis untuk makalah."""
    kiriman = list_submissions(tenant_id=tenant_id, exercise_id=exercise_id, limit=10_000)
    if not kiriman:
        return {"total_kiriman": 0, "lulus": 0, "rasio_lulus": None}
    lulus = sum(1 for k in kiriman if k.get("lulus"))
    return {
        "total_kiriman": len(kiriman),
        "lulus": lulus,
        # None saat belum ada kiriman sama sekali ditangani di atas; di sini
        # rasio selalu terdefinisi.
        "rasio_lulus": round(lulus / len(kiriman), 3),
    }


# --- Pembentukan latihan dari soal koding evaluasi -------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def exercise_from_item(
    item: dict[str, Any],
    *,
    course_id: str | None = None,
    week: int | None = None,
    index: int = 0,
) -> Exercise:
    """Ubah satu butir soal `koding` dari evaluasi menjadi latihan livecode.

    Menyatukan keduanya disengaja: soal koding pada ETS dan latihan mandiri
    adalah hal yang sama bagi mahasiswa, dan menilainya dengan dua jalur berbeda
    akan menghasilkan dua standar penilaian yang berbeda pula.
    """
    judul = (item.get("question") or "Latihan koding").strip()
    slug = _SLUG_RE.sub("-", judul.lower())[:40].strip("-") or "latihan"
    return Exercise(
        exercise_id=f"{course_id or 'umum'}-{week or 0}-{index}-{slug}",
        title=judul[:120],
        prompt=judul,
        starter_code=item.get("starter_code", ""),
        expected_behavior=item.get("expected_behavior", ""),
        required_function=item.get("required_function", ""),
        rubric=tuple(item.get("rubric") or ()),
        test_cases=tuple(item.get("test_cases") or ()),
        course_id=course_id,
        week=week,
    )
