"""Guided conversation: menuntun mahasiswa memilih mata kuliah → minggu → materi.

Kenapa ada: sebagian besar pengguna belum pernah memakai AI dan tidak tahu harus
bertanya apa. Daripada menghadapkan mereka pada kotak chat kosong, chatbot
menawarkan pilihan yang bisa diklik. Semua pilihan berasal dari materi yang
BENAR-BENAR terindex, jadi tidak pernah menawarkan minggu/materi yang kosong.

Tetap bisa dipakai bebas seperti chatbot biasa. Kalau pesan mahasiswa sudah
menyebut konteksnya sendiri ("saya mau sbd minggu 3"), langkah tanya-balik
dilewati dan sistem langsung memasang filternya.

Modul ini SENGAJA murni fungsi teks tanpa LLM dan tanpa I/O:
- cepat dan tidak menambah biaya token pada tiap pesan,
- tidak mungkin mengarang mata kuliah/minggu yang tidak ada, karena kosakatanya
  dipasok dari daftar yang sudah terindex,
- gampang diuji tanpa menyalakan server.

Orkestrasinya (mengambil daftar dari Qdrant, memanggil starter questions) ada di
`RAGPipeline.guided_turn`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

# --- Langkah percakapan ---------------------------------------------------------
# Literal (bukan str lepas) supaya nilainya ikut tervalidasi saat dipetakan ke
# schema API — salah tulis langkah jadi error di type checker, bukan di runtime.
StepName = Literal["course", "week", "material", "style", "question", "answer", "quiz"]
ChoiceKind = Literal[
    "course", "week", "material", "style", "question", "quiz", "back",
]

STEP_COURSE: StepName = "course"       # menanyakan mata kuliah
STEP_WEEK: StepName = "week"           # menanyakan minggu
STEP_MATERIAL: StepName = "material"   # menanyakan materi
STEP_QUESTION: StepName = "question"   # konteks lengkap, menawarkan pertanyaan template
STEP_ANSWER: StepName = "answer"       # tidak menuntun; jawab pertanyaannya
STEP_STYLE: StepName = "style"         # menanyakan gaya belajar
STEP_QUIZ: StepName = "quiz"           # sedang mengerjakan kuis

# "minggu 3", "minggu ke-3", "minggu ke 3", "week 3", "pertemuan 9", "tm9", "w3"
_WEEK_RE = re.compile(
    r"\b(?:minggu|week|pertemuan|tm|w)\s*(?:ke\s*-?\s*)?(\d{1,2})\b",
    re.IGNORECASE,
)

# Angka telanjang — hanya dipakai saat kita memang sedang menanyakan minggu/materi,
# supaya "3" tidak diartikan minggu 3 di tengah pertanyaan biasa.
_BARE_NUMBER_RE = re.compile(r"^\s*(\d{1,2})\s*$")

# Kata yang menandakan mahasiswa sedang BERTANYA, bukan memilih.
_QUESTION_WORDS = {
    "apa", "apakah", "bagaimana", "gimana", "mengapa", "kenapa", "jelaskan",
    "jelasin", "sebutkan", "contoh", "contohnya", "definisi", "pengertian",
    "perbedaan", "bedanya", "beda", "cara", "kapan", "siapa", "mana",
    "hitung", "hitunglah", "buktikan", "rangkum", "ringkas", "ringkasan",
    "maksud", "artinya", "fungsi", "tujuan", "manfaat", "langkah", "tahapan",
}

# Frasa navigasi: kelihatan seperti pertanyaan tapi maksudnya minta DAFTAR.
# "materi sbd minggu 3 apa saja?" harus memunculkan pilihan materi, bukan jawaban.
_NAV_PHRASES = (
    "apa saja", "apa aja", "ada apa", "apa sih", "daftar", "list",
    "lihat materi", "materinya", "ada berapa", "pilihan",
)


def _phrase_re(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Regex yang mencocokkan frasa hanya sebagai KATA UTUH.

    Wajib pakai batas kata, bukan `in`: "ulang" adalah substring dari
    "perulangan" (topik nyata di materi pemrograman), dan "list" substring dari
    "listrik". Dengan pencocokan substring, pertanyaan sungguhan tentang topik
    itu salah diartikan sebagai perintah navigasi dan tidak pernah dijawab.
    """
    return re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b"
    )


_NAV_RE = _phrase_re(_NAV_PHRASES)

# Pembuka niat belajar tanpa menyebut apa pun — layak dituntun dari langkah awal.
_START_PHRASES = (
    "mau belajar", "ingin belajar", "pengen belajar", "belajar", "mulai",
    "mulai belajar", "bantu saya", "bantu aku", "halo", "hai", "hello", "assalamualaikum",
    "pagi", "siang", "malam", "help", "menu", "mau tanya", "bisa bantu",
    "saya bingung", "bingung", "tidak tahu", "gak tahu", "ga tau", "gatau",
)

# Kata umum yang tidak boleh dipakai mencocokkan nama mata kuliah, supaya
# "materi" di "materi sbd" tidak ikut dianggap bagian nama.
_STOPWORDS = {
    "materi", "mata", "kuliah", "matkul", "minggu", "week", "pertemuan",
    "saya", "aku", "mau", "ingin", "pengen", "belajar", "tentang", "dari",
    "yang", "untuk", "dan", "di", "ke", "pada", "ada", "apa", "aja", "saja",
    "tolong", "bantu", "bisa", "dong", "ya", "nih", "sih",
}


@dataclass(frozen=True)
class Choice:
    """Satu tombol yang bisa diklik mahasiswa."""
    label: str
    value: str
    kind: ChoiceKind


@dataclass
class Refs:
    """Konteks yang berhasil dikenali dari satu pesan mahasiswa."""
    course_id: str | None = None
    weeks: list[int] = field(default_factory=list)
    source_file: str | None = None
    content_id: str | None = None
    # Sisa teks setelah bagian yang dikenali dibuang — dipakai menilai apakah
    # pesan ini sebetulnya sebuah pertanyaan.
    residual: str = ""
    matched: list[str] = field(default_factory=list)

    @property
    def any_found(self) -> bool:
        return bool(self.course_id or self.weeks or self.source_file)


# --- Normalisasi ---------------------------------------------------------------
def _fold(text: str) -> str:
    """Lowercase + buang aksen + rapikan pemisah jadi spasi tunggal."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> list[str]:
    return [t for t in _fold(text).split(" ") if t]


def _content_tokens(text: str) -> list[str]:
    """Token yang bukan stopword — dipakai menilai bobot sebuah pesan."""
    return [t for t in _tokens(text) if t not in _STOPWORDS]


# --- Pengenalan ----------------------------------------------------------------
# Rentang minggu: "minggu 3-5", "minggu 3 sampai 5", "minggu 3 s.d. 5"
_WEEK_RANGE_RE = re.compile(
    r"\b(?:minggu|week|pertemuan|tm|w)\s*(?:ke\s*-?\s*)?(\d{1,2})\s*"
    r"(?:-|–|s\.?d\.?|sampai|hingga|ke)\s*(\d{1,2})\b",
    re.IGNORECASE,
)

# Semua penyebutan minggu dalam satu pesan: "minggu 3 dan minggu 4"
_WEEK_ALL_RE = re.compile(
    r"\b(?:minggu|week|pertemuan|tm|w)\s*(?:ke\s*-?\s*)?(\d{1,2})\b",
    re.IGNORECASE,
)

# Deretan angka setelah kata minggu: "minggu 3 dan 4", "minggu 2,3,4"
_WEEK_LIST_RE = re.compile(
    r"\b(?:minggu|week|pertemuan|tm|w)\s*(?:ke\s*-?\s*)?"
    r"(\d{1,2}(?:\s*(?:,|dan|&|\+)\s*\d{1,2})+)\b",
    re.IGNORECASE,
)

_MAX_WEEK_SPAN = 16   # jaring pengaman agar salah baca tidak menyapu seluruh semester


def find_weeks(text: str, *, allow_bare_number: bool = False) -> list[int]:
    """Semua minggu yang disebut dalam satu pesan, terurut dan tanpa duplikat.

    Mahasiswa yang menyiapkan ujian sering perlu beberapa minggu sekaligus, dan
    menyebutnya dengan berbagai cara: "minggu 3 dan 4", "minggu 3-5",
    "minggu 2, 3, 4". Ketiganya dikenali.

    Rentang dibatasi `_MAX_WEEK_SPAN` supaya satu salah baca tidak diam-diam
    memperluas pencarian ke seluruh semester. Rentang yang ditolak karena terlalu
    lebar jatuh ke penyebutan minggu pertama ("minggu 1-40" → [1]), bukan kosong,
    supaya maksud mahasiswa tidak hilang sama sekali.
    """
    hasil: set[int] = set()

    for a, b in _WEEK_RANGE_RE.findall(text or ""):
        awal, akhir = int(a), int(b)
        if awal > akhir:
            awal, akhir = akhir, awal
        if akhir - awal < _MAX_WEEK_SPAN:
            hasil.update(range(awal, akhir + 1))

    for deret in _WEEK_LIST_RE.findall(text or ""):
        hasil.update(int(n) for n in re.findall(r"\d{1,2}", deret))

    if not hasil:
        hasil.update(int(n) for n in _WEEK_ALL_RE.findall(text or ""))

    if not hasil and allow_bare_number:
        polos = (text or "").strip()
        if re.fullmatch(r"\d{1,2}(?:\s*(?:,|dan|&|\+)\s*\d{1,2})*", polos):
            hasil.update(int(n) for n in re.findall(r"\d{1,2}", polos))

    return sorted(w for w in hasil if 1 <= w <= 52)


def find_week(text: str, *, allow_bare_number: bool = False) -> int | None:
    """Nomor minggu dari teks, atau None.

    `allow_bare_number` hanya diaktifkan saat kita memang sedang menanyakan
    minggu, supaya angka di tengah pertanyaan biasa tidak disalahartikan.
    """
    m = _WEEK_RE.search(text or "")
    if m:
        return int(m.group(1))
    if allow_bare_number:
        m = _BARE_NUMBER_RE.match(text or "")
        if m:
            return int(m.group(1))
    return None


# Kata yang mendahului nama mata kuliah saat mahasiswa menyebutnya sebagai
# KONTEKS ("... di sbd minggu 3", "buka mata kuliah struktur data"), bukan
# sebagai topik yang sedang ia tanyakan.
_CONTEXT_MARKERS = frozenset({
    "di", "pada", "untuk", "dari", "materi", "matkul", "kuliah", "mapel",
    "ganti", "pindah", "beralih", "buka",
})

# "ke" terlalu umum untuk berdiri sendiri sebagai penanda — "jalan ke data
# center" bukan permintaan pindah — jadi hanya dihitung bila didahului kata
# kerja perpindahan: "ganti ke basis data", "pindah ke struktur data".
_SWITCH_VERBS = frozenset({"ganti", "pindah", "beralih", "lanjut", "kembali", "balik"})


def _mentioned_as_context(folded: str, tokens: list[str]) -> bool:
    """Apakah salah satu `tokens` didahului kata penanda konteks di `folded`?"""
    kata = folded.split(" ")
    dicari = set(tokens)
    for i, w in enumerate(kata):
        if i == 0 or w not in dicari:
            continue
        sebelum = kata[i - 1]
        if sebelum in _CONTEXT_MARKERS:
            return True
        if sebelum == "ke" and i >= 2 and kata[i - 2] in _SWITCH_VERBS:
            return True
    return False


def match_course(text: str, courses: list[dict]) -> str | None:
    """Cocokkan teks ke salah satu course terindex; None kalau tidak yakin.

    `courses` = [{course_id, course_name}] dari store. Pencocokan dilakukan
    terhadap keduanya, dan yang paling banyak kata cocoknya menang — jadi
    "sistem basis data" mengalahkan tebakan berdasarkan satu kata saja.

    Pada pesan yang berupa PERTANYAAN ISI, nama mata kuliah hanya dianggap
    pemilihan konteks kalau ada penandanya ("di sbd minggu 3", "ganti ke basis
    data"). Tanpa syarat itu, kata umum yang kebetulan menjadi nama mata kuliah
    — "data", "struktur", "dasar" — membuat pertanyaan materi biasa memindahkan
    sesi mahasiswa ke mata kuliah lain tanpa ia minta: diam-diam, tanpa galat,
    dan paling sering justru dipicu oleh pertanyaan template buatan sistem
    sendiri. Pesan yang bukan pertanyaan (klik tombol, "sbd", "ganti ke X")
    tidak dibatasi — di situ menyebut nama mata kuliah memang berarti memilih.
    """
    folded = _fold(text)
    if not folded:
        return None
    haystack = set(folded.split(" "))
    perlu_penanda = looks_like_question(text)

    best_score = 0
    best_id: str | None = None
    for c in courses:
        cid = c.get("course_id")
        if not cid:
            continue
        score = 0
        for candidate in (cid, c.get("course_name") or ""):
            cand_tokens = [t for t in _tokens(candidate) if t not in _STOPWORDS]
            if not cand_tokens:
                continue
            hits = [t for t in cand_tokens if t in haystack]
            if not hits or (perlu_penanda
                            and not _mentioned_as_context(folded, hits)):
                continue
            if len(hits) == len(cand_tokens):
                score = max(score, len(hits) * 2)
            else:
                # Sebagian kata cocok: hanya kata panjang yang dihitung (hindari
                # cocok karena "di", "3"), dan harus mayoritas nama — satu kata
                # umum tidak boleh mewakili nama yang terdiri dari beberapa kata.
                panjang = [t for t in hits if len(t) >= 4]
                if len(panjang) * 2 >= len(cand_tokens):
                    score = max(score, len(panjang))
        if score > best_score:
            best_score, best_id = score, cid
    return best_id


def match_material(
    text: str, materials: list[dict], *, allow_ordinal: bool = False
) -> dict | None:
    """Cocokkan teks ke salah satu materi; None kalau tidak yakin.

    `materials` = [{source_file, content_id}]. Selain mencocokkan nama file,
    saat `allow_ordinal` aktif angka telanjang diartikan sebagai nomor urut
    pilihan ("1" = materi pertama) — itu cara paling gampang bagi pengguna baru.
    """
    if not materials:
        return None

    if allow_ordinal:
        m = _BARE_NUMBER_RE.match(text or "")
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(materials):
                return materials[idx]

    folded = _fold(text)
    if not folded:
        return None
    haystack = set(folded.split(" "))

    best_hits = 0
    best_total = 0
    best_mat: dict | None = None
    for mat in materials:
        sf = mat.get("source_file") or ""
        stem = re.sub(r"\.[a-z0-9]{1,5}$", "", sf, flags=re.IGNORECASE)
        # Nama file utuh disebut → langsung menang.
        if _fold(sf) and _fold(sf) in folded:
            return mat
        cand = [t for t in _tokens(stem) if t not in _STOPWORDS and len(t) >= 4]
        if not cand:
            continue
        hits = sum(1 for t in cand if t in haystack)
        if hits > best_hits:
            best_hits, best_total, best_mat = hits, len(cand), mat
    # Minta minimal separuh kata cocok supaya tidak asal comot.
    if best_mat is not None and best_hits * 2 >= best_total:
        return best_mat
    return None


def strip_phrase(text: str, phrase: str) -> str:
    """Buang kata-kata `phrase` dari `text`, hasilnya dinormalisasi.

    Dipakai untuk membuang nama file materi SEBELUM mencari nomor minggu. Nama
    file sering memuat penanda minggu ("Materi SBD TM9.pptx", "bab3-minggu-2.pdf")
    yang bukan pilihan mahasiswa — kalau ikut terbaca, konteksnya melompat ke
    minggu yang salah dan materinya jadi tidak ketemu.
    """
    residual = _fold(text)
    for tok in _tokens(phrase):
        residual = re.sub(rf"\b{re.escape(tok)}\b", " ", residual)
    return re.sub(r"\s+", " ", residual).strip()


def _strip_refs(text: str, matched: list[str]) -> str:
    """Buang bagian yang sudah dikenali, sisanya dipakai menilai niat."""
    residual = _fold(text)
    for phrase in matched:
        for tok in _tokens(phrase):
            residual = re.sub(rf"\b{re.escape(tok)}\b", " ", residual)
    residual = _WEEK_RE.sub(" ", residual)
    return re.sub(r"\s+", " ", residual).strip()


#  Batas panjang agar frasa navigasi tidak membajak pertanyaan sungguhan.
#  "materi sbd minggu 3 apa saja?" (pendek) = minta daftar, tapi
#  "Apa itu DML dan perintah apa saja yang termasuk di dalamnya?" (panjang) =
#  pertanyaan — dan itu justru bunyi pertanyaan template yang kita hasilkan sendiri.
_NAV_MAX_CONTENT_TOKENS = 3


def looks_like_question(text: str) -> bool:
    """True kalau pesan ini terbaca sebagai pertanyaan yang harus dijawab.

    Frasa navigasi ("apa saja", "daftar") pada pesan PENDEK dianggap permintaan
    daftar pilihan, bukan pertanyaan. Pada pesan panjang frasa itu diabaikan.
    """
    folded = _fold(text)
    if not folded:
        return False
    if (_NAV_RE.search(folded)
            and len(_content_tokens(text)) <= _NAV_MAX_CONTENT_TOKENS):
        return False
    if "?" in (text or ""):
        return True
    toks = set(folded.split(" "))
    if not (toks & _QUESTION_WORDS):
        return False
    # Ada kata tanya, tapi pastikan ada isi yang ditanyakan — bukan cuma "apa".
    return len(_content_tokens(text)) >= 2


def wants_start(text: str) -> bool:
    """True untuk sapaan / "mau belajar" tanpa menyebut konteks apa pun."""
    folded = _fold(text)
    if not folded:
        return True
    if folded in _START_PHRASES:
        return True
    if len(_content_tokens(text)) == 0:
        return True
    return any(folded.startswith(p) or folded == p for p in _START_PHRASES) and (
        len(_content_tokens(text)) <= 2
    )


_RESET_PHRASES = (
    "ganti mata kuliah", "ganti matkul", "ganti materi", "ganti minggu",
    "mata kuliah lain", "materi lain", "minggu lain", "pilih lagi", "ulang",
    "mulai lagi", "mulai dari awal", "kembali", "menu", "batal", "reset",
)


_RESET_RE = _phrase_re(_RESET_PHRASES)


def wants_reset(text: str) -> bool:
    """True kalau mahasiswa ingin kembali memilih dari awal.

    Penting untuk pengguna baru: tanpa jalan keluar, mereka bisa merasa terjebak
    pada satu materi dan tidak tahu cara berpindah.
    """
    return bool(_RESET_RE.search(_fold(text)))


# --- Kembali satu langkah -------------------------------------------------------
# Berbeda dari "ulangi dari awal" (_RESET_PHRASES) yang membuang SELURUH konteks.
# Mahasiswa yang salah pilih minggu hampir selalu ingin mengganti minggunya saja,
# bukan mengulang dari pemilihan mata kuliah — memaksa mereka mengulang semuanya
# membuat satu salah klik terasa seperti hukuman.
_BACK_PHRASES = (
    "kembali", "balik", "back", "sebelumnya", "batal", "ganti pilihan",
    "salah pilih", "salah klik", "bukan itu", "mundur",
)
_BACK_RE = _phrase_re(_BACK_PHRASES)

# Langkah sebelum sebuah langkah. STEP_COURSE tidak punya pendahulu — ia sudah
# paling awal, jadi tombol kembali tidak ditampilkan di sana.
_PREVIOUS_STEP: dict[str, StepName] = {
    STEP_WEEK: STEP_COURSE,
    STEP_MATERIAL: STEP_WEEK,
    STEP_STYLE: STEP_MATERIAL,
    STEP_QUESTION: STEP_STYLE,
}

# Langkah tujuan yang sah bagi tombol kembali — dipakai juga untuk mengenali
# `value` yang dikirim balik klien.
BACK_TARGETS: frozenset[str] = frozenset(_PREVIOUS_STEP.values())

# Kalimat pada tombolnya. Menyebut TUJUAN, bukan sekadar "kembali", supaya
# mahasiswa tahu akan dibawa ke mana sebelum menekannya.
_BACK_LABELS: dict[str, str] = {
    STEP_COURSE: "Kembali — pilih mata kuliah lain",
    STEP_WEEK: "Kembali — pilih minggu lain",
    STEP_MATERIAL: "Kembali — pilih materi lain",
    STEP_STYLE: "Kembali — ganti cara belajar",
}


def wants_back(text: str) -> bool:
    """Apakah pesan ini berarti "kembali ke langkah sebelumnya".

    Menerima DUA bentuk, karena klien wajar mengirim salah satunya:
    label tombol apa adanya ("Kembali — pilih materi lain"), atau `value`-nya
    yang berupa nama langkah tujuan ("material"). Menerima label saja membuat
    klien yang mengirim `value` — pola yang dipakai untuk jenis pilihan lain —
    menekan tombol tanpa terjadi apa pun: kegagalan yang tidak memunculkan
    galat, jadi tidak ada yang menyadarinya sampai ada mahasiswa mengeluh.
    """
    bersih = _fold(text).strip()
    if bersih in _PREVIOUS_STEP.values():
        return True
    return bool(_BACK_RE.search(bersih))


def previous_step(step: str | None) -> StepName | None:
    """Langkah sebelum `step`; None bila sudah paling awal."""
    return _PREVIOUS_STEP.get(step or "")


def back_choice(step: str | None) -> Choice | None:
    """Tombol kembali untuk sebuah langkah; None bila tidak ada langkah sebelumnya.

    `value` diisi nama langkah TUJUAN, bukan kata "kembali", supaya klien yang
    mengirim balik nilainya tetap terbaca benar walau teksnya tidak dikenali
    pengenal maksud.
    """
    sebelum = previous_step(step)
    if sebelum is None:
        return None
    return Choice(
        label=_BACK_LABELS.get(sebelum, "Kembali"), value=sebelum, kind="back",
    )


_QUIZ_PHRASES = (
    "kuis", "quiz", "latihan soal", "latihan", "soal", "tes", "test",
    "uji pemahaman", "uji kemampuan", "ujian", "evaluasi",
)
_QUIZ_RE = _phrase_re(_QUIZ_PHRASES)

_QUIZ_STOP_PHRASES = ("berhenti", "stop", "keluar", "sudah", "batal kuis", "hentikan")
_QUIZ_STOP_RE = _phrase_re(_QUIZ_STOP_PHRASES)


def wants_quiz(text: str) -> bool:
    """True kalau mahasiswa meminta kuis atas materi yang sedang dibahas."""
    return bool(_QUIZ_RE.search(_fold(text)))


def wants_quiz_stop(text: str) -> bool:
    """True kalau mahasiswa ingin menghentikan kuis yang sedang berjalan."""
    return bool(_QUIZ_STOP_RE.search(_fold(text)))


def match_quiz_option(text: str, options: list[str]) -> int | None:
    """Cocokkan jawaban mahasiswa ke salah satu opsi kuis; None kalau tidak jelas.

    Menerima tiga bentuk sekaligus karena pengguna baru menjawab dengan cara
    yang berbeda-beda: huruf ("B"), nomor urut ("2"), atau menyalin/mengklik
    teks opsinya.
    """
    if not options:
        return None
    raw = (text or "").strip()

    # Huruf: A/B/C/D — hanya bila berdiri sendiri, supaya kata biasa tidak terbaca.
    huruf = re.fullmatch(r"\s*([a-zA-Z])[.)]?\s*", raw)
    if huruf:
        idx = ord(huruf.group(1).upper()) - 65
        if 0 <= idx < len(options):
            return idx

    # Nomor urut: 1..n
    angka = _BARE_NUMBER_RE.match(raw)
    if angka:
        idx = int(angka.group(1)) - 1
        if 0 <= idx < len(options):
            return idx

    # Teks opsi — termasuk bentuk berlabel "B. isi opsi" seperti yang kita kirim.
    folded = _fold(raw)
    if not folded:
        return None
    # Dicocokkan dalam bentuk tanpa penomoran, supaya tetap kena walau huruf yang
    # kita tampilkan berbeda dari huruf bawaan teks opsinya.
    bersih = _fold(_OPTION_PREFIX_RE.sub("", raw))
    for i, opt in enumerate(options):
        fo = _fold(strip_option_prefix(opt))
        if not fo:
            continue
        if fo in (folded, bersih) or fo in bersih or bersih in fo:
            return i
    return None


# Huruf/angka penomoran di awal teks opsi, mis. "A. ", "b) ", "1. ".
_OPTION_PREFIX_RE = re.compile(r"^\s*[A-Za-z0-9][.)]\s+")


def strip_option_prefix(option: str) -> str:
    """Buang penomoran bawaan dari teks opsi kuis.

    LLM kerap sudah menomori opsinya sendiri ("A. WHERE ..."). Tanpa dibuang,
    penomoran yang kita tambahkan membuat label jadi dobel: "A. A. WHERE ...".
    """
    return _OPTION_PREFIX_RE.sub("", option or "").strip()


def quiz_option_choices(options: list[str]) -> list[Choice]:
    """Opsi kuis sebagai tombol berlabel huruf."""
    return [
        Choice(label=f"{chr(65 + i)}. {strip_option_prefix(opt)}",
               value=str(i), kind="quiz")
        for i, opt in enumerate(options)
    ]


def quiz_question_message(index: int, total: int, question: str) -> str:
    return f"**Soal {index + 1} dari {total}**\n\n{question}"


def quiz_start_message(source_file: str, total: int) -> str:
    return (
        f"Baik, kita uji pemahamanmu tentang **{source_file}**.\n\n"
        f"Ada **{total} soal**. Pilih satu jawaban tiap soal — boleh klik tombolnya, "
        "atau ketik hurufnya saja. Ketik *berhenti* kalau mau batal."
    )


def quiz_result_message(score: float, correct: int, total: int, results: list[dict]) -> str:
    """Skor beserta pembahasan tiap soal.

    Pembahasan ditulis untuk SEMUA soal, bukan hanya yang salah, supaya mahasiswa
    yang menebak dengan benar tetap tahu alasannya.
    """
    if score >= 80:
        nada = "Bagus sekali!"
    elif score >= 60:
        nada = "Lumayan, tinggal sedikit lagi."
    else:
        nada = "Belum apa-apa, wajar — mari kita lihat bagian yang belum pas."

    baris = [
        f"**Skor kamu: {score:.0f}** — benar {correct} dari {total}. {nada}",
        "",
        "**Pembahasan:**",
    ]
    def label(opsi: list[str], idx: object) -> str:
        if not (isinstance(idx, int) and 0 <= idx < len(opsi)):
            return "tidak dijawab"
        return f"{chr(65 + idx)}. {strip_option_prefix(opsi[idx])}"

    for i, r in enumerate(results, 1):
        tanda = "BENAR" if r.get("is_correct") else "SALAH"
        opsi = r.get("options") or []
        baris.append(f"\n**{i}. {r.get('question', '')}** — {tanda}")
        if not r.get("is_correct"):
            baris.append(f"- Jawabanmu: {label(opsi, r.get('your_answer'))}")
        baris.append(f"- Jawaban benar: {label(opsi, r.get('correct_answer'))}")
        if r.get("explanation"):
            baris.append(f"- {r['explanation']}")
    baris.append("\nMau lanjut bertanya tentang materi ini, atau coba materi lain?")
    return "\n".join(baris)


def resolve_refs(
    text: str,
    courses: list[dict],
    *,
    awaiting: str | None = None,
    materials: list[dict] | None = None,
) -> Refs:
    """Kenali mata kuliah / minggu / materi yang disebut dalam satu pesan.

    `awaiting` adalah langkah yang sedang ditanyakan chatbot. Nilainya melonggarkan
    penafsiran: saat menanyakan minggu, "3" boleh berarti minggu 3; saat menanyakan
    materi, "2" boleh berarti materi kedua.
    """
    refs = Refs()

    course_id = match_course(text, courses)
    if course_id:
        refs.course_id = course_id
        names = [course_id]
        for c in courses:
            if c.get("course_id") == course_id and c.get("course_name"):
                names.append(c["course_name"])
        refs.matched.extend(names)

    refs.weeks = find_weeks(text, allow_bare_number=(awaiting == STEP_WEEK))

    if materials:
        mat = match_material(text, materials, allow_ordinal=(awaiting == STEP_MATERIAL))
        if mat:
            refs.source_file = mat.get("source_file")
            refs.content_id = mat.get("content_id")
            refs.matched.append(refs.source_file or "")

    refs.residual = _strip_refs(text, [m for m in refs.matched if m])
    return refs


def next_step(
    *,
    course_id: str | None,
    weeks: list[int] | None,
    source_file: str | None,
    style: str | None = None,
) -> StepName:
    """Langkah berikutnya berdasarkan kelengkapan konteks.

    Gaya belajar ditanyakan SETELAH materi dipilih: pada titik itu mahasiswa
    sudah tahu apa yang akan dipelajari, sehingga pertanyaan "mau dijelaskan
    dengan cara apa" menjadi masuk akal baginya.
    """
    if not course_id:
        return STEP_COURSE
    if not weeks:
        return STEP_WEEK
    if not source_file:
        return STEP_MATERIAL
    if not style:
        return STEP_STYLE
    return STEP_QUESTION


def format_weeks(weeks: list[int]) -> str:
    """Minggu terpilih dalam bentuk yang enak dibaca: "3", "3 dan 4", "3, 4, dan 5"."""
    urut = sorted(set(weeks))
    if not urut:
        return ""
    if len(urut) == 1:
        return str(urut[0])
    if len(urut) == 2:
        return f"{urut[0]} dan {urut[1]}"
    return ", ".join(str(w) for w in urut[:-1]) + f", dan {urut[-1]}"


# --- Kalimat yang diucapkan chatbot --------------------------------------------
def prompt_for(step: str, *, course_name: str | None = None,
               weeks: list[int] | None = None, source_file: str | None = None,
               topic: str | None = None) -> str:
    """Kalimat tanya-balik untuk sebuah langkah.

    Nada sengaja ramah dan menyebut bahwa pilihan bisa diklik — pengguna sasaran
    banyak yang belum pernah memakai chatbot.
    """
    if step == STEP_COURSE:
        return (
            "Halo! Saya bisa membantu kamu belajar dari materi kuliah yang sudah "
            "diunggah dosen.\n\n**Mau belajar mata kuliah apa?** Pilih salah satu "
            "di bawah, atau langsung tulis namanya."
        )
    if step == STEP_WEEK:
        nm = course_name or "mata kuliah ini"
        return (
            f"Baik, **{nm}**.\n\n**Minggu ke berapa?** Pilih salah satu di bawah, "
            "atau tulis angkanya saja. Boleh lebih dari satu — misalnya "
            "*minggu 3 dan 4* atau *minggu 2-4* kalau kamu sedang menyiapkan ujian."
        )
    if step == STEP_MATERIAL:
        nm = course_name or "mata kuliah ini"
        label = format_weeks(weeks or [])
        kepala = f"**{nm} — minggu {label}**" if label else f"**{nm}**"
        bahasan = f"\n\nYang dibahas: {topic}" if topic else ""
        return (
            f"{kepala} punya materi berikut.{bahasan}\n\n"
            "**Pilih materi yang mau kamu pelajari:**"
        )
    if step == STEP_STYLE:
        return (
            f"Materi **{source_file}** siap kita bahas.\n\n"
            "**Kamu ingin saya menjelaskannya dengan cara apa?** Pilih yang paling "
            "cocok denganmu — bisa diganti kapan saja dengan mengetik *ganti gaya*."
        )
    if step == STEP_QUESTION:
        return (
            f"Siap, kita bahas **{source_file}**.\n\nKalau belum tahu mau tanya apa, "
            "pilih salah satu pertanyaan di bawah. Kamu juga bebas menulis "
            "pertanyaanmu sendiri tentang materi ini."
        )
    return ""


def empty_message(step: str, *, course_name: str | None = None,
                  weeks: list[int] | None = None) -> str:
    """Pesan saat sebuah langkah tidak punya isi sama sekali."""
    if step == STEP_COURSE:
        return (
            "Belum ada materi yang terindex, jadi belum ada yang bisa saya bahas. "
            "Minta dosen mengunggah materinya dulu ya."
        )
    if step == STEP_WEEK:
        return (
            f"Belum ada minggu dengan materi terindex untuk **{course_name}**. "
            "Coba pilih mata kuliah lain."
        )
    return (
        f"Belum ada materi untuk **{course_name}** minggu {format_weeks(weeks or [])}. "
        "Coba minggu yang lain."
    )
