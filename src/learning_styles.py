"""Gaya belajar: cara LLM menjelaskan, dan alat apa yang boleh ia pakai.

Mahasiswa tidak belajar dengan cara yang sama. Sebagian paham lewat gambar,
sebagian lewat kode yang bisa dijalankan, sebagian butuh dituntun bertanya.
Alih-alih satu gaya penjelasan untuk semua, gaya belajar dipilih mahasiswa lalu
MENGGANTI SYSTEM PROMPT yang mengatur bagaimana LLM menjawab — termasuk keluaran
tambahan apa yang wajib ia hasilkan (diagram Mermaid, kode yang dapat diekspor
ke notebook).

Beda dengan `level` (sederhana/standar/detail) yang mengatur KEDALAMAN. Gaya
mengatur CARA. Keduanya bisa dipakai bersamaan: "visual" + "sederhana" berarti
diagram dengan bahasa yang mudah.

Sambutan tiap gaya juga berbeda supaya mahasiswa langsung tahu apa yang akan ia
dapatkan, bukan sekadar berganti nada.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_STYLE = "naratif"


@dataclass(frozen=True)
class LearningStyle:
    key: str
    label: str            # teks tombol
    description: str      # penjelasan singkat untuk mahasiswa
    system_suffix: str    # ditambahkan ke SYSTEM_PROMPT
    greeting: str         # kalimat pembuka setelah gaya dipilih
    starter_hint: str     # arahan tambahan saat membuat pertanyaan template
    produces_notebook: bool = False


_STYLES: list[LearningStyle] = [
    LearningStyle(
        key="naratif",
        label="Penjelasan bertahap",
        description="Dijelaskan runtut dari dasar, pakai analogi sehari-hari",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Uraikan secara runtut dari yang paling dasar. "
            "Mulai dari gambaran besar, baru masuk ke rinciannya. Gunakan analogi "
            "kehidupan sehari-hari untuk setiap konsep yang abstrak. Hindari "
            "melompat ke istilah teknis sebelum maknanya dijelaskan."
        ),
        greeting="Kita bahas pelan-pelan dari dasar, lengkap dengan analogi.",
        starter_hint="Pertanyaan harus mengundang penjelasan konsep dari dasar.",
    ),
    LearningStyle(
        key="visual",
        label="Lewat diagram",
        description="Setiap penjelasan disertai diagram alur atau hubungan",
        system_suffix=(
            "\n\nCARA MENJELASKAN: WAJIB sertakan minimal satu diagram Mermaid pada "
            "setiap jawaban, ditulis dalam blok kode berpagar dengan penanda bahasa "
            "`mermaid`. Pilih jenis diagram yang paling pas: `flowchart TD` untuk "
            "alur atau proses, `sequenceDiagram` untuk urutan interaksi, "
            "`erDiagram` untuk relasi antar entitas/tabel, `classDiagram` untuk "
            "struktur. Tulis label node dalam bahasa Indonesia dan bungkus dengan "
            "tanda kutip bila memuat spasi. Setelah diagram, jelaskan singkat "
            "bagian-bagiannya. Jangan membuat diagram untuk hal yang tidak ada di "
            "materi."
        ),
        greeting="Setiap penjelasan akan saya sertai diagram supaya alurnya terlihat.",
        starter_hint=(
            "Utamakan pertanyaan tentang alur, proses, tahapan, atau hubungan "
            "antar konsep — yang paling terbantu bila digambarkan."
        ),
    ),
    LearningStyle(
        key="praktik",
        label="Lewat contoh & kode",
        description="Banyak contoh nyata dan kode yang bisa langsung dicoba",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Utamakan contoh konkret dan kode yang dapat "
            "dijalankan. Tulis kode dalam blok berpagar dengan penanda bahasanya "
            "(`python`, `sql`, dan sebagainya). Setiap potongan kode harus dapat "
            "dijalankan sendiri dan disertai komentar berbahasa Indonesia pada "
            "baris yang penting. Sesudahnya jelaskan apa keluarannya dan mengapa "
            "demikian. Kalau materinya bukan pemrograman, ganti kode dengan contoh "
            "kasus yang sangat konkret beserta langkah pengerjaannya."
        ),
        greeting=(
            "Saya akan banyak memberi contoh dan kode yang bisa langsung kamu coba. "
            "Kode dalam jawaban dapat kamu unduh sebagai notebook."
        ),
        starter_hint="Utamakan pertanyaan 'bagaimana cara' dan penerapan nyata.",
        produces_notebook=True,
    ),
    LearningStyle(
        key="ringkas",
        label="Poin-poin ringkas",
        description="Padat dan langsung ke inti, cocok untuk mengulang cepat",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Jawab sepadat mungkin dalam bentuk poin-poin. "
            "Maksimal 8 poin, satu kalimat per poin. Dahulukan definisi, lalu "
            "perbedaan, lalu hal yang sering keliru dipahami. Tanpa basa-basi "
            "pembuka maupun penutup."
        ),
        greeting="Saya akan menjawab padat dalam poin-poin, cocok untuk mengulang.",
        starter_hint="Utamakan pertanyaan tentang definisi, perbedaan, dan ringkasan.",
    ),
    LearningStyle(
        key="sokratik",
        label="Dituntun bertanya",
        description="Saya balik bertanya supaya kamu menemukan jawabannya sendiri",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Jangan langsung memberikan jawaban penuh. "
            "Tuntun mahasiswa menemukannya sendiri: berikan satu petunjuk atau "
            "satu potong informasi kunci dari materi, lalu ajukan SATU pertanyaan "
            "pancingan yang membuatnya berpikir selangkah lebih maju. Baru setelah "
            "ia menjawab atau menyatakan menyerah, berikan penjelasan utuh. Tetap "
            "sertakan [Sumber N] untuk informasi yang kamu berikan."
        ),
        greeting=(
            "Saya akan lebih banyak bertanya balik supaya kamu menemukan sendiri "
            "jawabannya. Kalau buntu, bilang saja dan saya jelaskan."
        ),
        starter_hint="Pertanyaan harus terbuka dan mengundang penalaran.",
    ),
]

_BY_KEY: dict[str, LearningStyle] = {s.key: s for s in _STYLES}


def all_styles() -> list[LearningStyle]:
    return list(_STYLES)


def get(key: str | None) -> LearningStyle | None:
    """Gaya belajar berdasarkan key; None bila tidak dikenal."""
    if not key:
        return None
    return _BY_KEY.get(key.strip().lower())


def resolve(key: str | None) -> LearningStyle:
    """Seperti `get`, tetapi selalu mengembalikan sesuatu (jatuh ke default)."""
    return get(key) or _BY_KEY[DEFAULT_STYLE]


def choice_label(spec: LearningStyle) -> str:
    """Teks tombol untuk sebuah gaya — satu sumber, dipakai UI dan pengenalnya.

    Dulu bentuk ini dirakit di dua tempat (pipeline saat membuat tombol, dan
    tak dikenali sama sekali saat tombolnya diklik). Menyatukannya di sini yang
    membuat keduanya tidak mungkin lagi berbeda.
    """
    return f"{spec.label} — {spec.description}"


def is_choice_label(text: str) -> bool:
    """Apakah teks ini persis label tombol gaya yang kita tawarkan.

    Dipakai untuk membedakan KLIK TOMBOL dari PERTANYAAN. Tanpa pembedaan itu,
    label yang panjang dan berisi banyak kata isi — "Poin-poin ringkas — Padat
    dan langsung ke inti, cocok saat mengulang sebelum ujian" — lolos sebagai
    pertanyaan, lalu dijawab sungguhan. Mahasiswa menunggu dua menit untuk
    jawaban atas teks tombol yang baru saja ia tekan.
    """
    if not text:
        return False
    t = " ".join(text.strip().lower().split())
    for spec in _STYLES:
        if t in (choice_label(spec).lower(), spec.label.lower(), spec.key):
            return True
    return False


def _mengandung_kata(teks: str, frasa: str) -> bool:
    """Apakah `frasa` muncul sebagai kata utuh di dalam `teks`.

    "poin" cocok pada "kasih poin-poin saja" tetapi TIDAK pada "titik poinsettia".
    Pembedaan itulah yang membuat sebuah nama — mata kuliah, judul berkas —
    berhenti menetapkan gaya belajar hanya karena memuat potongan kata yang
    kebetulan sama.
    """
    return re.search(rf"(?<!\w){re.escape(frasa)}(?!\w)", teks) is not None


def match(text: str) -> str | None:
    """Kenali gaya belajar yang disebut mahasiswa dalam teks bebas.

    Dicocokkan ke key maupun label, supaya "visual", "lewat diagram", dan
    "pakai gambar" sama-sama dikenali.

    Pencocokan memakai BATAS KATA, bukan substring. Sempat memakai substring
    dan itu keliru dengan akibat yang tidak kelihatan: mata kuliah "Dasar
    Pemrograman" memuat kata "dasar", sehingga sekadar MEMILIH mata kuliah
    diam-diam menetapkan gaya belajar — dan langkah "mau dijelaskan dengan cara
    apa?" terlewat begitu saja tanpa ada yang menyadarinya.

    Isyarat yang terlalu umum ("dasar", "contoh", "program") sengaja tidak
    dipakai sama sekali: ketiganya jauh lebih sering muncul sebagai bagian nama
    mata kuliah atau pertanyaan biasa daripada sebagai permintaan gaya belajar.
    """
    if not text:
        return None
    t = text.strip().lower()

    for s in _STYLES:
        if _mengandung_kata(t, s.key) or _mengandung_kata(t, s.label.lower()):
            return s.key

    isyarat = {
        "diagram": "visual", "gambar": "visual", "bagan": "visual",
        "visual": "visual", "flowchart": "visual", "skema": "visual",
        "kode": "praktik", "coding": "praktik", "ngoding": "praktik",
        "praktek": "praktik", "praktik": "praktik",
        "notebook": "praktik", "ipynb": "praktik",
        "ringkas": "ringkas", "singkat": "ringkas", "padat": "ringkas",
        "poin": "ringkas", "rangkuman": "ringkas", "intinya": "ringkas",
        "tanya balik": "sokratik", "sokratik": "sokratik", "dituntun": "sokratik",
        "pelan": "naratif", "bertahap": "naratif", "analogi": "naratif",
        "cerita": "naratif",
    }
    for kata, key in isyarat.items():
        if _mengandung_kata(t, kata):
            return key
    return None
