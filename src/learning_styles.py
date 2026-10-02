"""Gaya belajar: cara LLM menjelaskan, dan alat apa yang boleh ia pakai.

Mahasiswa tidak belajar dengan cara yang sama. Sebagian paham lewat gambar,
sebagian lewat kode yang bisa dijalankan, sebagian butuh dituntun bertanya.
Alih-alih satu gaya penjelasan untuk semua, gaya belajar dipilih mahasiswa lalu
MENGGANTI SYSTEM PROMPT yang mengatur bagaimana LLM menjawab — termasuk keluaran
tambahan apa yang wajib ia hasilkan (diagram Mermaid, kode yang dapat diekspor
ke notebook).

Beda dengan `level` (simple/detailed/academic) yang mengatur KEDALAMAN. Gaya
mengatur CARA. Keduanya bisa dipakai bersamaan: "visual" + "simple" berarti
diagram dengan bahasa yang mudah.

MENGAPA BUKAN MODEL VARK
------------------------
Sistem ini sempat memuat dua daftar gaya: daftar di bawah, dan satu lagi memakai
model VARK (visual, auditory, reading, kinesthetic). VARK sengaja TIDAK dipakai
sebagai dasar, dengan alasan yang dapat dirujuk.

Klaim inti VARK adalah mencocokkan cara mengajar dengan "tipe" pembelajar akan
meningkatkan hasil belajar — yang dikenal sebagai meshing hypothesis. Tinjauan
Pashler dkk. (2008) menemukan klaim itu nyaris tidak punya dukungan eksperimen
yang memenuhi syarat, dan percobaan berikutnya oleh Rogowsky dkk. (2015) juga
tidak menemukan keuntungan dari pencocokan tersebut. Newton dan Miah (2017)
mencatat keyakinan pada model ini tetap luas di kalangan pengajar meski
buktinya tidak ada.

Yang dipakai di sini bukan tipe pembelajar, melainkan TEKNIK PENYAJIAN yang
masing-masing punya bukti tersendiri. Perbedaannya penting: mahasiswa memilih
bentuk penjelasan yang ia inginkan saat itu, bukan dilabeli sebagai satu tipe
yang menetap.

DASAR TIAP GAYA
---------------
- narrative  — analogi dan interogasi elaboratif. Analogi membantu memindahkan
  struktur dari ranah yang sudah dikenal ke yang baru (Gentner & Holyoak,
  1997); interogasi elaboratif dinilai cukup menjanjikan oleh Dunlosky dkk.
  (2013).
- visual     — prinsip multimedia: belajar lebih baik dari kata DAN gambar
  daripada kata saja (Mayer, 2014).
- practical  — efek contoh terkerjakan (worked example). Bagi pemula, mempelajari
  contoh yang sudah dikerjakan lebih efektif daripada langsung memecahkan
  soal (Sweller dkk., 2019; Renkl, 2014).
- structured — prinsip penandaan (signaling): penanda struktur seperti subjudul
  dan penomoran mengarahkan perhatian ke organisasi materi (Mayer, 2014).
- concise    — merangkum dan mengulang terjadwal; Dunlosky dkk. (2013) menilai
  praktik menguji diri sebagai teknik bermanfaat tinggi, dan ringkasan padat
  adalah bahan untuk itu.
- socratic   — menggali penjelasan diri. Mahasiswa yang diminta menjelaskan
  sendiri memahami lebih baik daripada yang hanya membaca (Chi dkk., 1994).

DAFTAR PUSTAKA
--------------
Chi, M. T. H., de Leeuw, N., Chiu, M.-H., & LaVancher, C. (1994). Eliciting
    self-explanations improves understanding. Cognitive Science, 18(3), 439-477.
Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T.
    (2013). Improving students' learning with effective learning techniques.
    Psychological Science in the Public Interest, 14(1), 4-58.
Gentner, D., & Holyoak, K. J. (1997). Reasoning and learning by analogy.
    American Psychologist, 52(1), 32-34.
Mayer, R. E. (2014). Cognitive theory of multimedia learning. Dalam R. E. Mayer
    (Ed.), The Cambridge Handbook of Multimedia Learning (2nd ed., hlm. 43-71).
    Cambridge University Press.
Newton, P. M., & Miah, M. (2017). Evidence-based higher education - is the
    learning styles 'myth' important? Frontiers in Psychology, 8, 444.
Pashler, H., McDaniel, M., Rohrer, D., & Bjork, R. (2008). Learning styles:
    Concepts and evidence. Psychological Science in the Public Interest, 9(3),
    105-119.
Renkl, A. (2014). Toward an instructionally oriented theory of example-based
    learning. Cognitive Science, 38(1), 1-37.
Rogowsky, B. A., Calhoun, B. M., & Tallal, P. (2015). Matching learning style to
    instructional method. Journal of Educational Psychology, 107(1), 64-78.
Sweller, J., van Merrienboer, J. J. G., & Paas, F. (2019). Cognitive
    architecture and instructional design: 20 years later. Educational
    Psychology Review, 31(2), 261-292.

Sambutan tiap gaya juga berbeda supaya mahasiswa langsung tahu apa yang akan ia
dapatkan, bukan sekadar berganti nada.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_STYLE = "narrative"


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
        key="narrative",
        label="Penjelasan bertahap",
        description="Dijelaskan runtut dari dasar, pakai analogi sehari-hari",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Uraikan secara runtut dari yang paling dasar. "
            "Mulai dari gambaran besar, baru masuk ke rinciannya. Gunakan analogi "
            "kehidupan sehari-hari untuk setiap konsep yang abstrak. Hindari "
            "melompat ke istilah teknis sebelum maknanya dijelaskan. Jelaskan "
            "seolah sedang bercerita: pakai narasi yang mengalir, berikan contoh "
            "berupa skenario atau dialog singkat, dan hubungkan satu konsep ke "
            "konsep berikutnya dengan kalimat transisi yang jelas."
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
            "setiap jawaban selama konsepnya memang punya struktur, alur, "
            "hubungan, atau perbandingan. Tulis diagram dalam blok kode berpagar "
            "berpenanda `mermaid`."
            "\nAturan diagram, patuhi persis karena kalau tidak diagramnya gagal "
            "dirender:"
            "\n- Pakai HANYA salah satu tipe ini: `flowchart TD`, `flowchart LR`, "
            "`sequenceDiagram`, `classDiagram`, `erDiagram` untuk relasi antar "
            "entitas atau tabel, atau `mindmap`."
            "\n- Bungkus SETIAP label simpul dengan tanda kutip ganda, contoh: "
            "A[\"Analisa Pasar\"]. Ini mencegah kegagalan render saat label memuat "
            "spasi, koma, atau tanda kurung."
            "\n- Jangan pakai tanda kurung, kurung siku, atau titik dua di DALAM "
            "teks label."
            "\n- Maksimal 12 simpul supaya tetap terbaca di layar ponsel."
            "\n- Tulis label dalam bahasa Indonesia."
            "\n- Beri satu kalimat pengantar sebelum diagram dan satu kalimat "
            "penutup sesudahnya, sehingga jawaban tetap utuh bagi mahasiswa yang "
            "tidak dapat melihat diagramnya."
            "\nJangan membuat diagram untuk hal yang tidak ada di materi. Bila "
            "topiknya murni satu definisi tanpa struktur, lewati diagram dan cukup "
            "pakai tabel perbandingan atau poin yang mudah dipindai."
        ),
        greeting="Setiap penjelasan akan saya sertai diagram supaya alurnya terlihat.",
        starter_hint=(
            "Utamakan pertanyaan tentang alur, proses, tahapan, atau hubungan "
            "antar konsep \u2014 yang paling terbantu bila digambarkan."
        ),
    ),
    LearningStyle(
        key="practical",
        label="Lewat contoh & kode",
        description="Banyak contoh nyata dan kode yang bisa langsung dicoba",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Utamakan contoh konkret dan kode yang dapat "
            "dijalankan. Mulai dari bagaimana cara melakukannya sebelum "
            "menjelaskan mengapa. Tulis kode dalam blok berpagar dengan penanda "
            "bahasanya (`python`, `sql`, dan sebagainya). Setiap potongan kode "
            "harus dapat dijalankan sendiri dan disertai komentar berbahasa "
            "Indonesia pada baris yang penting. Sesudahnya jelaskan apa "
            "keluarannya dan mengapa demikian. Sertakan skenario dunia nyata atau "
            "studi kasus singkat, dan hindari teori abstrak tanpa contoh "
            "penerapannya. Kalau materinya bukan pemrograman, ganti kode dengan "
            "contoh kasus yang sangat konkret beserta langkah pengerjaannya."
        ),
        greeting=(
            "Saya akan banyak memberi contoh dan kode yang bisa langsung kamu coba. "
            "Kode dalam jawaban dapat kamu unduh sebagai notebook."
        ),
        starter_hint="Utamakan pertanyaan 'bagaimana cara' dan penerapan nyata.",
        produces_notebook=True,
    ),
    LearningStyle(
        key="structured",
        label="Terstruktur & formal",
        description="Definisi formal dulu, lalu elaborasi bernomor dan bersubjudul",
        system_suffix=(
            "\n\nCARA MENJELASKAN: Mulai dari definisi formal, baru elaborasi. "
            "Pakai subjudul dan penomoran supaya urutannya terlihat, dan kutip "
            "langsung dari materi bila kutipannya memang menjelaskan. Mahasiswa "
            "ini nyaman dengan teks panjang dan rinci, jadi jangan dipangkas demi "
            "keringkasan. Tetap sertakan [Sumber N] pada setiap klaim."
        ),
        greeting="Saya akan menyusun jawaban secara formal dan bernomor.",
        starter_hint=(
            "Utamakan pertanyaan tentang definisi, klasifikasi, dan uraian "
            "bertingkat."
        ),
    ),
    LearningStyle(
        key="concise",
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
        key="socratic",
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

# Nama lama Indonesia dan nama model VARK yang dipakai portal kampus. Keduanya
# dikenali supaya klien dan profil mahasiswa yang sudah tersimpan tetap jalan,
# tetapi hanya kunci Inggris di atas yang diiklankan lewat GET /learning-styles.
#
# Pemetaan VARK: "auditory" ke narrative karena keduanya meminta narasi yang
# mengalir; "reading" ke structured karena keduanya meminta teks formal
# bersubjudul; "kinesthetic" ke practical karena keduanya meminta langkah
# konkret dan penerapan.
ALIAS_GAYA: dict[str, str] = {
    "naratif": "narrative",
    "praktik": "practical",
    "ringkas": "concise",
    "sokratik": "socratic",
    "auditory": "narrative",
    "reading": "structured",
    "kinesthetic": "practical",
}

_BY_KEY: dict[str, LearningStyle] = {s.key: s for s in _STYLES}


def all_styles() -> list[LearningStyle]:
    return list(_STYLES)


def get(key: str | None) -> LearningStyle | None:
    """Gaya belajar berdasarkan key; None bila tidak dikenal.

    Nama lama Indonesia dan nama model VARK ikut dikenali lewat ALIAS_GAYA,
    sehingga profil mahasiswa yang sudah tersimpan tidak perlu dimigrasi.
    """
    if not key:
        return None
    k = key.strip().lower()
    return _BY_KEY.get(ALIAS_GAYA.get(k, k))


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
        "kode": "practical", "coding": "practical", "ngoding": "practical",
        "praktek": "practical", "praktik": "practical",
        "notebook": "practical", "ipynb": "practical",
        "ringkas": "concise", "singkat": "concise", "padat": "concise",
        "poin": "concise", "rangkuman": "concise", "intinya": "concise",
        "formal": "structured", "terstruktur": "structured",
        "bernomor": "structured", "subjudul": "structured",
        "tanya balik": "socratic", "sokratik": "socratic", "dituntun": "socratic",
        "pelan": "narrative", "bertahap": "narrative", "analogi": "narrative",
        "cerita": "narrative",
    }
    for kata, key in isyarat.items():
        if _mengandung_kata(t, kata):
            return key
    return None
