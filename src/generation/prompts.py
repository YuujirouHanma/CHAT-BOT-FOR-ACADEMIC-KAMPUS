"""Prompt templates and context formatting for generation.

Critical design choice: when a retrieved chunk has a `raw_html` table or
`image_base64`, we inject the RAW DATA into the LLM context, not the summary.
The summary was only used for retrieval; the actual answer should be grounded
in the original data.

Each retrieved chunk becomes a `[Sumber N]` block the LLM can cite.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.schemas import ElementType

SYSTEM_PROMPT = """Anda adalah asisten pembelajaran untuk mahasiswa. Tugas Anda:

1. Menjawab pertanyaan HANYA berdasarkan konteks materi yang diberikan.
2. Jika konteks tidak cukup, katakan dengan jujur: "Materi yang tersedia tidak mencakup informasi tersebut."
3. Untuk pertanyaan yang melibatkan tabel atau gambar, baca data mentah yang disertakan dan kutip angka secara presisi.
4. Sebutkan sumber di setiap klaim penting dengan format [Sumber N], di mana N adalah nomor sumber yang Anda kutip.
5. Gunakan bahasa Indonesia yang jelas dan akademis. Jangan mengulang pertanyaan.
6. Jika ada angka, formula, atau definisi penting, sajikan dengan tepat."""


# Level gaya jawaban — ditambahkan ke SYSTEM_PROMPT. Untuk mahasiswa pelosok yang
# bingung, "sederhana" menjelaskan seperti ke pemula; "detail" untuk yang mau mendalam.
ANSWER_LEVEL_INSTRUCTIONS = {
    "sederhana": (
        "\n\nGAYA JAWABAN: Jelaskan dengan bahasa SANGAT SEDERHANA, seolah menjelaskan "
        "ke siswa SMA atau orang yang baru pertama belajar. Hindari istilah teknis; jika "
        "terpaksa memakainya, jelaskan artinya dengan kata sehari-hari. Pakai kalimat "
        "pendek dan analogi yang mudah dibayangkan. Tetap sertakan [Sumber N]."
    ),
    "standar": "",
    "detail": (
        "\n\nGAYA JAWABAN: Jelaskan secara MENDALAM untuk mahasiswa tingkat lanjut. "
        "Sertakan detail penting, istilah teknis yang tepat beserta nuansanya, dan bila "
        "relevan tunjukkan keterkaitan antar konsep. Tetap sertakan [Sumber N]."
    ),
}


def build_system_prompt(level: str | None = None, style: str | None = None) -> str:
    """SYSTEM_PROMPT + gaya belajar (CARA menjawab) + level (KEDALAMAN).

    Keduanya sengaja terpisah dan bisa digabung: "visual" + "sederhana" berarti
    diagram dengan bahasa yang mudah. Level tidak dikenal → prompt standar; gaya
    tidak dikenal → tanpa tambahan gaya sama sekali.
    """
    from src import learning_styles

    prompt = SYSTEM_PROMPT
    spec = learning_styles.get(style)
    if spec:
        prompt += spec.system_suffix
    return prompt + ANSWER_LEVEL_INSTRUCTIONS.get(level or "standar", "")


USER_PROMPT_TEMPLATE = """KONTEKS MATERI:
{context}

PERTANYAAN MAHASISWA:
{question}

JAWABAN (sertakan [Sumber N] untuk setiap klaim):"""


# --- Stage 1: Query decomposition (enrich query before retrieval) ---
DECOMPOSE_SYSTEM_PROMPT = (
    "Anda adalah asisten akademik. Diberikan sebuah pertanyaan dari mahasiswa, "
    "uraikan pertanyaan tersebut menjadi komponen terstruktur berikut (jawab dalam JSON):\n"
    "{\n"
    '  "topik_utama": "<topik atau materi yang paling relevan>",\n'
    '  "konsep_kunci": ["<konsep 1>", "<konsep 2>", ...],\n'
    '  "tipe_pertanyaan": "<definisi | penjelasan | contoh | perhitungan | perbandingan | lainnya>",\n'
    '  "query_diperkaya": "<versi pertanyaan yang lebih eksplisit dan informatif untuk pencarian materi>"\n'
    "}\n"
    "Jawab HANYA dengan JSON valid. Jangan coba menjawab pertanyaannya."
)


# --- Starter questions: template questions generated from a material's content ---
STARTER_SYSTEM_PROMPT = (
    "Anda adalah tutor yang membuat pertanyaan pembuka untuk mahasiswa yang "
    "BARU membuka sebuah materi kuliah dan belum tahu harus bertanya apa. "
    "Berdasarkan isi materi yang diberikan, buat tepat 5 pertanyaan pembuka yang: "
    "(1) mencakup konsep-konsep utama materi, (2) sederhana dan mengundang, "
    "cocok untuk mahasiswa yang baru belajar, (3) bisa dijawab dari materi itu. "
    "Jawab HANYA dalam JSON valid berbentuk array string, tanpa markdown, tanpa "
    'penjelasan. Contoh: ["Apa itu ...?", "Bagaimana cara ...?", "Mengapa ...?"]'
)


# --- Stage 5: Follow-up question generation (separate from the main answer) ---
# General-purpose: works for ANY subject. Enforces 3 DIFFERENT question TYPES so
# the recommendations are varied instead of three near-duplicates.
FOLLOWUP_SYSTEM_PROMPT = (
    "Anda adalah tutor yang membuat pertanyaan lanjutan untuk membantu mahasiswa belajar, "
    "untuk materi kuliah APA PUN (berlaku umum, bukan satu bidang tertentu). "
    "Buat tepat 3 pertanyaan lanjutan yang relevan dengan pertanyaan awal dan jawaban final. "
    "WAJIB: ketiganya harus dari TIPE yang BERBEDA — pilih 3 tipe berbeda dari daftar ini: "
    "definisi/konsep, contoh/penerapan, perbandingan/perbedaan, sebab-akibat/alasan, "
    "langkah/proses, atau analisis/evaluasi. "
    "Pertanyaan harus singkat, natural, dan mendorong pemahaman lebih dalam. "
    "\n\nJIKA diberikan [RIWAYAT PERCAKAPAN]: pertanyaan lanjutan harus MELANJUTKAN alur "
    "belajar itu — bangun di atas apa yang sudah dipahami mahasiswa, dan JANGAN mengulang "
    "pertanyaan yang sudah pernah ia tanyakan atau yang jawabannya sudah dibahas. "
    "Arahkan ke konsep berikutnya yang logis. "
    "\n\nJawab HANYA dalam JSON valid berbentuk array string, tanpa markdown, tanpa penjelasan. "
    'Contoh (tipe berbeda): ["Apa yang dimaksud dengan ...?", '
    '"Bagaimana penerapan ... dalam kasus nyata?", "Apa perbedaan ... dan ...?"]'
)


# --- Topik minggu: menyimpulkan apa yang dibahas dari isi materi ---
# Tanpa ini chatbot hanya bisa menyebut nama berkas ("Materi SBD TM9.pptx"),
# yang tidak memberi tahu mahasiswa apa pun tentang isinya.
WEEK_TOPIC_SYSTEM_PROMPT = (
    "Anda membaca kumpulan materi kuliah untuk satu atau beberapa minggu, lalu "
    "menyimpulkan TOPIK yang dibahas. Jawab dalam SATU kalimat bahasa Indonesia "
    "(maksimal 25 kata) yang menyebutkan konsep-konsep utamanya secara konkret, "
    "misalnya: 'Perintah DML pada SQL: INSERT, UPDATE, DELETE, serta operator "
    "perbandingan dan LIKE untuk memfilter data.' "
    "JANGAN memakai kalimat pembuka seperti 'Materi ini membahas'. "
    "Langsung sebutkan topiknya. Jangan mengarang isi yang tidak ada di materi."
)


def format_history(history: list[dict] | None, max_turns: int = 6) -> str:
    """Ringkas riwayat percakapan untuk prompt follow-up.

    Hanya `max_turns` pesan terakhir dipakai, dan jawaban asisten dipotong —
    yang dibutuhkan cuma alur topiknya, bukan isi lengkapnya, dan prompt panjang
    memakan max_tokens yang sudah ketat.
    """
    if not history:
        return ""
    lines = []
    for turn in history[-max_turns:]:
        role = "Mahasiswa" if turn.get("role") == "user" else "Tutor"
        text = (turn.get("content") or "").strip().replace("\n", " ")
        if role == "Tutor" and len(text) > 200:
            text = text[:200] + "…"
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


# --- Quiz: multiple-choice questions generated from a material's content ---
QUIZ_SYSTEM_PROMPT = (
    "Anda adalah pembuat soal kuis untuk mahasiswa, untuk materi kuliah APA PUN "
    "(berlaku umum). Berdasarkan isi materi yang diberikan, buat 5 soal PILIHAN GANDA yang: "
    "(1) menguji pemahaman konsep utama materi, (2) punya TEPAT 4 opsi jawaban, "
    "(3) hanya SATU jawaban benar, (4) sertakan penjelasan singkat mengapa jawaban itu benar. "
    "Jawab HANYA dalam JSON valid berbentuk array objek, tanpa markdown, tanpa teks lain. "
    "Setiap objek berbentuk: "
    '{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "..."} '
    "di mana answer_index adalah indeks (0-3) dari opsi yang benar."
)


@dataclass(frozen=True)
class FormattedContext:
    """Result of formatting retrieval results into LLM-ready content.

    Attributes:
        text_block: The textual context to put in the user message.
        image_payloads: List of {data, mime, source_idx} for images that
            should be sent as separate vision content blocks.
    """

    text_block: str
    image_payloads: list[dict[str, Any]]


def format_retrieval_results(results: list[dict]) -> FormattedContext:
    """Convert reranker output into formatted LLM context.

    For TEXT chunks: include the chunk text inline.
    For TABLE chunks: include raw_html so the LLM reads actual numbers.
    For IMAGE chunks: include the description inline AND emit a separate
        image payload that the caller can attach as a vision content block.

    Each source gets a 1-indexed citation tag [Sumber N].
    """
    text_parts: list[str] = []
    image_payloads: list[dict[str, Any]] = []

    for idx, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        element_type = payload.get("element_type", "text")
        source_file = payload.get("source_file", "unknown")
        page = payload.get("page_number")
        page_str = f", hal. {page}" if page is not None else ""
        header = f"[Sumber {idx}] {source_file}{page_str}"

        if element_type == ElementType.TABLE.value and payload.get("raw_html"):
            block = (
                f"{header} (TABEL — data asli HTML, baca angka langsung dari sini):\n"
                f"{payload['raw_html']}"
            )
        elif element_type == ElementType.IMAGE.value:
            description = payload.get("text", "")
            block = (
                f"{header} (GAMBAR — deskripsi dan gambar asli disertakan):\n"
                f"Deskripsi: {description}"
            )
            if payload.get("image_base64"):
                image_payloads.append(
                    {
                        "source_idx": idx,
                        "image_base64": payload["image_base64"],
                    }
                )
        else:
            text = payload.get("text", "")
            block = f"{header}:\n{text}"

        text_parts.append(block)

    return FormattedContext(
        text_block="\n\n---\n\n".join(text_parts),
        image_payloads=image_payloads,
    )


def build_user_prompt(question: str, context: FormattedContext) -> str:
    return USER_PROMPT_TEMPLATE.format(
        context=context.text_block,
        question=question.strip(),
    )


def parse_decompose_json(raw: str, fallback_question: str) -> dict[str, Any]:
    """Parse Stage 1 decomposition output. Falls back to the raw question on failure."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "topik_utama": fallback_question,
        "konsep_kunci": [fallback_question],
        "tipe_pertanyaan": "lainnya",
        "query_diperkaya": fallback_question,
    }


def parse_followup_json(raw: str, limit: int = 3) -> list[str]:
    """Parse a JSON array of question strings, capped at `limit` items."""
    clean = raw.replace("```json", "").replace("```", "").strip()

    suggestions: list[str] = []
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            suggestions = [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    suggestions = [str(x).strip() for x in data if str(x).strip()]
            except Exception:
                pass

    if not suggestions:
        suggestions = [
            re.sub(r"^\s*[\-\d\.\)\]]+\s*", "", line).strip()
            for line in clean.splitlines()
            if line.strip()
        ]

    return [s for s in suggestions if s][:limit]


def parse_quiz_json(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array of multiple-choice quiz items.

    Keeps only well-formed items: non-empty question, exactly 4 string options,
    and an answer_index within range. Returns [] if nothing valid is found.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    data: Any = None
    try:
        data = json.loads(clean)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None
    if not isinstance(data, list):
        return []

    quiz: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = item.get("options")
        answer_index = item.get("answer_index")
        if not question or not isinstance(options, list) or len(options) != 4:
            continue
        if not isinstance(answer_index, int) or not (0 <= answer_index < 4):
            continue
        quiz.append({
            "question": question,
            "options": [str(o) for o in options],
            "answer_index": answer_index,
            "explanation": str(item.get("explanation", "")).strip(),
        })
    return quiz

# --- Evaluasi berkala (kuis / ETS / EAS) atas rentang minggu -------------------
# Bentuk JSON per jenis soal. Ditulis eksplisit di prompt karena model cenderung
# menyeragamkan semua soal menjadi pilihan ganda bila bentuknya tidak dipaksakan.
_EVAL_SHAPES: dict[str, str] = {
    "pilihan_ganda":
        '{"type":"pilihan_ganda","question":"...","options":["A","B","C","D"],'
        '"answer_index":0,"explanation":"..."}',
    "benar_salah":
        '{"type":"benar_salah","question":"pernyataan yang dinilai benar/salah",'
        '"options":["Benar","Salah"],"answer_index":0,"explanation":"..."}',
    "isian_singkat":
        '{"type":"isian_singkat","question":"...","expected_answer":"jawaban ringkas",'
        '"key_points":["kata kunci wajib"],"explanation":"..."}',
    "esai":
        '{"type":"esai","question":"...","rubric":["aspek 1","aspek 2","aspek 3"],'
        '"explanation":"..."}',
    "koding":
        '{"type":"koding","question":"perintah membuat program","starter_code":"",'
        '"expected_behavior":"apa yang harus dilakukan program",'
        '"rubric":["benar secara logika","sesuai perintah"],"explanation":"..."}',
}

_EVAL_TYPE_NAMES: dict[str, str] = {
    "pilihan_ganda": "PILIHAN GANDA (tepat 4 opsi, satu jawaban benar)",
    "benar_salah": "BENAR/SALAH (pernyataan, answer_index 0=Benar 1=Salah)",
    "isian_singkat": "ISIAN SINGKAT (jawaban 1-2 kalimat)",
    "esai": "ESAI (jawaban uraian, sertakan rubrik penilaian)",
    "koding": "KODING (mahasiswa menulis program)",
}


def build_evaluation_prompt(
    kind_label: str, range_text: str, counts: dict[str, int],
) -> str:
    """Susun instruksi pembuatan evaluasi dari komposisi soal yang diminta.

    Prompt dibangun dari `counts`, bukan ditulis tetap, supaya dosen yang
    mengubah komposisi tidak perlu menyentuh kode — dan supaya komposisi yang
    benar-benar dipakai pada sebuah penelitian dapat dilaporkan apa adanya.
    """
    diminta = [(t, n) for t, n in counts.items() if n > 0]
    rincian = "\n".join(
        f"- {n} soal {_EVAL_TYPE_NAMES.get(t, t.upper())}" for t, n in diminta
    )
    bentuk = "\n".join(
        f"  {t}: {_EVAL_SHAPES[t]}" for t, _ in diminta if t in _EVAL_SHAPES
    )
    total = sum(n for _, n in diminta)

    return (
        f"Anda adalah penyusun soal {kind_label} untuk mahasiswa, untuk materi "
        f"kuliah APA PUN (berlaku umum). Materi yang diberikan mencakup "
        f"{range_text}.\n\n"
        f"Susun TEPAT {total} soal dengan komposisi:\n{rincian}\n\n"
        "Ketentuan:\n"
        "1. Soal HARUS berdasarkan isi materi yang diberikan, bukan pengetahuan umum.\n"
        "2. Karena materi mencakup beberapa minggu, sertakan soal yang menuntut "
        "mahasiswa MENGHUBUNGKAN konsep antar minggu, bukan hanya mengingat satu bagian.\n"
        "3. Urutkan dari yang mudah ke yang sulit.\n"
        "4. Setiap soal wajib memuat `explanation` berisi alasan jawabannya.\n\n"
        "Jawab HANYA dengan JSON valid berupa array objek, tanpa markdown, tanpa "
        "teks pembuka maupun penutup. Bentuk tiap objek menurut jenisnya:\n"
        f"{bentuk}"
    )


def _clean_json_array(raw: str) -> Any:
    """Ambil array JSON dari keluaran model, tahan terhadap pagar markdown."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", clean)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _parse_choice_item(item: dict, tipe: str) -> dict[str, Any] | None:
    """Butir berbasis opsi (pilihan ganda / benar-salah)."""
    options = item.get("options")
    idx = item.get("answer_index")
    if tipe == "benar_salah" and not isinstance(options, list):
        options = ["Benar", "Salah"]      # model kerap menghilangkannya
    if not isinstance(options, list) or len(options) < 2:
        return None
    if tipe == "pilihan_ganda" and len(options) != 4:
        return None
    if not isinstance(idx, bool) and isinstance(idx, int) and 0 <= idx < len(options):
        return {"options": [str(o) for o in options], "answer_index": idx}
    return None


def parse_evaluation_json(raw: str) -> list[dict[str, Any]]:
    """Parse array soal evaluasi bercampur jenis.

    Butir yang bentuknya tidak lengkap DIBUANG, bukan diperbaiki dengan tebakan.
    Soal evaluasi menentukan nilai mahasiswa; menambal butir yang cacat berarti
    menilai dengan soal yang tidak pernah benar-benar disusun.
    """
    data = _clean_json_array(raw)
    if not isinstance(data, list):
        return []

    hasil: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tipe = str(item.get("type", "pilihan_ganda")).strip().lower()
        question = str(item.get("question", "")).strip()
        if not question or tipe not in _EVAL_SHAPES:
            continue

        butir: dict[str, Any] = {
            "type": tipe,
            "question": question,
            "explanation": str(item.get("explanation", "")).strip(),
        }

        if tipe in ("pilihan_ganda", "benar_salah"):
            inti = _parse_choice_item(item, tipe)
            if inti is None:
                continue
            butir.update(inti)
        elif tipe == "isian_singkat":
            jawab = str(item.get("expected_answer", "")).strip()
            if not jawab:
                continue
            kunci = item.get("key_points")
            butir["expected_answer"] = jawab
            butir["key_points"] = (
                [str(k) for k in kunci] if isinstance(kunci, list) else []
            )
        elif tipe == "esai":
            rubrik = item.get("rubric")
            butir["rubric"] = (
                [str(r) for r in rubrik] if isinstance(rubrik, list) else []
            )
        elif tipe == "koding":
            butir["starter_code"] = str(item.get("starter_code", ""))
            butir["expected_behavior"] = str(item.get("expected_behavior", "")).strip()
            rubrik = item.get("rubric")
            butir["rubric"] = (
                [str(r) for r in rubrik] if isinstance(rubrik, list) else []
            )

        hasil.append(butir)
    return hasil


GRADE_SYSTEM_PROMPT = (
    "Anda adalah pemeriksa jawaban mahasiswa. Nilai jawaban terhadap kunci atau "
    "rubrik yang diberikan, dengan adil dan konsisten.\n"
    "Ketentuan:\n"
    "1. Nilai ISI, bukan gaya bahasa atau panjang jawaban.\n"
    "2. Jawaban benar yang diungkapkan dengan kata berbeda tetap bernilai penuh.\n"
    "3. Beri `skor` antara 0.0 dan 1.0 - boleh pecahan untuk jawaban benar sebagian.\n"
    "4. `feedback` ditujukan KEPADA MAHASISWA: sebut apa yang sudah tepat dan apa "
    "yang kurang, maksimal 2 kalimat, memakai kata ganti orang kedua.\n"
    "5. Bila Anda ragu - jawaban ambigu, di luar cakupan kunci, atau menuntut "
    "penilaian manusia - isi `ragu` dengan true. Menandai ragu JAUH lebih baik "
    "daripada menebak, karena butir itu akan ditinjau dosen.\n"
    "Jawab HANYA JSON: "
    '{"skor": 0.0, "benar": false, "ragu": false, "feedback": "..."}'
)


def parse_grade_json(raw: str) -> dict[str, Any] | None:
    """Parse hasil penilaian LLM atas satu jawaban terbuka.

    Mengembalikan None bila tidak terbaca — pemanggil menandainya sebagai perlu
    tinjauan dosen, bukan memberinya nilai nol. Kegagalan mesin bukan kesalahan
    mahasiswa.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    data: Any = None
    try:
        data = json.loads(clean)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", clean)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return None
    try:
        skor = float(data.get("skor", 0.0))
    except (TypeError, ValueError):
        return None
    skor = max(0.0, min(1.0, skor))
    return {
        "skor": skor,
        # `benar` berarti BENAR PENUH, dan harus sejalan dengan skornya.
        # Model kerap mengembalikan benar=true bersama skor 0.5 — sekadar
        # bermaksud "arahnya sudah betul". Mempercayainya apa adanya membuat
        # hitungan "berapa soal yang benar" menggelembung, dan pada penelitian
        # angka itulah yang dilaporkan. Skor tetap menyimpan nilai parsialnya.
        "benar": bool(data.get("benar", True)) and skor >= 0.999,
        "ragu": bool(data.get("ragu", False)),
        "feedback": str(data.get("feedback", "")).strip(),
    }


# --- Tinjauan kode mahasiswa (livecode) ----------------------------------------
# Berbeda dari GRADE_SYSTEM_PROMPT yang menilai jawaban tulisan: di sini yang
# dinilai adalah program, dan tujuannya BELAJAR — bukan sekadar memberi angka.
CODE_REVIEW_SYSTEM_PROMPT = (
    "Anda adalah asisten dosen yang memeriksa program mahasiswa pemula.\n"
    "Kode TIDAK dijalankan; nilailah dengan membaca dan menalar alurnya.\n\n"
    "Ketentuan:\n"
    "1. Tentukan apakah program memenuhi perilaku yang diminta. Perbedaan gaya "
    "penulisan, nama variabel, atau pendekatan (iteratif vs rekursif) BUKAN "
    "kesalahan selama hasilnya benar.\n"
    "2. Sebutkan lebih dulu apa yang SUDAH BENAR. Mahasiswa pemula yang hanya "
    "menerima daftar kesalahan cenderung berhenti mencoba.\n"
    "3. Untuk tiap kekeliruan, sebut letaknya (nomor baris bila jelas) dan "
    "AKIBATNYA pada hasil program - bukan sekadar 'salah'.\n"
    "4. Beri `petunjuk` yang mengarahkan mahasiswa menemukan sendiri "
    "perbaikannya. JANGAN menuliskan kode perbaikan yang lengkap: menyodorkan "
    "jawaban menyelesaikan soalnya, tetapi menghapus proses belajarnya.\n"
    "5. `skor` antara 0.0 dan 1.0. Program yang logikanya benar tetapi kurang "
    "rapi tetap bernilai tinggi.\n"
    "6. Isi `ragu` dengan true bila Anda tidak yakin - misalnya alurnya terlalu "
    "rumit untuk ditelusuri tanpa menjalankannya. Ditinjau dosen jauh lebih "
    "baik daripada dinilai asal.\n\n"
    "Jawab HANYA JSON:\n"
    '{"lulus": false, "skor": 0.0, "ragu": false, "ringkasan": "...", '
    '"benar": ["..."], "keliru": [{"baris": 3, "masalah": "...", "akibat": "..."}], '
    '"petunjuk": ["..."]}'
)


def parse_code_review_json(raw: str) -> dict[str, Any] | None:
    """Parse hasil tinjauan kode. None bila tidak terbaca.

    None ditangani pemanggil sebagai perlu tinjauan dosen, bukan sebagai nilai
    nol — kegagalan mesin bukan kesalahan mahasiswa.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()
    data: Any = None
    try:
        data = json.loads(clean)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", clean)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return None

    try:
        skor = float(data.get("skor", 0.0))
    except (TypeError, ValueError):
        skor = 0.0

    def _daftar_teks(nilai: Any) -> list[str]:
        return [str(x).strip() for x in nilai if str(x).strip()] if isinstance(nilai, list) else []

    keliru: list[dict[str, Any]] = []
    for k in (data.get("keliru") or []):
        if isinstance(k, dict) and str(k.get("masalah", "")).strip():
            baris = k.get("baris")
            keliru.append({
                "baris": baris if isinstance(baris, int) else None,
                "masalah": str(k["masalah"]).strip(),
                "akibat": str(k.get("akibat", "")).strip(),
            })
        elif isinstance(k, str) and k.strip():
            keliru.append({"baris": None, "masalah": k.strip(), "akibat": ""})

    return {
        "lulus": bool(data.get("lulus", False)),
        "skor": max(0.0, min(1.0, skor)),
        "ragu": bool(data.get("ragu", False)),
        "ringkasan": str(data.get("ringkasan", "")).strip(),
        "benar": _daftar_teks(data.get("benar")),
        "keliru": keliru,
        "petunjuk": _daftar_teks(data.get("petunjuk")),
    }
