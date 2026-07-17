# Guided Catalog — Alur Menuntun (Mata Kuliah → Minggu → Materi)

Fitur ini untuk mahasiswa yang **belum terbiasa** memakai chatbot AI (target kampus pelosok).
Alih-alih halaman chat kosong, frontend menampilkan **pilihan siap-klik** bertahap.

Semua daftar **diturunkan dari materi yang benar-benar terindex**, jadi:
- **Auto-update**: begitu materi minggu baru diupload, minggu itu langsung muncul (tidak ada hardcode "sampai minggu 16").
- **Tanpa jalan buntu**: mahasiswa hanya melihat minggu/materi yang memang bisa dijawab.

Semua endpoint butuh header `X-API-Key` (lihat [API_INTEGRATION.md](../API_INTEGRATION.md)).

---

## Alur & Endpoint

```
1. Pilih Mata Kuliah  →  GET /catalog/courses
2. Pilih Minggu       →  GET /catalog/courses/{course_id}/weeks
3. Pilih Materi       →  GET /catalog/courses/{course_id}/weeks/{week}/materials
4. Pertanyaan Template→  GET /catalog/materials/{content_id}/{source_file}/starter-questions
5. Tanya (klik/bebas) →  POST /chat/ask   (sudah ada; kirim content_id + source_filter)
```

### 1. Daftar mata kuliah
`GET /catalog/courses`
```json
{ "courses": [
  { "course_id": "sbd", "course_name": "SBD" },
  { "course_id": "kka", "course_name": "KKA" }
] }
```

### 2. Daftar minggu (otomatis, terurut)
`GET /catalog/courses/sbd/weeks`
```json
{ "course_id": "sbd", "weeks": [1, 2] }
```
> Kalau minggu 3 diupload nanti → response otomatis jadi `[1, 2, 3]`. Tidak perlu ubah apa pun.

### 3. Daftar materi satu minggu
`GET /catalog/courses/sbd/weeks/2/materials`
```json
{ "course_id": "sbd", "week": 2, "materials": [
  { "source_file": "Branching and Iteration - MIT.pdf", "content_id": "sbd-minggu-2" }
] }
```
> `content_id` + `source_file` di sini dipakai untuk 2 langkah berikutnya.

### 4. Pertanyaan template (auto-generate + cache)
`GET /catalog/materials/sbd-minggu-2/Branching and Iteration - MIT.pdf/starter-questions`
```json
{ "content_id": "sbd-minggu-2",
  "source_file": "Branching and Iteration - MIT.pdf",
  "questions": [
    "Bagaimana cara program membuat keputusan berdasarkan kondisi Boolean?",
    "Apa perbedaan antara while loop dan for loop?",
    "..."
  ] }
```
> Dibuat AI dari isi materi saat pertama diminta, lalu **di-cache** (panggilan berikutnya instan).
> `source_file` mengandung spasi/karakter khusus → **URL-encode** saat memanggil.

### 5. Bertanya
Pakai endpoint chat yang sudah ada. Untuk membatasi jawaban ke materi terpilih, kirim
`content_id` **dan** `source_filter` (= `source_file`):
```json
POST /chat/ask
{ "question": "Apa perbedaan while dan for loop?",
  "content_id": "sbd-minggu-2",
  "source_filter": "Branching and Iteration - MIT.pdf",
  "model": "qwen3.7-plus" }
```
- **Pertanyaan template**: kirim teks pertanyaan yang diklik.
- **Pertanyaan rekomendasi**: ada di field `recommendations` — selalu **3 tipe berbeda** (definisi, contoh/penerapan, perbandingan, dst.), otomatis dari jawaban sebelumnya.
- **Tanya bebas**: sama, mahasiswa ketik sendiri. `source_filter` boleh dilepas kalau mau cari se-minggu/se-matkul.
- **`model`** (opsional): key model dari `GET /models`. Kosong = pakai model default server.
- **`level`** (opsional): gaya jawaban — `"sederhana"` (bahasa mudah, untuk pemula/mahasiswa
  bingung), `"standar"` (default, akademis), atau `"detail"` (mendalam/teknis). Cocok untuk
  tombol **"Sederhanakan"**: kirim ulang pertanyaan yang sama dengan `level: "sederhana"`.

### 6. Kuis (opsional, per materi)
`GET /catalog/materials/{content_id}/{source_file}/quiz` — kuis pilihan ganda auto-generate + cache.
```json
{ "content_id": "sbd-minggu-2", "source_file": "...",
  "questions": [
    { "question": "Apa prinsip Stack?",
      "options": ["FIFO", "LIFO", "Random", "FILO"],
      "answer_index": 1,
      "explanation": "Stack memakai prinsip LIFO." }
  ] }
```
`answer_index` = indeks opsi benar (0-3). Tambah `?model=<key>` untuk pilih LLM.

**Nilai jawaban** — `POST /catalog/materials/{content_id}/{source_file}/quiz/submit`
```json
// request
{ "answers": [1, 0, 2, 1, 1], "session_id": "opsional", "student_id": "opsional" }
// response
{ "content_id": "...", "source_file": "...",
  "total": 5, "correct": 3, "score": 60.0, "attempt_id": "quiz_2026...",
  "results": [
    { "question": "...", "options": ["..."], "your_answer": 1,
      "correct_answer": 1, "is_correct": true, "explanation": "..." }
  ] }
```
`answers` = indeks opsi yang dipilih mahasiswa, **urut sesuai soal** dari GET quiz. Dinilai
terhadap kuis yang sama (cache), `score` = 0-100. Setiap attempt dicatat ke
`data/hitl_logs/quiz_attempts.jsonl` (progres bisa direview dosen). Soal tak dijawab = salah.

---

## Ganti model dari UI (multi-provider)

`GET /models` → daftar model yang **bisa dipakai** (hanya provider yang punya API key di server):
```json
{ "default": "qwen3.7-plus",
  "models": [
    { "key": "qwen3.7-plus", "label": "Qwen 3.7 Plus", "provider": "openrouter", "vision": true },
    { "key": "gpt-4o-mini", "label": "GPT-4o mini", "provider": "openai", "vision": true },
    { "key": "groq-llama-70b", "label": "Llama 3.3 70B (Groq)", "provider": "groq", "vision": false }
  ] }
```
UI menampilkan dropdown dari `models`, lalu kirim `key` yang dipilih sebagai field **`model`** di
`/chat/ask`, atau `?model=<key>` di starter-questions & quiz. Provider didukung: **OpenRouter (Qwen),
OpenAI, Groq, Gemini** — cukup isi API key provider-nya di `.env`, model langsung muncul di `/models`.
Model yang belum ada key-nya otomatis **tidak ditampilkan** (UI tak akan menawarkan yang bakal gagal).

---

## Yang perlu tim BE lakukan saat upload

Agar hierarki rapi, kirim 3 field **opsional** ini di `POST /documents/upload`
(selain `file` dan `content_id`):

| Field | Contoh | Wajib? |
|---|---|---|
| `course_id` | `sbd` | opsional |
| `course_name` | `Sistem Basis Data` | opsional |
| `week` | `2` | opsional |

**Kalau tidak dikirim**, backend menurunkannya otomatis dari `content_id` dengan konvensi:
`"<course>-minggu-<n>"` (juga menerima `week`/`w` dan pemisah `_` atau `/`).

| content_id | course_id | week |
|---|---|---|
| `sbd-minggu-2` | `sbd` | 2 |
| `sbd_minggu_2` | `sbd` | 2 |
| `algo-w3` | `algo` | 3 |
| `class-3` | `class-3` | (tak ada minggu) |

> **Rekomendasi:** kirim `course_name` eksplisit untuk nama tampilan yang bagus
> (mis. "Sistem Basis Data" daripada auto "SBD"). Materi lama yang sudah terindex
> **sebelum** fitur ini perlu **di-upload ulang** agar punya course/week.

---

## Backward compatibility

- Endpoint lama (`/chat/ask`, `/documents/upload`, `/browse/*`) **tidak berubah** kontraknya.
- Field upload `course_id`/`course_name`/`week` bersifat **additive** — call lama tetap jalan.
- Frontend bebas memakai alur guided ini **atau** tetap pakai tanya-bebas seperti sekarang.
