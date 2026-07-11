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
  "source_filter": "Branching and Iteration - MIT.pdf" }
```
- **Pertanyaan template**: kirim teks pertanyaan yang diklik.
- **Pertanyaan rekomendasi**: ada di field `recommendations` pada response `/chat/ask` (otomatis dari jawaban sebelumnya).
- **Tanya bebas**: sama, mahasiswa ketik sendiri. `source_filter` boleh dilepas kalau mau cari se-minggu/se-matkul.

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
