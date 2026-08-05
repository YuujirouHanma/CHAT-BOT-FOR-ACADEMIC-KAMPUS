# Catatan Perubahan untuk Tim Back-End

**Untuk:** tim Back-End / Front-End yang memanggil layanan RAGAcademic
**Sejak:** integrasi awal 1 Juli 2026 (`API_INTEGRATION.md` versi pertama)
**Per:** 5 Agustus 2026 — layanan versi 0.3.0
**Kontrak terbaru:** `docs/api/openapi.json` (sudah diperbarui, 14 endpoint)

---

## RINGKAS: yang wajib dikerjakan tim BE

| Prioritas | Tindakan | Alasan |
|---|---|---|
| **1 — WAJIB** | Tangani field `mode` pada respons `POST /chat/ask`, **atau** kirim `"guided": false` | Tanpa ini, pengguna bisa menemui pesan chatbot yang menunggu jawaban pilihan padahal tombolnya tidak dirender |
| **2 — WAJIB sebelum produksi** | Pastikan `RAGACADEMIC_API_KEY` terisi dan kirim header `X-API-Key` | Auth sedang dimatikan untuk pengembangan; di `APP_ENV=production` server menolak start bila key kosong |
| 3 — Disarankan | Kirim `course_id`, `course_name`, `week` saat upload, atau ikuti pola `content_id` = `<matkul>-minggu-<n>` | Menentukan apakah materi muncul di navigasi mata kuliah → minggu → materi |
| 4 — Opsional | Pakai endpoint `/catalog/*`, `/models`, dan kuis | Fitur baru yang sudah tersedia |

---

## 1. Perubahan pada `POST /chat/ask` — paling penting

Endpoint ini sekarang punya **dua bentuk balasan**, dibedakan oleh field baru `mode`.

### Bentuk lama (masih ada)

```json
{ "mode": "answer", "step": "answer",
  "answer": "...jawaban dengan [Sumber 1]...",
  "sources": [...], "recommendations": [...],
  "session_id": "...", "interaction_id": "...",
  "choices": [], "context": {...} }
```

### Bentuk baru — chatbot bertanya balik

```json
{ "mode": "choices", "step": "course",
  "answer": "Mau belajar mata kuliah apa?",
  "sources": [], "recommendations": [],
  "session_id": "...",
  "choices": [
    { "label": "Sistem Basis Data", "value": "sbd", "kind": "course" },
    { "label": "KKA", "value": "kka", "kind": "course" }
  ],
  "context": { "course_id": null, "course_name": null, "week": null,
               "content_id": null, "source_file": null } }
```

**Yang harus dilakukan klien:**

- `mode == "answer"` → render seperti sebelumnya.
- `mode == "choices"` → render `answer` sebagai pesan, lalu **render `choices` sebagai tombol**.

Ketika pengguna mengklik sebuah tombol, kirim balik `label`-nya sebagai `question` biasa, ditambah field eksplisit sesuai `kind`:

| `kind` | Kirim bersama `question` | Contoh |
|---|---|---|
| `course` | `"course_id": <value>` | `{"question": "Sistem Basis Data", "course_id": "sbd"}` |
| `week` | `"week": <value sebagai angka>` | `{"question": "Minggu 3", "week": 3}` |
| `material` | `"source_filter": <value>` | `{"question": "bab3.pdf", "source_filter": "bab3.pdf"}` |
| `question` | tidak ada tambahan | `{"question": "Apa itu DML?"}` |
| `quiz` | tidak ada tambahan | `{"question": "kuis"}` |

Field eksplisit itu **opsional** — server juga mengenali maksud dari teksnya. Mengirim keduanya adalah yang paling aman.

### Kalau belum sempat menyesuaikan

Kirim `"guided": false` pada setiap permintaan. Balasan akan selalu `mode: "answer"` seperti sebelumnya, tanpa tanya-balik. Perilaku lama tetap utuh.

### Field baru pada request

| Field | Tipe | Keterangan |
|---|---|---|
| `guided` | bool, default `true` | `false` = matikan tanya-balik |
| `course_id` | string, opsional | diisi saat pengguna mengklik pilihan mata kuliah |
| `week` | int 1–52, opsional | diisi saat pengguna mengklik pilihan minggu |
| `model` | string, opsional | key dari `GET /models`; ganti model per permintaan |
| `level` | `sederhana` \| `standar` \| `detail` | kedalaman jawaban |

### Field baru pada response

| Field | Keterangan |
|---|---|
| `mode` | `answer` atau `choices` |
| `step` | `course`, `week`, `material`, `question`, `answer`, `quiz` |
| `choices` | daftar `{label, value, kind}` untuk dirender sebagai tombol |
| `context` | `{course_id, course_name, week, content_id, source_file}` — untuk breadcrumb |

`session_id` **wajib** dikirim balik pada permintaan lanjutan. Selain menyimpan riwayat percakapan, di dalamnya juga tersimpan materi yang sedang dipilih dan status kuis yang sedang berjalan.

---

## 2. Kuis kini berjalan di dalam percakapan

Tidak ada endpoint baru. Selama kuis berlangsung, `POST /chat/ask` membalas dengan `step: "quiz"` dan `choices` berisi opsi jawaban (`kind: "quiz"`), satu soal per pesan.

Alurnya:

1. Setelah materi dipilih, respons memuat satu pilihan `kind: "quiz"` berlabel *"Kerjakan kuis materi ini"*.
2. Klien mengirim `{"question": "kuis"}` → server membalas soal 1 beserta opsinya.
3. Klien mengirim jawaban (label opsi, huruf `A`, atau nomor) → server membalas soal berikutnya.
4. Setelah soal terakhir → server membalas skor dan pembahasan tiap soal, `step` kembali ke `question`.

Ketik *berhenti* untuk membatalkan kuis di tengah jalan. Jawaban yang tidak dikenali **tidak** dihitung salah — server menanyakan ulang soal yang sama.

Endpoint kuis lama (`GET /catalog/materials/.../quiz` dan `.../quiz/submit`) **tetap ada** dan tidak berubah, untuk klien yang ingin menampilkan kuis di halaman terpisah.

---

## 3. Endpoint yang bertambah sejak integrasi awal

| Endpoint | Guna |
|---|---|
| `GET /catalog/courses` | daftar mata kuliah yang punya materi terindeks |
| `GET /catalog/courses/{course_id}/weeks` | minggu yang tersedia |
| `GET /catalog/courses/{course_id}/weeks/{week}/materials` | materi per minggu |
| `GET /catalog/materials/{content_id}/{source_file}/starter-questions` | 5 pertanyaan pembuka dari isi materi |
| `GET /catalog/materials/{content_id}/{source_file}/quiz` | kuis pilihan ganda |
| `POST /catalog/materials/{content_id}/{source_file}/quiz/submit` | penilaian + pembahasan |
| `GET /models` | daftar model LLM yang boleh dipilih per permintaan |

Semua daftar di atas diturunkan dari materi yang **benar-benar terindeks**, sehingga tidak pernah menampilkan minggu atau materi kosong.

---

## 4. Kemampuan ingestion yang bertambah

`POST /documents/upload` sekarang menerima lebih banyak jenis berkas:

| Jenis | Catatan |
|---|---|
| PDF | termasuk hasil pemindaian — otomatis lewat OCR bila tidak ada lapisan teks |
| PPTX, DOCX, XLSX | sebelumnya gagal diproses |
| Audio & video | ditranskripsi otomatis sehingga rekaman kuliah dapat ditanyakan |

**Yang perlu diperhatikan tim BE:** agar materi muncul di navigasi terpandu, kirimkan `course_id`, `course_name`, dan `week` pada form upload. Bila tidak dikirim, ketiganya diturunkan dari `content_id` dengan pola `<matkul>-minggu-<n>` — misalnya `sbd-minggu-2` menjadi mata kuliah `sbd`, minggu 2. Bila polanya tidak cocok, materi tetap dapat ditanyakan tetapi **tidak muncul di navigasi**.

---

## 5. Autentikasi

`RAGACADEMIC_API_KEY` **sedang dikosongkan** di lingkungan pengembangan, sehingga API dapat diakses tanpa header `X-API-Key`. Ini disengaja untuk mempermudah pengujian.

Sebelum produksi:

1. Isi `RAGACADEMIC_API_KEY` di `.env`.
2. Kirim header `X-API-Key` pada setiap permintaan.

Server kini **menolak start** bila `APP_ENV=production` sementara key kosong, sehingga konfigurasi pengembangan tidak mungkin terbawa diam-diam.

---

## 6. Perilaku yang berubah tanpa perubahan kontrak

Tidak memerlukan penyesuaian di sisi klien, tetapi memengaruhi hasil:

- **Model default** kini Qwen 3.7 Flash. Waktu jawab satu pertanyaan turun dari ±140 detik menjadi ±86 detik.
- **Model dimuat saat server nyala**, bukan saat permintaan pertama. Waktu nyala bertambah ±17 detik, tetapi permintaan pertama tidak lagi melonjak beberapa menit.
- **Batas token** pada tahap internal dinaikkan agar keluaran tidak terpotong pada model *reasoning*.
- **Log** kini mencantumkan pemakaian token dan biaya per panggilan LLM.

---

## 7. Yang belum tersedia

Agar tim BE tidak menunggu hal yang belum ada:

- **Ketepatan materi belum terjamin.** Pada sebagian pertanyaan, sistem masih mengambil materi yang salah. Contoh terukur 5 Agustus: pertanyaan *"jelaskan perbedaan for dan while"* dijawab memakai materi SQL, padahal materi tentang perulangan ada dan terindeks. **Jangan menampilkan jawaban sebagai kebenaran final** — selalu tampilkan daftar `sources` agar mahasiswa dapat memeriksa sendiri, dan pertahankan tombol umpan balik (`POST /chat/feedback`) karena datanya dipakai menelusuri kasus seperti ini.
- Waktu jawab masih ±86 detik; sekitar 50 detik di antaranya untuk *embedding* dan *reranking* pada CPU. Sediakan indikator tunggu yang jelas di sisi antarmuka.
- Belum ada pembatasan kuota pemakaian per pengguna.
- Belum ada endpoint riwayat percakapan yang tersimpan permanen — riwayat hanya hidup di dalam sesi (kedaluwarsa 1 jam). Bila aplikasi web perlu menyimpan riwayat, simpanlah di sisi tim BE.

---

## Pertanyaan

Kontrak lengkap beserta contoh ada di `API_INTEGRATION.md`; skema mesin di `docs/api/openapi.json`. Bila ada bentuk respons yang tidak sesuai dokumen ini, sampaikan beserta contoh permintaannya.
