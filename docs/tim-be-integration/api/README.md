# Folder `api/` — Integrasi RAGAcademic

Folder ini berisi endpoint PHP yang menghubungkan sistem tim BE ke
**RAGAcademic** (backend AI berbasis Python/FastAPI milik tim RAG).

---

## File-file di folder ini

### `chatbot.php`
Endpoint utama untuk tanya-jawab mahasiswa.

- **Input:** `message` (pertanyaan), `class_id` (ID kelas dari MySQL)
- **Output:** `reply` (jawaban AI), `recommendations` (pertanyaan lanjutan), `sources` (sumber materi)
- **Dipanggil oleh:** Frontend saat mahasiswa mengirim pertanyaan ke chatbot

### `upload.php`
Endpoint untuk dosen/admin mengupload materi kuliah ke RAGAcademic.

- **Input:** `file` (dokumen PDF/DOCX/PPTX/dll), `class_id`
- **Output:** status sukses/gagal + ringkasan hasil indexing
- **Hanya bisa diakses role:** `dosen`, `admin`, `teacher`
- **Dipanggil oleh:** Halaman manajemen materi (setelah dosen upload file)

---

## Konfigurasi WAJIB sebelum deploy

Di **kedua file** (`chatbot.php` dan `upload.php`), ubah dua baris ini:

```php
define('RAG_BASE_URL', 'http://GANTI_DENGAN_IP_SERVER_RAG:8000');
define('RAG_API_KEY',  'GANTI_DENGAN_API_KEY_DARI_TIM_RAG');
```

| Variabel | Nilai | Darimana |
|---|---|---|
| `RAG_BASE_URL` | URL server RAGAcademic yang bisa diakses dari server ini | Tanya tim RAG — saat ini masih tunnel/localhost |
| `RAG_API_KEY` | Secret key untuk autentikasi | Minta ke tim RAG secara pribadi (jangan lewat chat grup) |

> **JANGAN** hardcode API key di file ini kalau repo bersifat publik.
> Sebaiknya simpan di environment variable atau file config yang di-`.gitignore`.

---

## Alur integrasi lengkap

```
Dosen upload materi
  → POST /api/upload.php  (file + class_id)
    → RAGAcademic /documents/upload
      → File disimpan & diindex ke Qdrant
        → Siap dicari

Mahasiswa kirim pertanyaan
  → POST /api/chatbot.php  (message + class_id)
    → RAGAcademic /chat/ask
      → Cari materi relevan di Qdrant
      → Generate jawaban via LLM
      → Kembalikan answer + recommendations + sources
        → Frontend tampilkan ke mahasiswa
```

---

## Pemetaan `class_id` → `content_id`

RAGAcademic mengenal materi lewat `content_id` (string).
Kedua file ini mengkonversi `class_id` (int dari MySQL) ke format:

```
content_id = "class-{class_id}"
// contoh: class_id=42 → content_id="class-42"
```

**Penting:** Format ini harus **konsisten** antara `upload.php` dan `chatbot.php`.
Kalau materi diupload dengan `content_id="class-42"`, maka pertanyaan untuk
kelas 42 juga harus memakai `content_id="class-42"`.

Kalau tim RAG mengganti format, update di kedua file secara bersamaan.

---

## Endpoint RAGAcademic yang dipakai

| Endpoint | File PHP | Fungsi |
|---|---|---|
| `POST /chat/ask` | `chatbot.php` | Tanya-jawab |
| `POST /documents/upload` | `upload.php` | Upload & index materi |
| `GET /health` | — | Cek status server RAG (bisa dipakai untuk monitoring) |
| `GET /browse/contents` | — | List semua content_id yang sudah diindex |
| `GET /browse/contents/{id}/files` | — | List file per content |

Dokumentasi lengkap ada di file `API_INTEGRATION.md` di root repo ini.

---

## Catatan teknis

- **Timeout `chatbot.php`: 60 detik** — RAGAcademic melakukan 3 LLM call
  per pertanyaan (decompose query → jawaban → followup). Jangan kurangi.
- **Timeout `upload.php`: 300 detik** — Indexing dokumen besar bisa lama.
- **Session chat:** `chatbot.php` menyimpan `rag_session_id` di PHP session
  supaya percakapan multi-turn tetap nyambung (AI ingat konteks sebelumnya).
- **Format respons error:** Kalau RAGAcademic tidak bisa dihubungi, kedua
  file mengembalikan HTTP 502 dengan field `error` berisi penjelasan.
