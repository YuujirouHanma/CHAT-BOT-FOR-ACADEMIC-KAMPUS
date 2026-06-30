# RAGAcademic — AI Chatbot Materi Kuliah

Backend AI berbasis RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan mahasiswa berdasarkan materi kuliah yang diupload dosen. Dibangun dengan Python + FastAPI, diintegrasikan ke sistem tim BE lewat REST API.

---

## Daftar Isi

1. [Prasyarat](#1-prasyarat)
2. [Cara Setup](#2-cara-setup)
3. [Konfigurasi .env](#3-konfigurasi-env)
4. [Menjalankan Server](#4-menjalankan-server)
5. [Verifikasi Server Berjalan](#5-verifikasi-server-berjalan)
6. [Upload Materi Kuliah](#6-upload-materi-kuliah)
7. [Struktur Folder](#7-struktur-folder)
8. [Integrasi dengan Tim BE](#8-integrasi-dengan-tim-be)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prasyarat

Pastikan sudah terinstall di laptop/server:

| Software | Versi | Cara cek |
|---|---|---|
| **Python** | 3.12 | `python --version` |
| **pip** | terbaru | `pip --version` |
| **Git** | bebas | `git --version` |

> **Catatan untuk Windows:** Gunakan **PowerShell** atau **Command Prompt**, bukan Git Bash, untuk menjalankan perintah di bawah.

---

## 2. Cara Setup

Jalankan perintah-perintah ini **satu per satu** di terminal:

```bash
# 1. Clone repo (skip kalau sudah ada)
git clone <URL_REPO_INI>
cd RAGAcademic

# 2. Buat virtual environment Python
python -m venv .venv

# 3. Aktifkan virtual environment
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / Mac

# 4. Install semua dependensi
pip install -r requirements.txt
```

> **Perhatian:** `pip install -r requirements.txt` bisa memakan waktu 5–15 menit tergantung koneksi internet karena mengunduh banyak library AI.

---

## 3. Konfigurasi `.env`

Salin file contoh konfigurasi dan isi sesuai kebutuhan:

```bash
cp .env.example .env     # Linux/Mac
copy .env.example .env   # Windows
```

Buka file `.env` dengan text editor, lalu isi bagian berikut:

```env
# =============================================
# API KEY — pilih salah satu provider di bawah
# =============================================

# OpenRouter (Rekomendasi — bisa pakai banyak model, 1 API key)
OPENROUTER_API_KEY=sk-or-v1-...    ← isi dengan key OpenRouter kamu

# Groq (gratis, tapi tidak ada model vision)
GROQ_API_KEY=gsk_...

# OpenAI (berbayar, kualitas terbaik)
OPENAI_API_KEY=sk-proj-...

# =============================================
# PROVIDER AKTIF — pilih salah satu
# =============================================
GENERATION_PROVIDER=openrouter
GENERATION_MODEL=openai/gpt-4o-mini

# =============================================
# API KEY UNTUK TIM BE (wajib diisi!)
# =============================================
# Key ini yang harus dikasih ke tim BE untuk
# memanggil endpoint RAGAcademic
RAGACADEMIC_API_KEY=isi-dengan-string-acak-panjang
```

> **Cara dapat API key OpenRouter:** Daftar di https://openrouter.ai → Settings → API Keys → Create Key

---

## 4. Menjalankan Server

Pastikan virtual environment sudah aktif (ada tulisan `(.venv)` di terminal), lalu:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Server berjalan di: `http://localhost:8000`

> **Pertama kali jalan:** Server akan otomatis mengunduh model AI (BGE-M3 ~1GB dan BGE-Reranker ~600MB) dari HuggingFace. **Ini hanya terjadi sekali.** Tunggu sampai selesai, bisa 5–20 menit tergantung koneksi.

---

## 5. Verifikasi Server Berjalan

Buka browser dan akses:

- **Health check:** http://localhost:8000/health → harus muncul `{"status":"ok",...}`
- **Swagger UI (dokumentasi API interaktif):** http://localhost:8000/docs

Atau lewat terminal:
```bash
curl http://localhost:8000/health
```

---

## 6. Upload Materi Kuliah

Sebelum chatbot bisa menjawab, materi kuliah harus diupload dan diindex terlebih dahulu.

### Via Swagger UI (termudah)
1. Buka http://localhost:8000/docs
2. Klik endpoint `POST /documents/upload`
3. Klik **Try it out**
4. Isi `content_id` dengan ID unik untuk kelas (contoh: `class-42`)
5. Upload file PDF/DOCX/PPTX
6. Klik **Execute**

### Via terminal (curl)
```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "X-API-Key: ISI_RAGACADEMIC_API_KEY" \
  -F "file=@/path/ke/materi.pdf" \
  -F "content_id=class-42"
```

> **`content_id`** adalah identifier unik per kelas/materi. Harus **konsisten** antara saat upload dan saat chatbot dipanggil. Gunakan format yang sama dengan `class_id` di sistem tim BE (contoh: `"class-42"` untuk kelas dengan ID 42).

---

## 7. Struktur Folder

```
RAGAcademic/
│
├── src/                        # Kode utama aplikasi
│   ├── api/                    # Endpoint FastAPI (REST API)
│   │   ├── routes/
│   │   │   ├── chat.py         # Endpoint /chat/ask dan /chat/feedback
│   │   │   ├── upload.py       # Endpoint /documents/upload
│   │   │   ├── batch.py        # Endpoint /documents/index-batch
│   │   │   └── browse.py       # Endpoint /browse/contents
│   │   ├── auth.py             # Autentikasi X-API-Key
│   │   ├── schemas.py          # Format request/response JSON
│   │   └── session.py          # Manajemen sesi chat
│   │
│   ├── ingestion/              # Parsing dokumen (PDF, DOCX, dll)
│   ├── indexing/               # Embedding + chunking teks
│   ├── retrieval/              # Pencarian semantik (hybrid search)
│   ├── generation/             # Generate jawaban via LLM
│   ├── storage/                # Koneksi ke Qdrant (vector database)
│   ├── hitl/                   # Log interaksi & feedback mahasiswa
│   ├── config.py               # Semua konfigurasi (baca dari .env)
│   └── pipeline.py             # Orkestrasi: index + query end-to-end
│
├── storage/                    # File materi kuliah yang diupload
│   └── {content_id}/           # Folder per kelas/materi
│       └── dokumen.pdf
│
├── qdrant_storage/             # Database vector (otomatis dibuat)
├── data/
│   ├── uploads/                # Upload sementara tanpa content_id
│   └── hitl_logs/              # Log percakapan & feedback (lokal)
│
├── docs/
│   └── api/
│       ├── openapi.json        # Spesifikasi API (untuk Postman/tools)
│       └── ragacademic.types.ts # TypeScript types untuk frontend TS
│
├── tests/                      # Unit & integration tests
├── .env                        # Konfigurasi lokal (JANGAN di-commit)
├── .env.example                # Template konfigurasi
├── requirements.txt            # Daftar dependensi Python
└── API_INTEGRATION.md          # Panduan integrasi untuk tim BE
```

### File/folder yang TIDAK ada di repo (dibuat otomatis atau diabaikan git)

| Path | Keterangan |
|---|---|
| `.env` | Berisi API key — buat sendiri dari `.env.example` |
| `qdrant_storage/` | Database vector — dibuat otomatis saat server pertama jalan |
| `data/hitl_logs/` | Log lokal — tidak di-push |
| `.venv/` | Virtual environment Python — buat sendiri |

---

## 8. Integrasi dengan Tim BE

Lihat file [`API_INTEGRATION.md`](API_INTEGRATION.md) untuk:
- Daftar lengkap endpoint
- Format request/response JSON
- Cara autentikasi (header `X-API-Key`)
- Contoh pemanggilan dari PHP

**Yang perlu dikasih ke tim BE:**
1. URL server ini (setelah running)
2. Nilai `RAGACADEMIC_API_KEY` dari `.env` (kirim via channel aman, bukan repo)
3. File `API_INTEGRATION.md`

---

## 9. Troubleshooting

### `ModuleNotFoundError` saat jalankan server
Virtual environment belum aktif. Jalankan dulu:
```bash
.venv\Scripts\activate   # Windows
```

### Server jalan tapi download model lama
Normal untuk pertama kali. Model BGE-M3 (~1GB) dan BGE-Reranker (~600MB) diunduh otomatis dari HuggingFace. Tunggu sampai selesai.

### Error `401 Unauthorized` dari tim BE
`X-API-Key` yang dipakai tim BE tidak sesuai dengan `RAGACADEMIC_API_KEY` di `.env`. Cek dan samakan.

### Error `insufficient_quota` dari OpenAI/OpenRouter
API key kehabisan kredit. Isi saldo di dashboard provider yang dipakai.

### File diupload tapi chatbot tidak tahu isinya
Pastikan `content_id` saat upload **sama persis** dengan `content_id` yang dikirim saat tanya (`/chat/ask`). Case-sensitive.

### Port 8000 sudah dipakai
Jalankan di port lain:
```bash
uvicorn src.api.main:app --reload --port 8001
```

---

## Stack teknologi

| Layer | Tool |
|---|---|
| Web framework | FastAPI (Python) |
| Parsing dokumen | Unstructured.io |
| Embedding model | BGE-M3 (lokal, CPU) |
| Vector database | Qdrant (embedded/lokal) |
| Reranker | BGE-Reranker-v2-m3 (lokal, CPU) |
| LLM generation | Dikonfigurasi via `.env` (OpenRouter/Groq/OpenAI) |
| Log interaksi | JSONL lokal (`data/hitl_logs/`) |
