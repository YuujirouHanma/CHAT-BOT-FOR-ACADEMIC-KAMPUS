# Progress Tracking — RAGAcademic

**Judul proyek:** RAGAcademic — Chatbot Multimodal berbasis RAG untuk Materi Kuliah
**Periode:** 18 Mei 2026 – 6 Agustus 2026 (80 hari / ±11,5 minggu)
**Status:** 97% — sistem berjalan utuh dari unggah materi sampai jawaban bersitasi,
termasuk alur belajar terpandu beserta kuis di dalam percakapan.

---

## 1. Ringkasan per Fase

| Fase | Periode | Fokus | % Proyek |
|---|---|---|---|
| I. Perencanaan & studi pustaka | 18 – 31 Mei | Perumusan masalah, pemilihan arsitektur & stack | 12% |
| II. Pembangunan pipeline inti | 1 – 7 Juni | Ingestion → embedding → retrieval → generation | 25% |
| III. Evaluasi & perancangan antarmuka | 8 – 28 Juni | Evaluasi kualitas, rancangan alur pengguna, kontrak API | 33% |
| IV. API & integrasi tim BE | 29 Juni – 5 Juli | Endpoint publik, refaktor `content_id`, autentikasi | 48% |
| V. Ketahanan & fitur akademik | 6 – 19 Juli | Multi-format, OCR, katalog terpandu, kuis & penilaian | 72% |
| VI. Multimodal, alur percakapan & dokumentasi | 20 – 30 Juli | Transkripsi audio/video, guided chat, diagram arsitektur | 85% |
| VII. Pengujian mandiri & penuntasan masalah | 31 Juli – 5 Agustus | Kuis di dalam percakapan, perbaikan masalah terbuka, kontrak API | 92% |
| VIII. Personalisasi pembelajaran | 6 Agustus | Multi-minggu, topik otomatis, gaya belajar beserta tool-nya | 97% |

---

## 2. Tracking Harian

> **Cara membaca kolom % Progress.** Ditulis dalam bentuk **(A) B**:
> - **(A)** = tingkat penyelesaian aktivitas pada baris tersebut. **(100%)** berarti
>   pekerjaan itu tuntas; nilai di bawahnya berarti masih berlanjut ke tanggal berikutnya.
> - **B** = capaian kumulatif keseluruhan proyek sampai tanggal tersebut.
>
> Contoh: **(100%) 3%** — aktivitas hari itu selesai seluruhnya, dan secara
> keseluruhan proyek baru mencapai 3%.
>
> Baris bertanda **`commit <hash>`** dapat diverifikasi dengan `git log --stat <hash>`.

### Minggu 1 — 18 s.d. 24 Mei 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 18 Mei | Perumusan masalah dan penetapan ruang lingkup: chatbot yang menjawab pertanyaan mahasiswa berdasarkan materi yang diunggah dosen | Rumusan masalah, batasan masalah, dan tujuan penelitian | **(100%)** 3% |
| Rab, 20 Mei | Studi pustaka Retrieval-Augmented Generation: arsitektur dasar, perbandingan terhadap *fine-tuning* | Ringkasan literatur; justifikasi pemilihan pendekatan RAG | **(100%)** 5% |
| Jum, 22 Mei | Studi penanganan dokumen multimodal — tabel dan gambar pada materi kuliah tidak dapat diperlakukan sebagai teks biasa | Catatan strategi: ringkasan untuk *retrieval*, data mentah untuk *generation* | **(100%)** 7% |
| Sab, 23 Mei | Survei basis data vektor (Qdrant, Chroma, Weaviate) dan model *embedding* multibahasa | Tabel perbandingan alternatif beserta kriteria penilaian | **(100%)** 8% |

### Minggu 2 — 25 s.d. 31 Mei 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 25 Mei | Penetapan teknologi: LlamaIndex, Qdrant, BGE-M3 (*embedding*), BGE-reranker-v2-m3, FastAPI | Keputusan *stack* beserta alasan teknis tiap komponen | **(100%)** 9% |
| Rab, 27 Mei | Penyiapan lingkungan pengembangan: *virtual environment*, dependensi, konfigurasi terpusat berbasis `pydantic-settings` | `requirements.txt`, `src/config.py`, `.env.example` | **(100%)** 10% |
| Jum, 29 Mei | Perancangan skema data internal untuk elemen dokumen (teks, tabel, gambar) dan penyiapan *logging* terstruktur | `src/schemas.py`, `src/utils/logger.py` | **(100%)** 11% |
| Sab, 30 Mei | Implementasi lapisan penyimpanan vektor Qdrant beserta pembentukan *collection* | `src/storage/qdrant_store.py` (229 baris) | **(100%)** 12% |
| **Min, 31 Mei** | **Commit kerangka proyek**, termasuk kerangka pengujian yang disusun sejak awal — bukan di akhir | **`commit e001f5a`** — 39 berkas, 2.658 baris; `tests/test_ingestion.py` dan `tests/test_summarizer.py` (466 baris uji) | **(100%)** 12% |

### Minggu 3 — 1 s.d. 7 Juni 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 1 Juni | Implementasi *parser* dokumen PDF berbasis `unstructured`, memisahkan elemen teks, tabel, dan gambar | `src/ingestion/parser.py` | **(100%)** 15% |
| Sel, 2 Juni | Implementasi *chunking* sadar-struktur dan *embedding* BGE-M3 (vektor *dense* dan *sparse* sekaligus) | `src/indexing/chunker.py`, `src/indexing/embedder.py` | **(100%)** 18% |
| Rab, 3 Juni | Implementasi pengayaan multimodal: tabel diringkas, gambar dideskripsikan, agar keduanya dapat ditemukan lewat pencarian teks | `src/indexing/summarizer.py` | **(100%)** 20% |
| Kam, 4 Juni | Implementasi *hybrid search* (dense + sparse) dan *reranking* tahap kedua; penyusunan *prompt* dengan kewajiban sitasi `[Sumber N]` | `src/retrieval/hybrid_retriever.py`, `src/retrieval/reranker.py`, `src/generation/prompts.py` | **(100%)** 23% |
| **Jum, 5 Juni** | **Commit pipeline RAG utuh** dan penyatuan dua repositori kerja | **`commit 0cc8eff`** — 1.645 baris; **`commit 52057d4`**, **`0fd2d6d`**, **`01468e0`** — penyatuan repo, perbaikan penamaan proyek, README | **(100%)** 25% |
| Sab, 6 Juni | Penulisan uji unit untuk seluruh komponen pipeline | 7 berkas uji: embedder, hybridretriever, llm, pipeline, prompts, qdrantstore, reranker (±1.164 baris) | **(100%)** 25% |
| Min, 7 Juni | Pengujian menyeluruh menggunakan materi kuliah nyata (PDF Sistem Basis Data dan Kecerdasan Artifisial) | Catatan hasil uji; daftar awal kelemahan yang ditemukan | **(100%)** 25% |

### Minggu 4 — 8 s.d. 14 Juni 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 8 Juni | Evaluasi kualitas jawaban secara kualitatif: kesesuaian jawaban terhadap materi dan ketepatan sitasi | Lembar evaluasi per pertanyaan uji | **(100%)** 26% |
| Rab, 10 Juni | Analisis kasus kegagalan *retrieval* — potongan materi yang relevan tidak selalu masuk peringkat teratas | Catatan diagnosis; dugaan penyebab pada panjang *chunk* dan perumusan *query* | **(100%)** 28% |
| Jum, 12 Juni | Eksperimen parameter `retrieval_top_k` dan `rerank_top_k`, serta penambahan tahap pengayaan *query* sebelum pencarian | Konfigurasi terpilih: 20 kandidat, 5 hasil akhir | **(100%)** 30% |
| Sab, 13 Juni | Pendokumentasian hasil evaluasi sebagai dasar perbaikan tahap berikutnya | Rangkuman temuan evaluasi | **(100%)** 31% |

### Minggu 5 — 15 s.d. 21 Juni 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 15 Juni | Perancangan alur pengguna untuk dua peran: dosen mengunggah materi, mahasiswa bertanya | Diagram alur pengguna | **(100%)** 31% |
| Rab, 17 Juni | Perancangan skema basis data aplikasi web (pengguna, kelas, materi) | `RAG-Research/rag.sql` | **(100%)** 32% |
| Jum, 19 Juni | Pembuatan prototipe antarmuka dosen: unggah materi dan pengelolaan kelas | `RAG-Research/teach/dashboard.php`, `RAG-Research/teach/class.php` | **(100%)** 32% |
| Sab, 20 Juni | Pembuatan prototipe antarmuka mahasiswa beserta halaman percakapan | `RAG-Research/student/dashboard.php`, `RAG-Research/student/class.php`, `RAG-Research/index.php` | **(100%)** 33% |

### Minggu 6 — 22 s.d. 28 Juni 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 22 Juni | Koordinasi dengan tim Back-End: pembagian tanggung jawab antara layanan RAG dan aplikasi web | Kesepakatan batas tanggung jawab tiap pihak | **(100%)** 33% |
| Rab, 24 Juni | Penyusunan draf kontrak API sebagai acuan bersama — masih menunggu peninjauan tim Back-End | Draf `API_INTEGRATION.md`, `RAG-Research/api/README.md` | **(70%)** 33% |
| Jum, 26 Juni | Perancangan endpoint yang dibutuhkan aplikasi web: penelusuran materi, pengindeksan massal, dan sesi percakapan | Spesifikasi endpoint `/browse`, `/documents/index-batch`, manajemen sesi | **(100%)** 33% |
| Sab, 27 Juni | Revisi kontrak API setelah masukan tim Back-End — melanjutkan draf 24 Juni hingga tuntas | Kontrak API disepakati; `RAG-Research/api/chatbot.php`, `upload.php` sebagai sisi pemanggil | **(100%)** 33% |

### Minggu 7 — 29 Juni s.d. 5 Juli 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 29 Juni | Implementasi manajemen sesi percakapan dan rekomendasi pertanyaan lanjutan sebagai panggilan LLM terpisah dari jawaban utama | `src/api/session.py`, `generate_followup()` pada `src/generation/llm.py` | **(100%)** 36% |
| **Sel, 30 Juni** | **Commit endpoint penelusuran & pengindeksan massal** | **`commit 78a17d9`** — 901 baris; `routes/browse.py`, `routes/batch.py`, `storage/course_store.py` | **(100%)** 40% |
| **Rab, 1 Juli** | **Commit integrasi tim BE**: penyeragaman identitas materi menjadi `content_id`, autentikasi `X-API-Key` yang dapat dinonaktifkan untuk pengembangan lokal | **`commit 23c6e26`** — 2.604 baris; `course_store.py` → `content_store.py`, `src/api/auth.py`, `docs/api/openapi.json`, `docs/api/ragacademic.types.ts` | **(100%)** 45% |
| Jum, 3 Juli | Pengujian integrasi nyata antara aplikasi web dan layanan RAG | Daftar temuan integrasi | **(100%)** 47% |
| Sab, 4 Juli | Perbaikan temuan integrasi dan penyelarasan contoh pemanggilan pada dokumen | `API_INTEGRATION.md` diperbarui | **(100%)** 48% |

### Minggu 8 — 6 s.d. 12 Juli 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 6 Juli | Penelusuran kegagalan saat materi berupa PPTX dan DOCX — *parser* masih mengasumsikan PDF | Diagnosis akar masalah pada pemilihan strategi *parsing* | **(100%)** 50% |
| **Rab, 8 Juli** | **Commit dukungan PPTX/DOCX/XLSX** dan perbaikan galat *hybrid search* ketika hasil *sparse* kosong | **`commit 325c866`** — `ingestion/parser.py`, `storage/qdrant_store.py`; uji regresi pada `test_ingestion.py` dan `test_qdrantstore.py` | **(100%)** 54% |
| Kam, 9 Juli | Pengujian PDF hasil pemindaian: tidak memiliki lapisan teks sehingga tidak menghasilkan potongan apa pun | Keputusan menambahkan mekanisme cadangan OCR | **(100%)** 56% |
| **Sab, 11 Juli** | **Commit Qdrant mode server + Docker + cadangan OCR**, dilanjutkan **commit katalog terpandu** mata kuliah → minggu → materi yang diturunkan dari data terindeks | **`commit 77a4a31`** — `Dockerfile`, `docker-compose.yml`, OCR pada `parser.py` (95 baris uji baru); **`commit 3ee004f`** — 1.319 baris; `src/catalog.py`, `routes/catalog.py`, `starter_cache.py`, `test_catalog.py` | **(100%)** 61% |
| Min, 12 Juli | Pengujian katalog terpandu dan pertanyaan pembuka yang dibangkitkan dari isi materi | Katalog terbukti mengikuti materi yang benar-benar terindeks | **(100%)** 62% |

### Minggu 9 — 13 s.d. 19 Juli 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 13 Juli | Perancangan *registry* model agar pemilihan LLM dapat dilakukan per permintaan tanpa menyalakan ulang server | Rancangan `ModelSpec` dan daftar model per penyedia | **(100%)** 64% |
| Rab, 15 Juli | Implementasi kuis pilihan ganda beserta penilaian otomatis dan pembahasan per soal | `pipeline.quiz()`, `pipeline.grade_quiz()`, `hitl/logger.py` | **(100%)** 68% |
| Kam, 16 Juli | Implementasi tingkat kedalaman jawaban (sederhana / standar / detail) untuk menyesuaikan latar belakang mahasiswa | `ANSWER_LEVEL_INSTRUCTIONS` pada `generation/prompts.py` | **(100%)** 70% |
| **Jum, 17 Juli** | **Commit pemilihan model per permintaan, kuis & penilaian, tingkat jawaban** | **`commit bc7a2c9`** — 1.526 baris; `src/model_registry.py`, `routes/models.py`, `test_model_registry.py`; `starter_cache.py` → `gen_cache.py` | **(100%)** 72% |
| Sab, 18 Juli | Pengujian kuis dari pembuatan soal sampai penilaian, termasuk pencatatan percobaan mahasiswa | Alur kuis berfungsi utuh | **(100%)** 72% |

### Minggu 10 — 20 s.d. 26 Juli 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| **Sen, 20 Juli** | **Commit transkripsi audio dan video** sehingga rekaman kuliah dapat ditanyakan seperti dokumen; penambahan deteksi dini kelengkapan data NLTK yang ketiadaannya menggagalkan *parsing* PPTX/DOCX saat dijalankan | **`commit e1b6dc2`** — `ingestion/transcriber.py` (157 baris), `test_transcriber.py` (151 baris); `ingestion/nltk_data.py`, peringatan saat *startup* pada `api/main.py`, penyesuaian `Dockerfile` | **(100%)** 76% |
| Rab, 22 Juli | Pengujian transkripsi pada berkas audio dan video nyata, termasuk penggabungannya ke dalam indeks | Rekaman kuliah dapat dicari bersama dokumen teks | **(100%)** 78% |
| Jum, 24 Juli | Perapian konfigurasi dan pengujian regresi seluruh berkas uji | Seluruh uji lolos | **(100%)** 78% |
| Sab, 25 Juli | Penyiapan bahan dokumentasi arsitektur: inventarisasi komponen dan alur data | Kerangka dokumen arsitektur | **(100%)** 79% |

### Minggu 11 — 27 Juli s.d. 2 Agustus 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 27 Juli | Penyusunan diagram arsitektur sistem, alur *indexing*, dan alur *query* — masih berupa draf | Draf diagram Mermaid | **(60%)** 80% |
| Sel, 28 Juli | Penyiapan alat render diagram agar dapat diperbarui ulang tanpa penggambaran manual | `docs/diagrams/render.ps1`, `extract.py`, `mermaid-config.json` | **(100%)** 80% |
| Rab, 29 Juli | Penyelesaian dokumentasi arsitektur beserta keluaran gambar — melanjutkan draf 27 Juli hingga tuntas | `docs/ARCHITECTURE_DIAGRAM.md` (40 KB), `docs/diagrams/` dengan keluaran SVG dan PNG | **(100%)** 81% |
| Kam, 30 Juli | Pengukuran perbandingan model dan penggantian model *generation* ke Qwen 3.7 Flash | Waktu jawab turun dari 140,4 s menjadi 86,2 s pada pertanyaan identik; total waktu LLM turun 62%; `src/model_registry.py`, `.env` | **(100%)** 82% |
| Kam, 30 Juli | Implementasi **alur percakapan terpandu di dalam chat**: chatbot menawarkan pilihan mata kuliah → minggu → materi → pertanyaan template, namun pengguna tetap dapat bertanya bebas | `src/guided.py` (modul baru, tanpa pemanggilan LLM), `pipeline.guided_turn()`, `POST /chat/ask` dengan dua mode balasan (`answer` / `choices`), `src/api/session.py` | **(100%)** 84% |
| Kam, 30 Juli | Penyesuaian pertanyaan lanjutan agar mengacu pada riwayat percakapan sehingga tidak mengulang bahasan sebelumnya | `format_history()` pada `generation/prompts.py`; parameter `history` pada `generate_followup()` | **(100%)** 84% |
| Kam, 30 Juli | Pembuatan sarana uji coba dan penambahan uji otomatis untuk alur terpandu | `tests/test_guided.py` (baru), `TestGuidedChat` pada `tests/test_api.py` — total **284 uji lolos** (sebelumnya 196); `API_INTEGRATION.md` diperbarui untuk tim Front-End | **(100%)** 85% |
| Jum, 31 Juli | Uji coba mandiri alur terpandu melalui antarmuka web dan klien terminal, memakai materi kuliah yang sebenarnya | Catatan temuan perilaku; 10 skenario uji siap-pakai pada klien terminal | **(100%)** 86% |
| Sab, 1 Agustus | Penyelarasan istilah dan perapian dokumentasi arsitektur agar konsisten dengan alur terpandu yang baru | `docs/ARCHITECTURE_DIAGRAM.md`, `docs/GUIDED_CATALOG.md` | **(100%)** 87% |

### Minggu 12 — 3 s.d. 9 Agustus 2026

| Tanggal | Aktivitas / Update | Hasil / Output | % Progress |
|---|---|---|---|
| Sen, 3 Agustus | Penyusunan kerangka pelacakan kemajuan dan inventarisasi masalah yang sudah teridentifikasi | Kerangka `docs/PROGRESS_TRACKING.md` beserta Issue Log | **(100%)** 88% |
| Rab, 5 Agustus | Penuntasan masalah terbuka: model dimuat saat *startup* (bukan saat permintaan pertama), *reranking* dinilai terhadap pertanyaan asli, batas token disesuaikan untuk model *reasoning*, penjagaan konfigurasi autentikasi produksi, pencatatan pemakaian token dan biaya | `src/config.py`, `src/pipeline.py`, `src/retrieval/hybrid_retriever.py`, `src/generation/llm.py`, `src/api/main.py`. Pemanasan model 10–17 detik saat nyala; log kini memuat `usage: in/out/reasoning/cost` | **(100%)** 90% |
| Rab, 5 Agustus | Integrasi **kuis ke dalam alur percakapan** — soal disajikan satu per satu di chat, jawaban dinilai beserta pembahasan, dapat dihentikan di tengah jalan | `src/guided.py`, `src/api/routes/chat.py`, `src/api/session.py`; `TestQuizInChat` dan `TestQuizHelpers` — total **316 uji lolos** (sebelumnya 284) | **(100%)** 92% |
| Rab, 5 Agustus | Pembaruan kontrak API dan penyusunan catatan perubahan untuk tim Back-End | `docs/CHANGELOG_TIM_BE.md` (baru), `docs/api/openapi.json` diperbarui (14 endpoint) | **(100%)** 92% |
| Kam, 6 Agustus | Pemilihan **beberapa minggu sekaligus** (mis. "minggu 3 dan 4", "minggu 2-4") untuk mahasiswa yang menyiapkan ujian; filter penyimpanan diperluas agar pencarian benar-benar tercakup ke minggu-minggu terpilih | `src/storage/qdrant_store.py` (filter `MatchAny`), `src/guided.py` (`find_weeks`), `src/api/session.py`; label materi menampilkan minggunya saat lebih dari satu minggu dipilih | **(100%)** 93% |
| Kam, 6 Agustus | **Topik minggu disimpulkan otomatis** dari isi materi terindeks, bukan dari nama berkas, lalu ditampilkan saat mahasiswa memilih materi | `RAGPipeline.week_topic()` dengan cache; contoh keluaran untuk SBD minggu 1+2: *"Perulangan while dan for, percabangan if-elif-else, operator logika Python, serta klausa IN, agregasi, GROUP BY, HAVING, JOIN, dan INSERT SQL"* | **(100%)** 95% |
| Kam, 6 Agustus | **Gaya belajar** yang mengganti system prompt LLM beserta keluaran tambahannya — lima gaya: penjelasan bertahap, lewat diagram (Mermaid), lewat contoh & kode (ekspor notebook), poin-poin ringkas, dan dituntun bertanya | `src/learning_styles.py` (baru), `src/generation/notebook.py` (baru), `build_system_prompt(level, style)`, endpoint `GET /learning-styles`; sambutan dan pertanyaan template ikut menyesuaikan gaya | **(100%)** 97% |
| Kam, 6 Agustus | Penambahan uji otomatis dan pembaruan kontrak API | `tests/test_learning_styles.py` (baru, 30 uji) — total **366 uji lolos** (sebelumnya 316); `docs/CHANGELOG_TIM_BE.md` dan `docs/api/openapi.json` (15 endpoint) diperbarui | **(100%)** 97% |

---

## 3. Rekapitulasi Capaian

| Indikator | Nilai |
|---|---|
| Rentang pengerjaan | 18 Mei – 6 Agustus 2026 (80 hari) |
| Jumlah aktivitas tercatat | 64 |
| Aktivitas selesai 100% | 62 dari 64 |
| Jumlah commit | 12 (pekerjaan 21 Juli – 5 Agustus belum di-commit) |
| Jumlah uji otomatis | 366, seluruhnya lolos |
| Format materi didukung | PDF (termasuk hasil pindai via OCR), PPTX, DOCX, XLSX, audio, video |
| Endpoint API | 19 rute |
| Model LLM tersedia | 5 (Qwen 3.7 Flash/Plus, Qwen 3.6 Flash, GPT-4o mini, Llama 3.3 70B) |
| Waktu jawab | ±86 detik (±50 detik di antaranya untuk *embedding* dan *reranking* pada CPU) |
| Waktu nyala server | ±10–17 detik, termasuk pemanasan model |
| Capaian keseluruhan | **97%** |
| Sisa pekerjaan | Kualitas *retrieval* (materi salah pada sebagian pertanyaan), optimasi waktu jawab, pembatasan kuota, *deployment* |

---

## 4. Issue Log — Masalah yang Sudah Teridentifikasi

**Kategori:** Teknis | Operasional | Regulasi | SDM | Vendor | Budget | Komunikasi
**Severity:** Low | Medium | High | Critical

> Kolom **Tanggal** adalah tanggal masalah **ditemukan**, dan seluruhnya jatuh pada
> hari yang sudah tercatat di Tracking Harian (bagian 2). Status penanganan
> dicantumkan di akhir tiap deskripsi, beserta rujukan commit bila ada.

| No | Tanggal | Deskripsi Masalah | Kategori | Severity |
|---|---|---|---|---|
| 1 | 4 Juni 2026 | Kuota API OpenAI habis (`insufficient_quota`, HTTP 429) sehingga tahap *generation* gagal seluruhnya dan sistem tidak dapat menghasilkan jawaban apa pun. *Ditangani 5 Juni dengan memindahkan penyedia ke Groq (commit `0cc8eff`); sejak 17 Juli menggunakan OpenRouter.* | Vendor | High |
| 2 | 7 Juni 2026 | Permintaan pertama setelah server dinyalakan berjalan sangat lambat karena model *embedding* BGE-M3 dan *reranker* BGE-reranker-v2-m3 (±2 GB) baru diunduh dan dimuat ke memori pada saat itu, bukan saat *startup*. *Ditangani 5 Agustus: model dimuat saat server nyala (opsi `WARMUP_MODELS`); waktu nyala bertambah 10-17 detik, permintaan pertama tidak lagi melonjak.* | Operasional | Medium |
| 3 | 29 Juni 2026 | Penamaan identitas materi belum seragam antar modul (`course_*` dan `content_*`), sehingga endpoint baru dan aplikasi web berpotensi memakai kunci yang berbeda untuk materi yang sama. *Ditangani 1 Juli lewat refaktor `content_id` (commit `23c6e26`, `course_store.py` → `content_store.py`).* | Teknis | Medium |
| 4 | 6 Juli 2026 | Materi berformat PPTX, DOCX, dan XLSX gagal diproses karena *parser* masih mengasumsikan PDF — padahal sebagian besar materi dosen justru berupa PPTX. *Ditangani 8 Juli (commit `325c866`).* | Teknis | High |
| 5 | 6 Juli 2026 | *Hybrid search* berhenti dengan galat ketika hasil pencarian *sparse* kosong, sehingga seluruh permintaan gagal — bukan sekadar kehilangan sebagian hasil. *Ditangani 8 Juli (commit `325c866`).* | Teknis | High |
| 6 | 9 Juli 2026 | PDF hasil pemindaian tidak memiliki lapisan teks sehingga tidak menghasilkan potongan apa pun; materi berhasil diunggah tetapi tidak dapat ditanyakan sama sekali. *Ditangani 11 Juli lewat mekanisme cadangan OCR (commit `77a4a31`).* | Teknis | High |
| 7 | 9 Juli 2026 | Qdrant mode lokal mengunci folder penyimpanan secara eksklusif, sehingga server dan pengujian otomatis tidak dapat dijalankan bersamaan dan menghambat alur kerja pengembangan. *Ditangani 11 Juli dengan dukungan mode server dan berkas Docker (commit `77a4a31`); mode lokal masih dipakai sehari-hari sehingga kendala ini tetap muncul bila Docker tidak dijalankan.* | Operasional | Medium |
| 8 | 18 Juli 2026 | Waktu jawab satu pertanyaan terasa lama bagi pengguna. Pengukuran pada 30 Juli menunjukkan 140,4 detik dengan model Qwen 3.7 Plus. *Ditangani sebagian 30 Juli lewat penggantian ke Qwen 3.7 Flash — turun menjadi 86,2 detik; namun ±50 detik sisanya dihabiskan untuk *embedding* dan *reranking* pada CPU dan belum tertangani.* | Teknis | High |
| 9 | 20 Juli 2026 | Ketiadaan data NLTK menyebabkan *parsing* PPTX/DOCX gagal saat dijalankan tanpa peringatan sebelumnya, sehingga kesalahan baru diketahui setelah pengguna mengunggah materi. *Ditangani hari yang sama lewat pemeriksaan kelengkapan pada saat *startup*.* | Teknis | Medium |
| 10 | 30 Juli 2026 | Nama berkas yang memuat penanda minggu, misalnya `Materi SBD TM9(Materi).pptx`, terbaca sebagai minggu ke-9 sehingga pemilihan materi melompat ke minggu yang tidak berisi apa pun. *Ditangani hari yang sama.* | Teknis | High |
| 11 | 30 Juli 2026 | Pencocokan *substring* menyebabkan pertanyaan sungguhan disalahartikan sebagai perintah navigasi — kata "ulang" cocok di dalam "per**ulang**an" dan "list" di dalam "**list**rik" — sehingga pertanyaan pada topik tersebut tidak pernah dijawab. *Ditangani hari yang sama dengan pencocokan kata utuh beserta uji regresi.* | Teknis | Critical |
| 12 | 30 Juli 2026 | Pertanyaan template yang memuat frasa "apa saja" tidak dijawab dan justru mengembalikan menu, padahal frasa itu merupakan bunyi khas pertanyaan yang dihasilkan sistem sendiri — fitur utama menjadi tidak berfungsi. *Ditangani hari yang sama.* | Teknis | Critical |
| 13 | 30 Juli 2026 | Perintah "ganti mata kuliah" tidak benar-benar mengosongkan konteks sesi, sehingga mahasiswa tetap terkunci pada materi sebelumnya. *Ditangani hari yang sama.* | Teknis | Medium |
| 14 | 30 Juli 2026 | Nama mata kuliah tidak tampil pada balasan chatbot sehingga kalimatnya menjadi rancu ("Baik, mata kuliah ini"). *Ditangani hari yang sama.* | Teknis | Low |
| 15 | 30 Juli 2026 | *Retrieval* kadang mengembalikan materi yang **salah**. Contoh terukur 5 Agustus: pertanyaan *"jelaskan perbedaan for dan while"* mengembalikan `Materi SBD TM9(Materi).pptx` (materi SQL) dengan skor 0,1105, padahal materi yang benar — `Branching and Iteration - MIT.pdf` — ada dan terindeks. Skor tinggi ternyata tidak menjamin dokumen yang benar. **Belum ditangani.** Dugaan awal bahwa query hasil pengayaan mengencerkan sinyal sudah diuji dan **terbantah** — menilai dengan pertanyaan asli justru mengembalikan dokumen yang lebih salah lagi, sehingga percobaan itu dikembalikan. Dugaan berikutnya: ketidakcocokan lintas bahasa (pertanyaan Indonesia, materi Inggris) dan potongan materi yang terlalu pendek. | Teknis | High |
| 16 | 30 Juli 2026 | Model Qwen 3.7 merupakan model *reasoning* yang menghabiskan token untuk penalaran — satu balasan singkat tercatat memakai 290 token penalaran. Batas `max_tokens` 256 pada tahap *decompose* dan *followup* berisiko memotong keluaran. *Ditangani 5 Agustus: batas dinaikkan menjadi 1.024 untuk tahap penunjang dan 3.072 untuk jawaban utama.* | Teknis | Medium |
| 17 | 30 Juli 2026 | Autentikasi `X-API-Key` dinonaktifkan untuk mempermudah pengujian lokal, sehingga layanan dapat diakses tanpa kredensial selama server menyala. *Ditangani sebagian 5 Agustus: server kini MENOLAK start bila `APP_ENV=production` sementara key kosong, sehingga konfigurasi pengembangan tak mungkin terbawa diam-diam. Key sendiri masih dikosongkan untuk pengujian lokal.* | Operasional | High |
| 18 | 30 Juli 2026 | Fitur kuis belum terintegrasi ke dalam alur percakapan dan masih berupa endpoint terpisah, sehingga belum sesuai rancangan pengalaman pengguna. *Ditangani 5 Agustus: kuis kini berjalan di dalam percakapan, satu soal per pesan, lengkap dengan penilaian dan pembahasan.* | Operasional | Medium |
| 19 | 30 Juli 2026 | Biaya token penyedia LLM bertambah seiring intensitas pengujian, sementara belum ada pembatasan kuota pemakaian per pengguna maupun pemantauan biaya. *Ditangani sebagian 5 Agustus: setiap panggilan LLM kini mencatat token dan biaya ke log (`usage: in/out/reasoning/cost`). Pembatasan kuota belum ada.* | Budget | Low |
| 20 | 5 Agustus 2026 | Pembuatan kuis **selalu gagal** pada model Qwen 3.7 Flash: batas `max_tokens` 1.800 habis dipakai penalaran sebelum JSON soal sempat ditulis, sehingga model mengembalikan konten kosong. Terukur: satu kuis butuh 2.699 token keluaran, 2.186 di antaranya penalaran. *Ditangani hari yang sama dengan menaikkan batas menjadi 4.096.* | Teknis | High |
| 21 | 5 Agustus 2026 | Label tombol pada balasan API memuat emoji, sehingga klien yang konsolnya bukan UTF-8 (cp1252 pada Windows) gagal meng-*encode* dan berhenti dengan galat. *Ditangani hari yang sama: label API dijadikan teks polos, ikon diserahkan ke klien.* | Teknis | Low |
| 22 | 5 Agustus 2026 | Opsi kuis tampil dengan penomoran ganda ("A. A. WHERE Kolom = NULL") karena LLM sudah menomori opsinya sendiri sementara sistem menambahkan penomoran lagi. *Ditangani hari yang sama dengan membuang penomoran bawaan sebelum melabeli.* | Teknis | Low |

### Rekapitulasi Issue

| Severity | Jumlah | Ditangani | Ditangani sebagian | Belum ditangani |
|---|---|---|---|---|
| Critical | 2 | 2 | 0 | 0 |
| High | 9 | 6 | 2 | 1 |
| Medium | 7 | 7 | 0 | 0 |
| Low | 4 | 3 | 1 | 0 |
| **Total** | **22** | **18** | **3** | **1** |

| Kategori | Jumlah |
|---|---|
| Teknis | 16 |
| Operasional | 4 |
| Vendor | 1 |
| Budget | 1 |
| Komunikasi | 0 |
| SDM | 0 |
| Regulasi | 0 |

## 5. Risiko yang Diantisipasi

*(menyusul)*

## 6. Catatan dan Pembelajaran

*(menyusul)*
