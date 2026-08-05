# RAGAcademic — Arsitektur & Diagram Alir Sistem

Dokumen ini memuat spesifikasi diagram sistem RAGAcademic pada tingkat detail yang
lazim dipakai pada naskah jurnal (IEEE). Setiap gambar disertai (a) sumber Mermaid
yang bisa langsung dirender, (b) daftar notasi, dan (c) parameter riil yang diambil
langsung dari kode — bukan asumsi.

**Basis kode yang didokumentasikan:** `RAGAcademic v0.3.0`
(commit `e1b6dc2`, branch `main`).

**Daftar Gambar**

| Gambar | Judul | Tipe |
|---|---|---|
| **Fig. 0** | **Ikhtisar Alur Sistem dalam Satu Gambar** | **Overview diagram** |
| Fig. 1 | Arsitektur Sistem Berlapis (Layered System Architecture) | Block diagram |
| Fig. 2 | Alur Pipa Indexing Luring (Offline Indexing Pipeline) | Data-flow diagram |
| Fig. 3 | Alur Pipa Kueri Daring Lima-Tahap (Online Query Pipeline) | Data-flow diagram |
| Fig. 4 | Mekanisme Hybrid Retrieval dengan Fusi RRF | Detail diagram |
| Fig. 5 | Diagram Sekuens: Unggah & Indexing Materi | Sequence diagram |
| Fig. 6 | Diagram Sekuens: Tanya-Jawab Multimodal | Sequence diagram |
| Fig. 7 | Mesin Keadaan Navigasi Katalog Terpandu | State machine |
| Fig. 8 | Diagram Deployment (Docker Compose) | Deployment diagram |
| Fig. 9 | Skema Data & Metadata Payload | Entity diagram |
| Fig. 10 | Lingkar Umpan Balik Human-in-the-Loop | Feedback loop |

---

## I. Ringkasan Sistem

RAGAcademic adalah *retrieval-augmented generation* (RAG) multimodal untuk materi
kuliah. Sistem menerima berkas heterogen (dokumen teks, presentasi, lembar kerja,
rekaman video/audio kuliah), menormalisasinya menjadi representasi tekstual yang
dapat ditelusuri, dan menjawab pertanyaan mahasiswa dengan jawaban terberkas
(*grounded*) beserta sitasi sumber.

Empat properti yang membedakannya dan perlu ditonjolkan di naskah:

1. **Ingesti multimodal terpadu.** Tabel dan gambar tidak dibuang; keduanya
   diringkas oleh LLM menjadi teks yang dapat disematkan (*embeddable*), tetapi
   **data mentahnya** (HTML tabel, base64 gambar) tetap disimpan dan diinjeksikan
   kembali ke konteks LLM saat menjawab. Ringkasan dipakai untuk *retrieval*, data
   asli dipakai untuk *grounding*. Ini menghindari kehilangan presisi angka.
2. **Retrieval hibrida satu-model.** BGE-M3 menghasilkan vektor *dense* (semantik,
   1024-d) dan *sparse* (leksikal ala BM25) dalam **satu kali forward pass**,
   lalu difusikan server-side oleh Qdrant memakai *Reciprocal Rank Fusion*.
3. **Dekomposisi kueri pra-retrieval.** Pertanyaan mahasiswa yang pendek/ambigu
   diperkaya lebih dulu menjadi kueri eksplisit sebelum penelusuran.
4. **Katalog terpandu berbasis indeks.** Hierarki mata kuliah → minggu → materi
   diturunkan dari isi indeks vektor itu sendiri, bukan dari basis data terpisah,
   sehingga tidak pernah menampilkan minggu/materi yang tidak punya konten.

---

## Fig. 0 — Ikhtisar Alur Sistem dalam Satu Gambar

Gambar tunggal yang memuat keseluruhan sistem: dari materi kuliah masuk, diolah,
disimpan sebagai vektor, lalu dipakai menjawab pertanyaan mahasiswa. Batas sistem
sengaja berhenti di respons API — aplikasi frontend milik pihak lain **tidak**
digambarkan karena bukan bagian yang dibangun di sini.

Kuncinya ada pada dua alur yang bertemu di satu titik. Alur A berjalan **luring**,
sekali saja per materi. Alur B berjalan **daring**, setiap kali ada pertanyaan.
Keduanya bertemu di basis data vektor Qdrant.

```mermaid
flowchart TB
    subgraph ALUR_A["ALUR A — Indexing (luring, sekali per materi)"]
        IN["Materi kuliah diunggah<br/>PDF · DOCX · PPTX · XLSX · TXT · CSV<br/>rekaman kuliah: video · audio"]
        ING["INGESTION<br/>Unstructured partition() → teks, tabel, gambar<br/>PDF hasil pindai → fallback OCR (ind + eng)<br/>Video/audio → ffmpeg + Whisper → transkrip"]
        ENR["PENGAYAAN MULTIMODAL<br/>Tabel dan gambar diringkas LLM jadi teks<br/>agar bisa ditelusuri.<br/>Data asli (HTML tabel, base64 gambar) TETAP disimpan"]
        CHK["CHUNKING + METADATA + EMBEDDING<br/>SentenceSplitter 512/64<br/>cap course_id · week · source_file<br/>BGE-M3 satu forward pass: dense 1024-d + sparse"]
    end

    DB[("QDRANT — koleksi classroom_docs<br/>vektor dense (Cosine) + sparse (modifier IDF)<br/>payload 12 medan metadata")]

    subgraph ALUR_B["ALUR B — Tanya jawab (daring, tiap pertanyaan)"]
        Q["Pertanyaan mahasiswa<br/>materi dipilih lewat katalog terpandu<br/>mata kuliah → minggu → materi"]
        DEC["TAHAP 1 — Dekomposisi kueri (LLM)<br/>pertanyaan pendek/ambigu → kueri diperkaya"]
        RET["TAHAP 2–3 — Retrieval hibrida + reranking<br/>dense + sparse difusikan RRF → 20 kandidat<br/>cross-encoder menyaring → 5 potongan terbaik"]
        GEN["TAHAP 4–5 — Perakitan konteks + generasi (LLM)<br/>tabel HTML asli dan gambar diinjeksikan ke konteks<br/>tiap sumber diberi label Sumber N<br/>plus 3 pertanyaan lanjutan"]
        ANS["Respons API<br/>jawaban bersitasi + daftar sumber<br/>+ rekomendasi pertanyaan lanjutan"]
    end

    LOG[("Log HITL — berkas JSONL<br/>interaksi · umpan balik · nilai kuis")]
    EXT{{"Layanan model eksternal<br/>OpenAI · Groq · OpenRouter · Gemini<br/>HuggingFace · Ollama · Whisper ASR"}}

    IN --> ING --> ENR --> CHK --> DB
    Q --> DEC --> RET
    DB --> RET
    RET --> GEN --> ANS
    ANS --> LOG

    ING -.-> EXT
    ENR -.-> EXT
    DEC -.-> EXT
    GEN -.-> EXT
```

Tiga hal yang paling perlu dijelaskan saat mempresentasikan gambar ini:

1. **Ringkasan dipakai untuk mencari, data asli dipakai untuk menjawab.** Tabel dan
   gambar diringkas agar bisa ditemukan lewat pencarian semantik, tetapi saat
   menyusun jawaban yang diinjeksikan ke LLM adalah HTML tabel dan gambar aslinya.
   Inilah sebabnya angka pada tabel tetap presisi dan tidak rusak oleh ringkasan.
2. **Rekaman kuliah ikut bisa ditanya.** Video dan audio tidak diperlakukan sebagai
   lampiran mati; keduanya ditranskripsi lalu diindeks seperti dokumen biasa.
3. **Satu model menghasilkan dua representasi.** BGE-M3 mengeluarkan vektor dense
   dan sparse sekaligus dalam satu forward pass, sehingga pencarian makna dan
   pencarian kata kunci berjalan berdampingan tanpa memuat dua model terpisah.

Gambar ini adalah ikhtisar. Rincian tiap kotak ada pada Fig. 1 sampai Fig. 10.
Bila naskah hanya memuat sedikit gambar, **pakai gambar ini sebagai Fig. 1** lalu
geser penomoran gambar berikutnya.

---

## Fig. 1 — Arsitektur Sistem Berlapis

Diagram blok lima lapis. Panah padat = aliran permintaan/data; panah putus-putus =
pemanggilan layanan eksternal (jaringan).

```mermaid
flowchart TB
    subgraph L1["LAPIS 1 — Klien"]
        C1["Aplikasi Frontend (Tim BE / PHP)<br/>Klien uji: cURL · Postman · Swagger UI"]
    end

    subgraph L2["LAPIS 2 — Antarmuka Layanan (FastAPI, port 8000)"]
        MW["Middleware<br/>CORS + verify_api_key (X-API-Key)"]
        RU["Router unggah<br/>/documents/upload<br/>/documents/index-batch"]
        RC["Router chat<br/>/chat/ask · /chat/feedback"]
        RK["Router katalog & telusur<br/>/catalog/* · /browse/*<br/>/models · /health"]
        SS["SessionStore<br/>in-memory, TTL 3600 s"]
    end

    subgraph L3["LAPIS 3 — Orkestrasi: RAGPipeline"]
        P1["index_document()"]
        P2["query()"]
        P3["starter_questions()<br/>quiz() · grade_quiz()"]
    end

    subgraph L4["LAPIS 4 — Modul Pemrosesan"]
        M1["Ingestion<br/>validators · parser (Unstructured)<br/>transcriber (ffmpeg + Whisper)"]
        M2["Indexing<br/>summarizer · chunker<br/>embedder (BGE-M3)"]
        M3["Retrieval<br/>hybrid_retriever<br/>reranker (cross-encoder)"]
        M4["Generation<br/>LLMGenerator (multi-provider)<br/>prompts · parser JSON"]
        M5["Observability<br/>hitl.logger · gen_cache"]
    end

    subgraph L5["LAPIS 5 — Persistensi & Layanan Eksternal"]
        D1[("Qdrant<br/>classroom_docs<br/>dense + sparse")]
        D2[("Filesystem<br/>storage/{content_id}/")]
        D3[("data/hitl_logs/*.jsonl<br/>data/gen_cache/*.json")]
        E1{{"Penyedia LLM<br/>OpenAI · Groq · OpenRouter<br/>Gemini · HuggingFace · Ollama"}}
        E2{{"Whisper ASR<br/>Groq / OpenAI"}}
    end

    C1 --> MW
    MW --> RU
    MW --> RC
    MW --> RK
    RC --> SS

    RU --> P1
    RC --> P2
    RK --> P3

    P1 --> M1
    P1 --> M2
    P2 --> M3
    P2 --> M4
    P2 --> M5
    P3 --> M5

    M1 --> D2
    M1 -.-> E2
    M2 --> D1
    M2 -.-> E1
    M3 --> D1
    M4 -.-> E1
    M5 --> D3
```

**Catatan implementasi untuk naskah**

| Komponen | Berkas | Parameter kunci |
|---|---|---|
| Autentikasi | `src/api/auth.py` | Header `X-API-Key`; nonaktif bila `RAGACADEMIC_API_KEY` kosong |
| Siklus hidup | `src/api/main.py` | `lifespan()` → `RAGPipeline()` + `ensure_collection()` sekali saat *startup* |
| Sesi | `src/api/session.py` | TTL 3600 s, riwayat maks. 10 giliran (20 pesan) |
| Konfigurasi | `src/config.py` | `pydantic-settings`, validasi silang `chunk_overlap < chunk_size` dan `rerank_top_k ≤ retrieval_top_k` |

---

## Fig. 2 — Alur Pipa Indexing Luring

Ini gambar terpenting untuk bagian *Methodology*. Perhatikan cabang media dan
cabang OCR — keduanya adalah kontribusi rekayasa yang layak dinarasikan.

Pipa ini terlalu panjang untuk satu gambar cetak, jadi disajikan sebagai dua panel.
Panel (a) menormalisasi berkas heterogen menjadi `ParsedElement`; panel (b)
mengubahnya menjadi titik vektor di Qdrant. Titik sambung keduanya adalah simpul
**E — "elements kosong?"**.

### Fig. 2(a) — Normalisasi masukan: berkas → ParsedElement

```mermaid
flowchart TB
    A["Berkas masuk<br/>storage/{content_id}/{nama}"] --> B{"is_media(ext)?<br/>video: mp4,mkv,mov,…<br/>audio: mp3,wav,m4a,…"}

    B -- Ya --> T1["validate_file()<br/>ada · bukan direktori · ≠0 B · ≤100 MB"]
    T1 --> T2{"provider ≠ disabled<br/>∧ API key ada<br/>∧ ffmpeg di PATH?"}
    T2 -- Tidak --> T9["Kembalikan daftar kosong<br/>(berkas tersimpan, tidak terindeks)"]
    T2 -- Ya --> T3["ffmpeg: -vn, mono, 16 kHz, 32 kbps<br/>segmentasi 600 s → seg_%03d.mp3"]
    T3 --> T4["Whisper large-v3 per segmen<br/>(bahasa: id) — segmen gagal dilewati"]
    T4 --> T5["Gabung transkrip<br/>→ 1 ParsedElement bertipe TEXT"]
    T5 --> E

    B -- Tidak --> P1["validate_indexable()<br/>ekstensi termasuk: pdf, docx, pptx, txt, md,<br/>csv, xlsx, html, rtf, rst, epub, tsv"]
    P1 --> P2["Unstructured partition()<br/>PDF: strategy='fast' (pdfminer)<br/>Lainnya: strategy='auto'"]
    P2 --> P3{"PDF ∧<br/>Σ|teks| < 20 karakter?"}
    P3 -- Ya --> P4["Fallback OCR<br/>strategy='ocr_only'<br/>languages=[ind, eng]"]
    P4 --> P5{"OCR menghasilkan<br/>lebih banyak teks?"}
    P5 -- Ya --> P6
    P5 -- Tidak --> P6
    P3 -- Tidak --> P6["_assemble_elements()"]

    P6 --> P7["Klasifikasi elemen:<br/>· Table → simpan text_as_html<br/>· Image → simpan image_base64 (tanpa base64 ⇒ dibuang)<br/>· Teks → di-buffer & digabung sampai penanda berikutnya"]
    P7 --> E

    E{"elements kosong?"} -- Ya --> Z0["IndexResult(0,0,0)"]
    E -- Tidak --> GO(["Lanjut ke panel (b)"])
```

### Fig. 2(b) — Pengayaan, pemotongan, penyematan, dan penyimpanan

```mermaid
flowchart TB
    IN(["Dari panel (a): elements tidak kosong"]) --> S1["enrich_elements()<br/>asyncio.Semaphore(5)"]

    S1 --> S2{"element_type"}
    S2 -- TEXT --> S5["Lewati<br/>(content sudah embeddable)"]
    S2 -- TABLE --> S3["summarize_table(raw_html)<br/>ringkas 2–4 kalimat, temp 0.2, 512 tok<br/>tenacity: 3 percobaan, backoff 1–8 s"]
    S2 -- IMAGE --> S4["describe_image(base64)<br/>prompt vision 3–5 kalimat, detail='low'<br/>tenacity: 3 percobaan, backoff 1–8 s"]

    S3 --> S8
    S4 --> S8
    S8["Berhasil → element.summary ← ringkasan<br/>Gagal → elemen lolos TANPA summary,<br/>embeddable_text() jatuh ke .content"]

    S5 --> K1
    S8 --> K1

    K1{"Chunker.chunk()<br/>element_type"}
    K1 -- TEXT --> K3["SentenceSplitter<br/>chunk_size=512, overlap=64<br/>→ n potongan, batas kalimat dijaga"]
    K1 -- "TABLE / IMAGE" --> K4["1 chunk = embeddable_text()<br/>raw_html & image_base64 diteruskan<br/>ringkasan TIDAK dipecah<br/>teks kosong ⇒ chunk dibuang + warning"]

    K3 --> C1
    K4 --> C1
    C1["resolve_course_week()<br/>pola content_id 'course-minggu-N'; juga week / w<br/>nilai eksplisit dari form menang atas hasil parsing<br/>cap course_id / course_name / week pada SETIAP chunk"]

    C1 --> M1["Embedder.embed_chunks() — asyncio.to_thread, batch=12<br/>BGE-M3 satu forward pass → dense_vecs (1024-d)<br/>+ lexical_weights (sparse: token_id → bobot)"]
    M1 --> Q1["QdrantStore.upsert_chunks()<br/>batch 64 titik, wait=True"]
    Q1 --> Q2[("Koleksi classroom_docs<br/>vektor 'dense' (Cosine) + 'sparse' (modifier IDF)<br/>payload: 12 medan metadata")]
    Q2 --> Z1["IndexResult(elements, chunks, points, content_id)"]
```

**Parameter numerik yang harus muncul di tabel naskah**

| Parameter | Nilai | Sumber |
|---|---|---|
| Ukuran chunk / tumpang tindih | 512 / 64 token | `chunk_size`, `chunk_overlap` |
| Dimensi vektor dense | 1024 | `embed_dim` |
| Ukuran batch embedding | 12 | `_DEFAULT_BATCH_SIZE` |
| Ukuran batch upsert | 64 titik | `_UPSERT_BATCH` |
| Konkurensi ringkasan multimodal | 5 | `max_concurrency` |
| Percobaan ulang LLM | 3, backoff eksponensial 1–8 s | `tenacity` |
| Ambang deteksi PDF hasil pindai | 20 karakter | `_MIN_PDF_TEXT_CHARS` |
| Durasi segmen audio | 600 s | `transcription_segment_seconds` |
| Batas ukuran unggahan | 100 MB | `max_upload_size_mb` |

---

## Fig. 3 — Alur Pipa Kueri Daring Lima-Tahap

```mermaid
flowchart TB
    Q0["Pertanyaan mahasiswa q<br/>+ content_id, source_filter, model, level"] --> ST1

    subgraph ST1["TAHAP 1 — Dekomposisi Kueri"]
        A1["LLMGenerator.decompose_query()<br/>temp=0.1, max_tokens=256"]
        A2["Keluaran JSON berisi:<br/>topik_utama, konsep_kunci,<br/>tipe_pertanyaan, query_diperkaya"]
        A3{"JSON valid?"}
        A4["q' ← query_diperkaya"]
        A5["Fallback: q' ← q<br/>(kegagalan tahap 1 tidak menggagalkan permintaan)"]
        A1 --> A2 --> A3
        A3 -- Ya --> A4
        A3 -- Tidak --> A5
    end

    ST1 --> ST2

    subgraph ST2["TAHAP 2 — Retrieval Hibrida"]
        B1["Embedder.embed_query(q')<br/>→ (v_dense, v_sparse)"]
        B2["Bangun filter Qdrant:<br/>must content_id ∧ must source_file"]
        B3["query_points dengan Prefetch ganda<br/>+ FusionQuery(RRF), limit = 20"]
        B4{"Sparse tersedia?"}
        B5["Degradasi: pencarian dense-only<br/>(tangkap KeyError dari IDF rescoring)"]
        B1 --> B2 --> B4
        B4 -- Ya --> B3
        B4 -- Tidak --> B5
    end

    ST2 --> D1{"Kandidat kosong?"}
    D1 -- Ya --> Z1["Jawaban penolakan terkendali:<br/>'Materi yang tersedia tidak mencakup informasi tersebut.'<br/>sources kosong, recommendations kosong<br/>→ tetap dicatat ke HITL"]
    D1 -- Tidak --> ST3

    subgraph ST3["TAHAP 3 — Reranking"]
        C1["bge-reranker-v2-m3 (cross-encoder)<br/>skor pasangan (q, teks_kandidat), normalize=True"]
        C2["Urut menurun, ambil 5 teratas<br/>(recall@20 → precision@5)"]
        C1 --> C2
    end

    ST3 --> ST4

    subgraph ST4["TAHAP 4 — Perakitan Konteks & Generasi"]
        E1["format_retrieval_results()"]
        E2{"element_type tiap chunk"}
        E3["TABLE → injeksikan raw_html<br/>(angka dibaca dari data asli, bukan ringkasan)"]
        E4["IMAGE → teks deskripsi inline<br/>+ blok vision image_url data:mime;base64"]
        E5["TEXT → teks chunk apa adanya"]
        E6["Tiap blok diberi label sitasi [Sumber N]"]
        E7["build_system_prompt(level)<br/>level: sederhana, standar, atau detail"]
        E8["chat.completions.create()<br/>temp=0.2, max_tokens=1024"]
        E1 --> E2
        E2 --> E3 --> E6
        E2 --> E4 --> E6
        E2 --> E5 --> E6
        E6 --> E7 --> E8
    end

    ST4 --> ST5

    subgraph ST5["TAHAP 5 — Pembangkitan Pertanyaan Lanjutan"]
        F1["generate_followup(q, dq, jawaban)<br/>temp=0.5, max_tokens=256<br/>panggilan TERPISAH dari jawaban utama"]
        F2["Wajib 3 pertanyaan dari 3 TIPE berbeda:<br/>definisi, contoh, perbandingan,<br/>sebab-akibat, proses, analisis"]
        F3["Gagal → daftar kosong<br/>(tidak menggagalkan jawaban)"]
        F1 --> F2
        F1 -.-> F3
    end

    ST5 --> L1["log_interaction() → conversation_logs.jsonl<br/>mengembalikan interaction_id"]
    L1 --> R1["QueryResult berisi: answer, sources,<br/>recommendations, interaction_id, decomposition"]
    Z1 --> R1
```

**Perhatikan untuk pembahasan (*Discussion*):** total 3 panggilan LLM per pertanyaan
(dekomposisi → jawaban → lanjutan). Latensi ujung-ke-ujung direkam di
`elapsed_seconds` pada log HITL, sehingga sudah tersedia bahan untuk tabel evaluasi
latensi tanpa instrumentasi tambahan.

---

## Fig. 4 — Mekanisme Hybrid Retrieval dengan Fusi RRF

Gambar khusus untuk bagian metode retrieval. Ini yang paling sering diminta reviewer.

```mermaid
flowchart LR
    Q["Kueri diperkaya q'"] --> BM["BGE-M3<br/>SATU forward pass"]
    BM --> DV["v_dense ∈ ℝ¹⁰²⁴"]
    BM --> SV["v_sparse = {(token_id, bobot)}"]

    DV --> PF1["Prefetch A<br/>using='dense'<br/>metrik: Cosine<br/>limit = 2 × top_k = 40"]
    SV --> PF2["Prefetch B<br/>using='sparse'<br/>modifier: IDF<br/>limit = 2 × top_k = 40"]

    FL["Filter payload<br/>must: content_id<br/>must: source_file"] --> PF1
    FL --> PF2

    PF1 --> RA["Peringkat dense R_d"]
    PF2 --> RB["Peringkat sparse R_s"]

    RA --> RRF["FusionQuery(Fusion.RRF)<br/>dieksekusi di sisi server Qdrant"]
    RB --> RRF

    RRF --> TK["top_k = 20 kandidat<br/>berisi chunk_id, score, payload"]
    TK --> RR["Cross-encoder bge-reranker-v2-m3<br/>skor(q, teks) ternormalisasi 0 sampai 1"]
    RR --> FIN["rerank_top_k = 5<br/>diurutkan menurun"]

    ERR["KeyError('sparse') dari _rescore_idf<br/>saat indeks sparse masih kosong"] -.-> FB["Degradasi anggun:<br/>pencarian dense-only"]
    FB -.-> TK
```

**Formulasi matematis untuk naskah**

Kemiripan dense antara kueri dan chunk:

$$\mathrm{sim}_{d}(q,c) = \frac{\mathbf{v}_q \cdot \mathbf{v}_c}{\lVert \mathbf{v}_q \rVert \lVert \mathbf{v}_c \rVert}$$

Skor sparse dengan modifier IDF pada Qdrant:

$$\mathrm{sim}_{s}(q,c) = \sum_{t \in q \cap c} w_{q,t} \cdot w_{c,t} \cdot \mathrm{idf}(t)$$

Fusi Reciprocal Rank Fusion atas kedua daftar peringkat:

$$\mathrm{RRF}(c) = \sum_{r \in \{R_d, R_s\}} \frac{1}{k + \mathrm{rank}_r(c)}, \quad k = 60 \ \text{(default Qdrant)}$$

Skor akhir yang menentukan urutan konteks LLM:

$$s_{\text{final}}(c) = \sigma\big(f_{\text{CE}}(q, \text{teks}(c))\big), \quad \text{ambil } \arg\max_{5}$$

dengan $f_{\text{CE}}$ adalah cross-encoder dan $\sigma$ normalisasi sigmoid.

---

## Fig. 5 — Diagram Sekuens: Unggah & Indexing Materi

```mermaid
sequenceDiagram
    autonumber
    actor U as Dosen / Tim BE
    participant API as POST /documents/upload
    participant AU as verify_api_key
    participant FS as Filesystem
    participant PL as RAGPipeline
    participant PR as parser / transcriber
    participant EX as Penyedia LLM
    participant SM as summarizer
    participant CH as Chunker
    participant EM as Embedder (BGE-M3)
    participant QD as QdrantStore

    U->>API: multipart: file, content_id, course_id, course_name, week
    API->>AU: X-API-Key
    alt kunci tidak valid
        AU-->>U: 401 Unauthorized
    end
    AU-->>API: lolos

    alt content_id diberikan
        API->>FS: tulis storage/{content_id}/{nama}
    else tanpa content_id
        API->>FS: tulis data/uploads/{uuid}_{nama}
    end
    FS-->>API: target_path

    API->>PL: index_document(path, content_id, course_id, course_name, week)

    alt berkas media (video/audio)
        PL->>PR: parse_media()
        PR->>PR: ffmpeg → segmen mp3 mono 16 kHz / 600 s
        loop tiap segmen
            PR->>EX: audio.transcriptions.create (whisper-large-v3)
            EX-->>PR: teks segmen
        end
        PR-->>PL: [1 elemen TEXT] atau [ ]
    else dokumen
        PL->>PR: parse_document()
        PR->>PR: Unstructured partition()
        opt PDF hasil pindai (< 20 karakter)
            PR->>PR: ulang dengan strategy='ocr_only' (ind+eng)
        end
        PR-->>PL: [ParsedElement] (TEXT / TABLE / IMAGE)
    end

    alt tidak ada elemen
        PL-->>API: IndexResult(0,0,0)
        API-->>U: 201 dengan statistik nol
    end

    PL->>SM: enrich_elements(elements)
    par maks. 5 panggilan serentak
        SM->>EX: ringkas tabel (HTML)
        SM->>EX: deskripsikan gambar (vision)
    end
    EX-->>SM: ringkasan
    Note over SM: Kegagalan per elemen dicatat,<br/>elemen tetap lolos tanpa summary
    SM-->>PL: elemen diperkaya

    PL->>CH: chunk(enriched)
    CH-->>PL: [Chunk] (TEXT dipecah, TABLE/IMAGE utuh)

    PL->>PL: resolve_course_week() → cap course_id/course_name/week
    PL->>EM: embed_chunks(chunks)
    EM->>EM: to_thread → encode(batch=12, dense+sparse)
    EM-->>PL: chunks berisi embedding

    PL->>QD: upsert_chunks() (batch 64, wait=True)
    QD-->>PL: jumlah titik tersimpan
    PL-->>API: IndexResult
    API-->>U: 201 {source_file, elements_parsed, chunks_created, points_stored}

    Note over API,FS: Pada kegagalan indexing, berkas target di-unlink<br/>agar storage tidak menyimpan sisa yang tak terindeks
```

---

## Fig. 6 — Diagram Sekuens: Tanya-Jawab Multimodal

```mermaid
sequenceDiagram
    autonumber
    actor S as Mahasiswa
    participant API as POST /chat/ask
    participant SE as SessionStore
    participant PL as RAGPipeline
    participant GE as LLMGenerator
    participant MR as model_registry
    participant RT as HybridRetriever
    participant EM as Embedder
    participant QD as Qdrant
    participant RK as Reranker
    participant PM as prompts
    participant HL as hitl.logger

    S->>API: {question, content_id, source_filter, session_id, model, level}
    API->>SE: get_or_create(session_id)
    SE-->>API: Session (TTL 3600 s)
    API->>SE: set_context(content_id, source_filter)
    Note over SE: Ganti content_id ⇒ source_filter direset<br/>agar tidak menyaring dokumen milik materi lain

    API->>PL: query(question, content_id, source_filter, session_id, model, level)

    PL->>GE: decompose_query(question, model)
    GE->>MR: get(model) → (provider, model_id)
    MR-->>GE: ModelSpec, atau None sehingga jatuh ke provider default
    GE-->>PL: dq{topik_utama, konsep_kunci, tipe_pertanyaan, query_diperkaya}

    PL->>RT: retrieve(q', content_id, source_filter)
    RT->>EM: embed_query(q')
    EM-->>RT: (dense 1024-d, sparse)
    RT->>QD: query_points(Prefetch dense+sparse, RRF, limit 20)
    QD-->>RT: 20 kandidat + payload
    alt tidak ada kandidat
        RT-->>PL: [ ]
        PL->>HL: log_interaction(jawaban penolakan)
        PL-->>API: QueryResult(answer='Materi ... tidak mencakup ...')
        API-->>S: 200 tanpa sumber
    end
    RT->>RK: rerank(q', kandidat, top_k=5)
    RK-->>RT: 5 teratas + rerank_score
    RT-->>PL: hasil terurut

    PL->>PM: format_retrieval_results()
    PM-->>PL: FormattedContext{text_block, image_payloads}
    Note over PM: Tabel → raw_html asli<br/>Gambar → deskripsi + payload vision<br/>Tiap blok berlabel [Sumber N]

    PL->>GE: generate(question, context, model, level)
    GE->>GE: susun pesan multimodal bila ada gambar
    GE-->>PL: jawaban (dengan sitasi [Sumber N])

    PL->>GE: generate_followup(question, dq, jawaban)
    GE-->>PL: 3 pertanyaan lanjutan bertipe berbeda

    PL->>HL: log_interaction(...) → conversation_logs.jsonl
    HL-->>PL: interaction_id
    PL-->>API: QueryResult
    API->>SE: add_turn('user'), add_turn('assistant')
    API-->>S: {answer, sources[ ], recommendations[ ], session_id, interaction_id}

    opt Umpan balik mahasiswa
        S->>API: POST /chat/feedback {interaction_id, rating, issues, comment}
        API->>HL: log_feedback() → student_feedback_logs.jsonl
        API-->>S: 204 No Content
    end
```

**Peringatan akurasi untuk naskah.** `Session.history` diisi oleh `add_turn()`
tetapi **tidak pernah dibaca** kembali — riwayat percakapan tidak diinjeksikan ke
prompt LLM (lihat `src/api/session.py`; tidak ada pembaca `.history` di seluruh
`src/`). Sesi saat ini hanya berfungsi mempertahankan `content_id` dan
`source_filter` antar permintaan. Jadi jangan mengklaim sistem ini mendukung
*multi-turn conversational memory*; klaim yang benar adalah **persistensi konteks
materi antar permintaan**. Bila memang ingin mengklaim yang pertama, riwayat harus
diteruskan ke `RAGPipeline.query()` dan dimasukkan ke daftar `messages`.

---

## Fig. 7 — Mesin Keadaan Navigasi Katalog Terpandu

Alur ini menjawab masalah *"mahasiswa tidak tahu harus bertanya apa"* — navigasi
dengan klik, bukan mengetik prompt.

```mermaid
stateDiagram-v2
    [*] --> DaftarMataKuliah

    DaftarMataKuliah: GET /catalog/courses
    note right of DaftarMataKuliah
        Scroll payload Qdrant medan course_id/course_name.
        Nama kustom (dikirim saat unggah) menang atas
        nama hasil humanize_course() otomatis.
    end note

    DaftarMataKuliah --> DaftarMinggu: pilih course_id
    DaftarMinggu: GET /catalog/courses/COURSE/weeks
    DaftarMinggu --> DaftarMataKuliah: 404 tidak ada minggu terindeks

    DaftarMinggu --> DaftarMateri: pilih week
    DaftarMateri: GET /catalog/courses/COURSE/weeks/W/materials
    DaftarMateri --> DaftarMinggu: 404 tidak ada materi

    DaftarMateri --> MateriTerpilih: pilih source_file lalu dapat content_id

    state MateriTerpilih {
        [*] --> Cabang
        Cabang --> PertanyaanPembuka: GET starter-questions
        Cabang --> Kuis: GET quiz
        Cabang --> TanyaBebas: POST /chat/ask

        PertanyaanPembuka: 5 pertanyaan pembuka hasil LLM
        Kuis: 5 soal pilihan ganda, 4 opsi
        Penilaian: skor = 100 x benar/total plus pembahasan per soal

        PertanyaanPembuka --> TanyaBebas: klik satu pertanyaan
        Kuis --> Penilaian: POST quiz/submit
        Penilaian --> [*]: dicatat ke quiz_attempts.jsonl
        TanyaBebas --> TanyaBebas: pertanyaan lanjutan dari rekomendasi
    }

    note left of MateriTerpilih
        Cache on-disk: data/gen_cache/KIND_HASH.json
        kunci = SHA-1(kind + content_id + source_file), 16 heksa pertama.
        Penilaian memakai kuis TER-CACHE yang sama
        dengan yang diterima mahasiswa, sehingga kunci jawaban konsisten.
    end note
```

---

## Fig. 8 — Diagram Deployment

```mermaid
flowchart TB
    subgraph HOST["Host / Server"]
        subgraph NET["Jaringan Docker Compose"]
            subgraph SVC1["Kontainer: ragacademic-backend"]
                A1["python:3.12-slim"]
                A2["Paket sistem:<br/>tesseract-ocr + tesseract-ocr-ind<br/>poppler-utils · libmagic1<br/>libgl1 · libglib2.0-0 · ffmpeg"]
                A3["NLTK_DATA=/usr/share/nltk_data<br/>punkt_tab + averaged_perceptron_tagger_eng<br/>(diunduh saat BUILD, bukan runtime)"]
                A4["uvicorn src.api.main:app<br/>0.0.0.0:8000"]
                A5["Bobot model diunduh saat run pertama:<br/>BGE-M3 ≈1 GB · reranker ≈600 MB"]
            end

            subgraph SVC2["Kontainer: classroom-rag-qdrant"]
                B1["qdrant/qdrant:v1.12.4"]
                B2["REST :6333 · gRPC :6334"]
                B3["healthcheck tiap 10 s"]
            end
        end

        subgraph VOL["Volume / Bind Mount"]
            V1[("./storage → /app/storage<br/>materi persisten")]
            V2[("./data → /app/data<br/>log HITL, gen_cache, unggahan sementara")]
            V3[("qdrant_storage (volume bernama)<br/>→ /qdrant/storage")]
        end
    end

    CL["Klien HTTP"] -->|":8000"| A4
    A4 -->|"QDRANT_MODE=server<br/>QDRANT_HOST=qdrant:6333"| B2
    SVC1 --- V1
    SVC1 --- V2
    SVC2 --- V3

    A4 -.->|"HTTPS"| EXT{{"API eksternal:<br/>OpenAI · Groq · OpenRouter<br/>Gemini · HuggingFace"}}

    subgraph ALT["Mode alternatif (pengembangan)"]
        D1["QDRANT_MODE=local<br/>QdrantClient(path='./qdrant_storage')<br/>proses tunggal, tanpa server"]
    end
```

**Catatan penting untuk bagian reproduksibilitas naskah:** mode `local` memakai
klien Qdrant tertanam berbasis berkas (satu proses saja); mode `server` dipakai di
Docker/produksi. `docker-compose.yml` menimpa `QDRANT_MODE=server` secara eksplisit
agar backend tidak pernah tanpa sengaja memakai mode tertanam di dalam kontainer.

---

## Fig. 9 — Skema Data & Metadata Payload

```mermaid
erDiagram
    direction LR
    PARSED_ELEMENT ||--o{ CHUNK : "dipecah menjadi"
    CHUNK ||--|| QDRANT_POINT : "dimaterialisasi sebagai"
    COURSE ||--o{ WEEK : memiliki
    WEEK ||--o{ MATERIAL : memiliki
    MATERIAL ||--o{ CHUNK : "sumber dari"
    INTERACTION ||--o{ FEEDBACK : "menerima"
    MATERIAL ||--o{ QUIZ_ATTEMPT : "diujikan pada"

    PARSED_ELEMENT {
        string element_id PK
        enum element_type "text|table|image"
        string content
        string raw_html "khusus tabel"
        string image_base64 "khusus gambar"
        string summary "hasil LLM, untuk retrieval"
        string source_file
        int page_number
        string content_id
    }

    CHUNK {
        string chunk_id PK
        string text "teks yang disematkan"
        string parent_element_id FK
        enum element_type
        string source_file
        int page_number
        int chunk_index
        string content_id
        string course_id
        string course_name
        int week
        string raw_html
        string image_base64
        float_array dense_embedding "1024-d"
        map sparse_embedding "token_id → bobot"
    }

    QDRANT_POINT {
        uuid id PK "= chunk_id"
        vector dense "bernama 'dense', Cosine"
        sparse_vector sparse "bernama 'sparse', modifier IDF"
        json payload "12 medan metadata"
    }

    INTERACTION {
        string interaction_id PK "hitl_YYYYmmdd_HHMMSS_xxxxxxxx"
        datetime timestamp
        string session_id
        string content_id
        string question
        json decomposition
        string answer
        json sources
        json recommendations
        float elapsed_seconds
    }

    FEEDBACK {
        string feedback_id PK
        string interaction_id FK
        datetime timestamp
        string rating
        json issues
        string comment
    }

    QUIZ_ATTEMPT {
        string attempt_id PK
        datetime timestamp
        string session_id
        string student_id
        string content_id
        string source_file
        int correct
        int total
        float score
    }
```

Perhatikan bahwa tidak ada RDBMS. Hierarki katalog **diturunkan** dari medan payload
`course_id` / `week` / `source_file` di dalam Qdrant melalui operasi `scroll` +
deduplikasi. Ini keputusan desain yang layak dibahas: menghilangkan sinkronisasi
ganda antara basis data metadata dan indeks vektor, dengan konsekuensi biaya
pemindaian meningkat seiring pertumbuhan koleksi.

---

## Fig. 10 — Lingkar Umpan Balik Human-in-the-Loop

```mermaid
flowchart LR
    Q["Interaksi tanya-jawab"] --> L1["log_interaction()<br/>conversation_logs.jsonl"]
    L1 --> ID["interaction_id dikembalikan ke klien"]
    ID --> FB["POST /chat/feedback<br/>rating, issues, comment"]
    FB --> L2["log_feedback()<br/>student_feedback_logs.jsonl"]

    QZ["Pengerjaan kuis"] --> L3["log_quiz_attempt()<br/>quiz_attempts.jsonl"]

    L1 --> AN["Analisis luring oleh dosen/asisten"]
    L2 --> AN
    L3 --> AN

    AN --> O1["Validasi kualitas jawaban<br/>(akurasi, sitasi, halusinasi)"]
    AN --> O2["Identifikasi celah materi<br/>(pertanyaan tanpa kandidat retrieval)"]
    AN --> O3["Dataset kandidat untuk fine-tuning"]
    AN --> O4["Analitik progres belajar dari skor kuis"]

    O2 -.->|"unggah materi baru"| Q
    O1 -.->|"perbaikan prompt / parameter"| Q
```

Lingkar ini **luring dan tidak otomatis** — tidak ada pembelajaran daring. Nyatakan
demikian secara eksplisit di naskah agar tidak dituduh mengklaim *online learning*
yang tidak ada.

---

## II. Panduan Menggambar Ulang di Platform Lain

Blok Mermaid di atas bisa langsung dipakai, tetapi untuk naskah IEEE Anda
kemungkinan besar butuh gambar vektor yang memenuhi ketentuan format. Berikut
panduannya.

### A. Ketentuan gambar IEEE yang wajib dipenuhi

| Aspek | Ketentuan |
|---|---|
| Lebar | 3,5 inci (88 mm) untuk satu kolom; 7,16 inci (181 mm) untuk dua kolom |
| Format | Vektor: PDF atau EPS. Bila raster: ≥600 dpi untuk *line art* |
| Fonta | Times New Roman / Times, 8–10 pt di dalam gambar |
| Warna | Harus tetap terbaca dalam skala abu-abu; bedakan dengan pola/garis, jangan hanya warna |
| Caption | Di **bawah** gambar, format `Fig. 1. Kalimat deskriptif.` (titik setelah nomor) |
| Ketebalan garis | Minimal 0,5 pt agar tidak hilang saat dicetak |

### B. Alur kerja tercepat: Mermaid → SVG → penyuntingan vektor

1. Buka <https://mermaid.live>, tempel salah satu blok di atas.
2. Panel **Config**, atur agar cocok untuk cetak:
   ```json
   {
     "theme": "neutral",
     "fontFamily": "Times New Roman, serif",
     "fontSize": 14,
     "flowchart": { "curve": "linear", "nodeSpacing": 40, "rankSpacing": 55 },
     "themeVariables": {
       "primaryColor": "#ffffff",
       "primaryBorderColor": "#000000",
       "primaryTextColor": "#000000",
       "lineColor": "#000000"
     }
   }
   ```
   Konfigurasi ini menghasilkan gambar hitam-putih bergaris tegas — persis yang
   diinginkan penerbit.
3. **Actions → SVG**, unduh.
4. Buka SVG di Inkscape (gratis) atau Illustrator, rapikan penempatan label, lalu
   **File → Save As → PDF**. Di Inkscape, centang *"Convert text to paths"* agar
   fonta tidak bergeser di komputer penerbit.
5. Untuk LaTeX: `\includegraphics[width=\columnwidth]{fig1.pdf}`.

### C. Alternatif: draw.io (paling terkontrol)

1. Buka <https://app.diagrams.net> → **Extras → Edit Diagram**, atau
   **Arrange → Insert → Advanced → Mermaid** dan tempel blok Mermaid — draw.io akan
   mengubahnya menjadi bentuk yang bisa Anda geser satu per satu.
2. Terapkan gaya seragam. Pilih semua → **Edit Style**:
   ```
   rounded=0;whiteSpace=wrap;html=1;fontFamily=Times New Roman;fontSize=9;
   strokeWidth=1;fillColor=none;strokeColor=#000000;fontColor=#000000;
   ```
3. Konvensi bentuk yang konsisten (dan sebutkan konvensinya di caption):
   - Persegi panjang → modul komputasi
   - Belah ketupat → titik keputusan
   - Silinder → penyimpanan persisten
   - Heksagon / persegi panjang bergaris putus → layanan eksternal
   - Persegi panjang bersudut ganda → subproses yang dirinci di gambar lain
4. **File → Export as → PDF**, centang *Crop* dan *Transparent Background*.

### D. Untuk pengguna LaTeX: TikZ

Kualitas terbaik dan fonta otomatis konsisten dengan badan naskah. Kerangka untuk
Fig. 3 (pipa kueri lima-tahap):

```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

\tikzset{
  mod/.style   = {rectangle, draw, minimum width=2.6cm, minimum height=0.8cm,
                  align=center, font=\footnotesize},
  dec/.style   = {diamond, draw, aspect=2, align=center, font=\scriptsize,
                  inner sep=1pt},
  store/.style = {cylinder, draw, shape aspect=.3, shape border rotate=90,
                  align=center, font=\scriptsize},
  ext/.style   = {rectangle, draw, dashed, align=center, font=\scriptsize},
  fl/.style    = {-{Latex[length=2mm]}, thick},
  stage/.style = {draw, dotted, thick, inner sep=4pt, rounded corners}
}

\begin{figure}[!t]
\centering
\begin{tikzpicture}[node distance=0.75cm]
  \node[mod] (q)  {Pertanyaan $q$};
  \node[mod, below=of q]  (s1) {Tahap 1: Dekomposisi\\$q \rightarrow q'$};
  \node[mod, below=of s1] (s2) {Tahap 2: Retrieval hibrida\\(RRF, top-$k$=20)};
  \node[mod, below=of s2] (s3) {Tahap 3: Reranking\\(cross-encoder, top-5)};
  \node[mod, below=of s3] (s4) {Tahap 4: Perakitan konteks\\+ generasi};
  \node[mod, below=of s4] (s5) {Tahap 5: Pertanyaan lanjutan};
  \node[store, right=1.4cm of s2] (qd) {Qdrant};
  \node[ext, right=1.4cm of s4]   (llm) {LLM};

  \draw[fl] (q)--(s1); \draw[fl] (s1)--(s2); \draw[fl] (s2)--(s3);
  \draw[fl] (s3)--(s4); \draw[fl] (s4)--(s5);
  \draw[fl, <->] (s2)--(qd);
  \draw[fl, <->, dashed] (s4)--(llm);
\end{tikzpicture}
\caption{Alur pipa kueri daring lima-tahap pada RAGAcademic.}
\label{fig:query-pipeline}
\end{figure}
```

### E. Prioritas: gambar mana yang masuk naskah

Naskah IEEE biasanya memuat 4–6 gambar. Rekomendasi untuk *Conference Paper*
8 halaman:

| Prioritas | Gambar | Ditempatkan di bagian |
|---|---|---|
| **Wajib** | Fig. 1 (arsitektur berlapis) | *System Overview* |
| **Wajib** | Fig. 2 (pipa indexing) | *Methodology — Ingestion & Indexing* |
| **Wajib** | Fig. 3 (pipa kueri) | *Methodology — Query Processing* |
| **Sangat dianjurkan** | Fig. 4 (RRF) | *Methodology — Hybrid Retrieval*, sertai persamaan |
| Opsional | Fig. 7 (katalog terpandu) | *User Interaction Design* |
| Opsional | Fig. 9 (skema data) | *Implementation* |
| Lampiran | Fig. 5, 6, 8, 10 | Materi tambahan / repositori |

Fig. 5 dan Fig. 6 (sekuens) sebaiknya **digabung menjadi satu gambar dua panel**
(panel (a) indexing, panel (b) kueri) bila tetap ingin dimasukkan — diagram sekuens
memakan ruang vertikal besar dan sering ditolak reviewer karena informasinya sudah
tercakup di Fig. 2 dan Fig. 3.

### F. Contoh caption siap pakai

> Fig. 1. Arsitektur berlapis RAGAcademic. Panah padat menyatakan aliran
> permintaan/data internal; panah putus-putus menyatakan pemanggilan layanan
> eksternal melalui jaringan. Silinder menyatakan penyimpanan persisten.

> Fig. 2. Pipa indexing luring. Berkas media dialihkan ke jalur transkripsi
> (ffmpeg + Whisper), sedangkan dokumen melalui Unstructured dengan mekanisme
> mundur OCR untuk PDF hasil pindai. Elemen tabel dan gambar diringkas oleh LLM
> untuk keperluan penelusuran, sementara data mentahnya tetap dipertahankan.

> Fig. 3. Pipa kueri daring lima-tahap. Setiap pertanyaan memicu tiga kali
> pemanggilan LLM: dekomposisi kueri, sintesis jawaban, dan pembangkitan
> pertanyaan lanjutan.

> Fig. 4. Mekanisme retrieval hibrida. BGE-M3 menghasilkan representasi dense dan
> sparse dalam satu forward pass; keduanya difusikan dengan Reciprocal Rank Fusion
> di sisi server sebelum diperingkat ulang oleh cross-encoder.

---

## III. Rujukan Silang Kode

| Elemen diagram | Berkas | Fungsi/kelas kunci |
|---|---|---|
| Titik masuk API | [src/api/main.py](../src/api/main.py) | `lifespan()`, `app` |
| Autentikasi | [src/api/auth.py](../src/api/auth.py) | `verify_api_key()` |
| Orkestrator | [src/pipeline.py](../src/pipeline.py) | `RAGPipeline.index_document()`, `.query()` |
| Parser dokumen | [src/ingestion/parser.py](../src/ingestion/parser.py) | `parse_document()`, `_try_ocr()` |
| Transkripsi media | [src/ingestion/transcriber.py](../src/ingestion/transcriber.py) | `parse_media()`, `_extract_audio_segments()` |
| Validasi berkas | [src/ingestion/validators.py](../src/ingestion/validators.py) | `validate_indexable()` |
| Ringkasan multimodal | [src/indexing/summarizer.py](../src/indexing/summarizer.py) | `enrich_elements()` |
| Pemotongan | [src/indexing/chunker.py](../src/indexing/chunker.py) | `Chunker.chunk()` |
| Penyematan | [src/indexing/embedder.py](../src/indexing/embedder.py) | `Embedder.embed_chunks()` |
| Retrieval hibrida | [src/retrieval/hybrid_retriever.py](../src/retrieval/hybrid_retriever.py) | `HybridRetriever.retrieve()` |
| Reranking | [src/retrieval/reranker.py](../src/retrieval/reranker.py) | `Reranker.rerank()` |
| Basis data vektor | [src/storage/qdrant_store.py](../src/storage/qdrant_store.py) | `search()`, `upsert_chunks()` |
| Generasi | [src/generation/llm.py](../src/generation/llm.py) | `LLMGenerator.generate()`, `.decompose_query()` |
| Prompt & konteks | [src/generation/prompts.py](../src/generation/prompts.py) | `format_retrieval_results()` |
| Katalog | [src/catalog.py](../src/catalog.py) | `resolve_course_week()` |
| Registry model | [src/model_registry.py](../src/model_registry.py) | `available()` |
| Log HITL | [src/hitl/logger.py](../src/hitl/logger.py) | `log_interaction()`, `log_feedback()` |
| Cache generasi | [src/gen_cache.py](../src/gen_cache.py) | `load()`, `save()` |
| Konfigurasi | [src/config.py](../src/config.py) | `Settings` |
| Deployment | [docker-compose.yml](../docker-compose.yml), [Dockerfile](../Dockerfile) | — |
