# Catatan Perubahan untuk Tim Back-End

**Untuk:** tim Back-End / Front-End yang memanggil layanan RAGAcademic
**Sejak:** integrasi awal 1 Juli 2026 (`API_INTEGRATION.md` versi pertama)
**Per:** 20 Agustus 2026 — layanan versi 0.4.0 (**multi-tenant**)
**Kontrak terbaru:** `docs/api/openapi.json`
**Rincian keamanan:** `docs/SECURITY.md`

---

## 🔴 0. VERSI 0.4.0 — MULTI-TENANT (perubahan yang memutus kompatibilitas)

Layanan ini sekarang melayani **banyak kampus dalam satu deployment**. Sampai
versi 0.3.0, seluruh data — katalog, hasil pencarian, riwayat percakapan —
berada dalam satu ruang bersama tanpa pemisah. Begitu ada kampus kedua, itu
berarti kebocoran data. Versi 0.4.0 memisahkannya.

### Yang WAJIB dikerjakan tim BE

| # | Tindakan | Akibat bila tidak |
|---|---|---|
| **1** | Ganti `X-API-Key` dengan **kunci per tenant** yang baru | Seluruh permintaan dibalas `401` di produksi |
| **2** | Tangani `429` beserta header `Retry-After` | Permintaan gagal diam-diam saat kuota terlampaui |
| **3** | Jangan kirim field yang tidak ada di kontrak | Dibalas `422` — skema kini menolak field asing |
| **4** | Sesuaikan pembacaan body galat (bentuknya berubah) | Penanganan galat membaca field yang sudah tidak ada |

### 1 — Kunci API per tenant

Bentuk kunci baru:

```
ragk_<16 hex>.<rahasia>
```

Dikirim sama seperti sebelumnya, lewat header `X-API-Key`. Bedanya, **kunci itu
sendiri yang menentukan data siapa yang kalian akses.**

```http
POST /chat/ask
X-API-Key: ragk_3f2a91c07be4d5a8.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Kunci diterbitkan oleh tim AI per kampus, dan bisa berbeda per peruntukan:

- kunci **sistem akademik** — boleh mengunggah & mengindeks materi
- kunci **aplikasi mahasiswa** — hanya boleh bertanya & membaca katalog

Kalau memakai kunci baca-saja untuk mengunggah, balasannya `403`. Minta kunci
yang sesuai, bukan menaikkan hak kunci yang sudah ada.

**Kunci lama (`RAGACADEMIC_API_KEY`) masih berjalan di lingkungan pengembangan**
supaya integrasi kalian tidak putus hari ini. Di produksi kunci itu ditolak, dan
server bahkan menolak menyala kalau kunci itu masih terpasang — karena satu
kunci untuk semua pelanggan tidak dapat membedakan siapa pun.

### 2 — `tenant_id` di body: opsional, dan BUKAN penentu akses

Seluruh body permintaan boleh menyertakan `tenant_id`:

```json
{ "question": "apa itu normalisasi", "tenant_id": "kampus-a" }
```

**Penting — ini sering disalahpahami:** field itu **tidak menentukan** data siapa
yang diakses. Cakupan data selalu diturunkan dari kunci API. Nilai di body hanya
**dicocokkan**, dan kalau berbeda permintaan ditolak `403`.

Gunanya menangkap kunci yang tertukar di sisi kalian sedini mungkin. Kalau body
dipercaya sebagai penentu, siapa pun cukup menulis `tenant_id` milik kampus lain
untuk membaca seluruh datanya — dan tidak ada lapisan lain yang akan menahannya.

Field ini boleh tidak dikirim sama sekali.

### 3 — Skema menolak field asing

Semua body permintaan sekarang memakai `extra="forbid"`. Field yang tidak ada di
kontrak menghasilkan `422`, bukan diabaikan diam-diam. Beberapa field juga kini
dibatasi pola: `content_id`, `course_id`, `session_id`, `interaction_id` tidak
boleh memuat `/`, `\`, spasi, atau `..`.

### 4 — Bentuk balasan galat berubah

Sebelumnya: `{"detail": "..."}`. Sekarang seragam untuk semua galat:

```json
{
  "error": "tidak_ditemukan",
  "detail": "Percakapan tidak ditemukan",
  "request_id": "a1b2c3d4e5f6"
}
```

Kode `error` yang stabil: `permintaan_tidak_valid`, `tidak_terautentikasi`,
`akses_ditolak`, `tidak_ditemukan`, `terlalu_banyak_permintaan`,
`muatan_terlalu_besar`, `galat_internal`.

**Pakai `error`, bukan teks `detail`, untuk logika program** — teks `detail`
sengaja dibuat tidak informatif dan bisa berubah.

`request_id` juga dikirim sebagai header `X-Request-ID` pada setiap balasan.
Simpan di log kalian: kalau ada masalah, satu kode itu cukup bagi kami untuk
menemukan barisnya tanpa perlu kalian mengirim rincian apa pun.

Dua perubahan turunan yang perlu diketahui:

- **Galat 500 tidak lagi memuat pesan aslinya.** Dulu `detail` berisi galat
  Python apa adanya — itu membocorkan path berkas dan versi pustaka kami.
- **Percakapan milik tenant lain dibalas `404`, bukan `403`.** Disengaja:
  membedakan keduanya akan memberi tahu bahwa sebuah id memang ada.

### 5 — Kuota dan pembatasan laju

Setiap tenant punya kuota (bawaan 60 permintaan/menit, 5.000/hari; dapat diatur
per pelanggan). Melampauinya menghasilkan:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 12
```

**Hormati `Retry-After`.** Mencoba ulang seketika hanya memperpanjang penolakan.
Selain kuota tenant, ada juga batas per alamat IP dan per kunci — sehingga satu
integrasi yang salah membuat perulangan tidak menghabiskan jatah kampus lain.

Kunci yang salah berulang kali membuat alamat IP kalian diperlambat secara
progresif. Kalau integrasi tiba-tiba lambat setelah salah pasang kunci, tunggu
sebentar — bukan menambah percobaan.

### 6 — `student_id` kini diperlakukan sebagai data pribadi

Tidak ada perubahan kontrak: kirim seperti biasa, terbaca kembali seperti biasa.
Yang berubah ada di sisi kami — ia disimpan dalam bentuk tersandi, dan
penyaringan riwayat berjalan lewat indeks satu arah.

Konsekuensi bagi kalian: **`student_id` yang sama di dua kampus adalah dua orang
berbeda**, dan memang diperlakukan begitu. Jadi NIM yang kebetulan bertabrakan
antar kampus tidak akan mempertemukan riwayat keduanya.

### 7 — Perubahan lain-lain

- `/health` hanya membalas `{"status": "ok"}`. Field `version` dan `llm` dihapus —
  menyebut versi komponen mempermudah pencarian kerentanan yang sudah diketahui.
- `/docs`, `/redoc`, dan `/openapi.json` **dimatikan di produksi**. Pakai
  `docs/api/openapi.json` di repo sebagai kontrak.
- Batas ukuran unggahan kini ditegakkan per tenant, dan ada kuota penyimpanan.
  Melampauinya menghasilkan `413`.

### Yang TIDAK berubah

Supaya jelas apa yang tidak perlu disentuh:

- Seluruh path endpoint tetap sama.
- Bentuk balasan sukses (`mode`, `step`, `choices`, `context`, `attachments`,
  `sources`) tetap sama.
- Alur guided, kuis di dalam chat, gaya belajar, dan multi-minggu tetap sama.
- `session_id` tetap dipakai seperti sebelumnya — hanya saja kini terikat pada
  tenant, sehingga id milik kampus lain tidak akan pernah termuat.

---

## RINGKAS: yang wajib dikerjakan tim BE

| Prioritas | Tindakan | Alasan |
|---|---|---|
| **1 — WAJIB** | Tangani field `mode` pada respons `POST /chat/ask`, **atau** kirim `"guided": false` | Tanpa ini, pengguna bisa menemui pesan chatbot yang menunggu jawaban pilihan padahal tombolnya tidak dirender |
| **2 — WAJIB sebelum produksi** | Pastikan `RAGACADEMIC_API_KEY` terisi dan kirim header `X-API-Key` | Auth sedang dimatikan untuk pengembangan; di `APP_ENV=production` server menolak start bila key kosong |
| 3 — WAJIB bila memakai guided | Tangani `step: "style"` dan `kind: "style"` | Langkah baru pada alur; klien yang tidak mengenalinya akan menampilkan pertanyaan tanpa tombol |
| 4 — Disarankan | Baca `context.weeks` (daftar), bukan `context.week` | Mahasiswa kini boleh memilih beberapa minggu sekaligus |
| 5 — Disarankan | Kirim `course_id`, `course_name`, `week` saat upload, atau ikuti pola `content_id` = `<matkul>-minggu-<n>` | Menentukan apakah materi muncul di navigasi mata kuliah → minggu → materi |
| 6 — Opsional | Tampilkan `attachments` sebagai tombol unduh | Jawaban gaya "praktik" menghasilkan berkas notebook `.ipynb` |
| 7 — Opsional | Pakai `/conversations` untuk daftar & hapus riwayat | Percakapan kini tersimpan permanen dan dapat dibuka kembali |
| 8 — Opsional | Pakai endpoint `/catalog/*`, `/models`, `/learning-styles`, dan kuis | Fitur yang sudah tersedia |

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
| `style` | `"style": <value>` | `{"question": "Lewat diagram", "style": "visual"}` |
| `quiz` | tidak ada tambahan | `{"question": "kuis"}` |

Field eksplisit itu **opsional** — server juga mengenali maksud dari teksnya. Mengirim keduanya adalah yang paling aman.

### Kalau belum sempat menyesuaikan

Kirim `"guided": false` pada setiap permintaan. Balasan akan selalu `mode: "answer"` seperti sebelumnya, tanpa tanya-balik. Perilaku lama tetap utuh.

### Field baru pada request

| Field | Tipe | Keterangan |
|---|---|---|
| `guided` | bool, default `true` | `false` = matikan tanya-balik |
| `style` | string, opsional | key dari `GET /learning-styles`; mengatur CARA menjawab |
| `weeks` | list[int], opsional | beberapa minggu sekaligus; menang atas `week` |
| `student_id` | string, opsional | pemilik percakapan; wajib bila ingin menyaring riwayat per mahasiswa |
| `course_id` | string, opsional | diisi saat pengguna mengklik pilihan mata kuliah |
| `week` | int 1–52, opsional | diisi saat pengguna mengklik pilihan minggu |
| `model` | string, opsional | key dari `GET /models`; ganti model per permintaan |
| `level` | `sederhana` \| `standar` \| `detail` | kedalaman jawaban |

### Field baru pada response

| Field | Keterangan |
|---|---|
| `mode` | `answer` atau `choices` |
| `step` | `course`, `week`, `material`, `style`, `question`, `answer`, `quiz` |
| `choices` | daftar `{label, value, kind}` untuk dirender sebagai tombol |
| `context` | `{course_id, course_name, weeks, week, style, topic, content_id, source_file}` |
| `attachments` | berkas turunan jawaban (mis. notebook `.ipynb`); biasanya kosong |

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

## 2b. Beberapa minggu sekaligus, dan gaya belajar

Dua penambahan pada 6 Agustus. Keduanya opsional bagi klien, tetapi memengaruhi
bentuk balasan sehingga perlu diketahui.

### Beberapa minggu sekaligus

Mahasiswa yang menyiapkan ujian sering perlu membaca beberapa minggu bersamaan.
Kini dikenali dari teks bebas — *"minggu 3 dan 4"*, *"minggu 2-4"*, *"minggu 2, 3, 4"* —
maupun dikirim eksplisit.

| Bagian | Perubahan |
|---|---|
| Request | `weeks: [3, 4]` (baru). `week: 3` **tetap didukung** dan otomatis menjadi `[3]` |
| Response `context` | `weeks: [3, 4]` (baru). `week` tetap ada, berisi minggu pertama, agar klien lama tidak rusak |
| Label materi | Saat lebih dari satu minggu dipilih, label berubah menjadi `"Minggu 2 — bab2.pdf"` karena nama berkas saja menjadi ambigu |

`context` juga memuat **`topic`** — satu kalimat berisi topik yang dibahas pada
minggu terpilih, disimpulkan dari isi materi yang terindeks (bukan dari nama
berkas). Contoh nyata untuk minggu 1+2 mata kuliah SBD:

> *"Perulangan while dan for, percabangan if-elif-else, operator logika Python, serta klausa IN, fungsi agregasi, GROUP BY, HAVING, JOIN, dan INSERT SQL."*

### Gaya belajar

Mahasiswa memilih **cara** ia ingin dijelaskan. Pilihan ini mengganti system
prompt LLM, sehingga bukan sekadar berganti nada — isi jawaban, keluaran
tambahan, dan pertanyaan template ikut berubah.

`GET /learning-styles` mengembalikan daftarnya, jadi klien tidak perlu menyalin
nama gaya secara manual:

| key | label | keluaran tambahan |
|---|---|---|
| `naratif` (default) | Penjelasan bertahap | — |
| `visual` | Lewat diagram | blok ` ```mermaid ` di dalam `answer` |
| `praktik` | Lewat contoh & kode | blok kode + **lampiran notebook** |
| `ringkas` | Poin-poin ringkas | — |
| `sokratik` | Dituntun bertanya | — |

Pada alur guided, gaya ditanyakan **setelah materi dipilih** lewat `step: "style"`
dengan `choices` ber-`kind: "style"`. Klien mengirim balik `value`-nya sebagai
`style`, atau cukup sebagai `question` biasa. Bisa juga diganti kapan saja lewat
teks bebas: *"pakai diagram"*, *"ringkas aja"*.

Berbeda dari `level` yang sudah ada: **gaya mengatur CARA, level mengatur
KEDALAMAN.** Keduanya dapat dipakai bersamaan.

### Lampiran

Field baru `attachments` pada response. Saat ini hanya terisi untuk gaya
`praktik` yang jawabannya memuat kode:

```json
"attachments": [
  { "kind": "notebook",
    "filename": "branching-and-iteration---mitpdf.ipynb",
    "content": "{ ...isi .ipynb... }",
    "mime": "application/x-ipynb+json" }
]
```

`content` adalah isi berkas apa adanya — klien cukup menawarkannya sebagai unduhan.
Notebook dibentuk **deterministik dari jawaban yang sama**, tanpa panggilan LLM
tambahan, sehingga isinya dijamin sama dengan yang dibaca mahasiswa di layar.

Untuk gaya `visual`, diagram Mermaid ada **di dalam `answer`** sebagai blok
berpagar — bukan lampiran. Klien perlu me-render blok `mermaid` agar tampil
sebagai gambar; bila tidak, blok itu tampil sebagai teks kode dan tetap terbaca.

---

## 2c. Riwayat percakapan: tersimpan, dapat dibuka kembali, dapat dihapus

Sebelumnya percakapan hangus setelah satu jam. Kini setiap percakapan tersimpan
permanen dan dapat dibuka lagi berhari-hari kemudian.

**Tidak ada konsep baru bagi klien.** `session_id` yang sudah dipakai
`POST /chat/ask` sekaligus menjadi identitas percakapan. Membuka kembali cukup
dengan mengirim `session_id` lama — server memuatnya dari penyimpanan bila sudah
tidak ada di memori, lengkap dengan mata kuliah, minggu, materi, dan gaya belajar
yang sedang aktif.

| Endpoint | Guna |
|---|---|
| `GET /conversations?student_id=…&limit=50` | daftar percakapan, terbaru dulu |
| `GET /conversations/{id}` | isi lengkap untuk ditampilkan ulang |
| `DELETE /conversations/{id}` | hapus permanen (`204`, atau `404` bila tidak ada) |

### Kirim `student_id`

Field baru `student_id` pada `POST /chat/ask`. Tanpa itu percakapan tersimpan
tanpa pemilik dan **tidak muncul** saat daftar disaring per mahasiswa. Kirimkan
identitas mahasiswa dari aplikasi kalian agar tiap orang hanya melihat miliknya.

### Bentuk transkrip

`transcript` memuat **semua** yang tampil di layar, termasuk langkah navigasi
beserta tombol-tombolnya, sehingga percakapan dapat digambar ulang persis:

```json
{ "role": "assistant", "content": "Minggu ke berapa?",
  "at": "2026-08-06T11:24:03", "mode": "choices", "step": "week",
  "choices": [ { "label": "Minggu 1", "value": "1", "kind": "week" } ] }
```

Pesan `mode: "answer"` membawa `sources` dan `interaction_id`, sehingga tombol
umpan balik tetap berfungsi pada percakapan lama.

**Catatan:** kuis yang sedang berjalan sengaja **tidak** dipulihkan saat
percakapan dibuka kembali — melanjutkan kuis di tengah soal berhari-hari kemudian
lebih membingungkan daripada memulainya lagi. Riwayat kuisnya tetap terlihat di
transkrip.

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
| `GET /learning-styles` | daftar gaya belajar beserta labelnya |
| `GET /conversations` | daftar riwayat percakapan |
| `GET /conversations/{id}` | isi satu percakapan |
| `DELETE /conversations/{id}` | hapus satu percakapan |

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
- Belum ada pencarian di dalam riwayat percakapan, dan belum ada penomoran halaman pada daftarnya (dibatasi `limit`, maksimal 200).

---

## Pertanyaan

Kontrak lengkap beserta contoh ada di `API_INTEGRATION.md`; skema mesin di `docs/api/openapi.json`. Bila ada bentuk respons yang tidak sesuai dokumen ini, sampaikan beserta contoh permintaannya.
