# RAGAcademic — Panduan Integrasi untuk Tim BE

API ini adalah backend RAG (FastAPI) yang berdiri sendiri. Dokumen ini untuk tim BE lain yang akan **memanggil** API ini dari service mereka (mis. Next.js API route, backend lain, dsb).

## Base URL

```
http://<host>:8000
```

Saat development lokal: `http://127.0.0.1:8000`.

## Eksplorasi interaktif

- Swagger UI: `GET /docs`
- OpenAPI JSON spec (live): `GET /openapi.json`
- Snapshot OpenAPI spec: [`docs/api/openapi.json`](docs/api/openapi.json)
- TypeScript types auto-generated: [`docs/api/ragacademic.types.ts`](docs/api/ragacademic.types.ts)

## Autentikasi

Semua endpoint **kecuali `/health`** butuh header:

```
X-API-Key: <RAGACADEMIC_API_KEY>
```

Minta key aktif ke pemilik project (disimpan di `.env`, tidak boleh dikirim lewat chat/channel publik). Kalau key tidak diset di server (`RAGACADEMIC_API_KEY` kosong), auth otomatis nonaktif — biasanya hanya untuk development lokal.

Request tanpa header atau dengan key salah akan dapat `401 Unauthorized`.

## Konsep `content_id`

`content_id` adalah **identifier folder/konten** yang dipakai RAGAcademic untuk mengorganisir file dan memfilter hasil pencarian. Nilai ini bebas (string), harus konsisten antara saat upload dan saat query:

- Saat upload file: kirim `content_id` yang sama dengan identifier di MySQL tim BE
- Saat tanya jawab: kirim `content_id` yang sama supaya RAG hanya mencari di materi yang relevan
- Kalau tidak diisi, RAG mencari di seluruh materi yang ada

Struktur file di storage: `storage/{content_id}/{filename}`

## Alur Guided (Catalog) — untuk UI menuntun

Untuk tampilan menuntun (mahasiswa pilih **mata kuliah → minggu → materi → pertanyaan template**,
tanpa harus tahu cara prompt AI), tersedia endpoint navigasi `/catalog/*` + pertanyaan template
auto-generate. Lihat panduan lengkap: [`docs/GUIDED_CATALOG.md`](docs/GUIDED_CATALOG.md).

---

## Endpoint

### `GET /health`
Tidak butuh auth. Cek status service + LLM aktif.

```json
{ "status": "ok", "version": "0.3.0", "llm": "openrouter/openai/gpt-4o-mini" }
```

---

### `POST /chat/ask`
Tanya jawab terhadap materi yang sudah diindex.

**Request body:**
```json
{
  "question": "Apa itu Artificial Intelligence?",
  "session_id": null,
  "content_id": "kka-minggu-1",
  "source_filter": null
}
```
`session_id` boleh `null` di request pertama — server akan generate dan mengembalikannya; kirim balik nilai ini di request lanjutan supaya histori sesi tersambung.

**Response 200:**
```json
{
  "answer": "...jawaban dengan sitasi [Sumber 1]...",
  "sources": [
    {
      "index": 1,
      "source_file": "materi6.pdf",
      "page_number": 3,
      "element_type": "text",
      "content_id": "kka-minggu-1",
      "rerank_score": 0.91
    }
  ],
  "recommendations": ["Pertanyaan lanjutan 1?", "Pertanyaan lanjutan 2?"],
  "session_id": "uuid-sesi",
  "interaction_id": "hitl_20260701_..."
}
```
Simpan `interaction_id` kalau mau kirim feedback lewat `/chat/feedback`.

**Contoh fetch (Next.js):**
```ts
const res = await fetch(`${RAGACADEMIC_BASE_URL}/chat/ask`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.RAGACADEMIC_API_KEY!,
  },
  body: JSON.stringify({ question, content_id: contentId, session_id: sessionId }),
});
if (!res.ok) throw new Error(`RAGAcademic error: ${res.status}`);
const data: components["schemas"]["QueryResponse"] = await res.json();
```

---

### `POST /chat/feedback`
Kirim rating mahasiswa atas satu jawaban. Response `204 No Content` (tanpa body) kalau sukses.

```json
{
  "interaction_id": "hitl_20260701_...",
  "rating": "membantu",
  "issues": [],
  "comment": "Jawaban jelas"
}
```
`rating` hanya boleh salah satu dari: `"membantu"`, `"cukup"`, `"tidak_membantu"`.

---

### `POST /documents/upload`
Upload satu dokumen (PDF, DOCX, PPTX, dll) dan langsung diindex. `multipart/form-data`, bukan JSON.

| Field | Tipe | Wajib |
|---|---|---|
| `file` | file | ✅ |
| `content_id` | string | opsional (tapi sangat disarankan diisi) |

**Response 201:**
```json
{
  "source_file": "materi6.pdf",
  "elements_parsed": 42,
  "chunks_created": 18,
  "points_stored": 18,
  "content_id": "kka-minggu-1"
}
```

---

### `POST /documents/index-batch`
Index semua file di `storage/{content_id}/` sekaligus.

```json
{ "content_id": "kka-minggu-1" }
```

---

### `GET /browse/contents`
List semua `content_id` yang ada di storage.

### `GET /browse/contents/{content_id}/files`
List semua file (dokumen/video/audio/gambar) untuk satu `content_id`, termasuk status `indexed`.

---

## Error format

Semua error pakai bentuk standar FastAPI:
```json
{ "detail": "pesan error" }
```

| Status | Arti |
|---|---|
| 400 | Input tidak valid (mis. file kosong) |
| 401 | `X-API-Key` salah/tidak ada |
| 404 | Content/file tidak ditemukan |
| 422 | Body request tidak sesuai schema (mis. `question` kosong) |
| 500 | Error internal (LLM gagal, dll) |

## Catatan operasional

- CORS saat ini `allow_origins=["*"]` (semua domain diizinkan) — untuk production sebaiknya dipersempit ke domain BE/FE yang sah.
- `GENERATION_PROVIDER` bisa berubah-ubah selama development (saat ini: OpenRouter `openai/gpt-4o-mini`) — field `llm` di `/health` selalu menunjukkan provider/model yang aktif saat itu.
- Biaya panggilan ke `/chat/ask` melibatkan 2-3 LLM call internal (decompose query, jawaban, follow-up) — perlu diperhitungkan kalau tim BE membuat rate limit sendiri di sisi mereka.
- Data lama yang diindex dengan skema `course/week` tidak kompatibel dengan skema `content_id` ini — perlu index ulang semua dokumen setelah upgrade.
