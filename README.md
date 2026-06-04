# Classroom RAG — Multimodal Educational Chatbot

RAG system untuk classroom dengan dukungan dokumen multimodal (teks, tabel, gambar).

## Arsitektur

```
Upload doc → Unstructured.io → [text, table HTML, image base64]
                                          ↓
              Summarize (Groq Llama 3.1 untuk teks/tabel, GPT-4o mini untuk gambar)
                                          ↓
                 Chunking (semantic) → Embedding (bge-m3)
                                          ↓
                Qdrant (payload menyimpan raw HTML & base64)
                                          ↓
Query → Hybrid retrieval → BGE reranker → Top-5
                                          ↓
            Inject raw table/image ke LLM multimodal (gpt-4o-mini)
                                          ↓
                              Jawaban dengan sitasi
```

## Stack

| Layer | Tool |
|---|---|
| Ingestion | Unstructured.io |
| Summarization (teks, tabel) | Llama 3.1 via Groq |
| Summarization (gambar) | GPT-4o mini |
| Embedding | bge-m3 (multilingual) |
| Vector store | Qdrant (local mode) |
| Reranker | bge-reranker-v2-m3 |
| Generation | Groq Llama 3.3 70B / GPT-4o mini |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Edit .env: isi OPENAI_API_KEY dan GROQ_API_KEY

uvicorn src.api.main:app --reload --port 8000
```

Swagger UI tersedia di: http://localhost:8000/docs

## Struktur folder

```
src/
├── config.py              # Pydantic settings
├── schemas.py             # ParsedElement, Chunk, shared types
├── ingestion/             # Unstructured.io parsing
├── indexing/              # Summarization, chunking, embedding
├── storage/               # Qdrant client wrapper
├── retrieval/             # Hybrid retrieval + reranking
├── generation/            # LLM generation with citation
└── utils/                 # Logger, helpers
tests/                     # Unit & integration tests
```

## Status pengembangan

- [x] Project structure & config
- [x] Ingestion (Unstructured.io)
- [x] Multimodal summarization
- [x] Chunking & embedding (BGE-M3)
- [x] Qdrant storage (local mode)
- [x] Hybrid retrieval & reranking
- [x] Generation pipeline (Groq / OpenAI)
- [x] FastAPI endpoints
- [ ] Evaluation (RAGAS)
