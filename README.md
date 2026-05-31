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
| Vector store | Qdrant |
| Reranker | bge-reranker-v2-m3 |
| Orchestration | LlamaIndex |
| Generation | GPT-4o mini (multimodal) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

docker compose up -d
curl http://localhost:6333/healthz

pytest tests/ -v
```

## Struktur folder

```
src/
├── config.py              # Pydantic settings
├── schemas.py             # ParsedElement, shared types
├── ingestion/             # Unstructured.io parsing
├── indexing/              # Summarization, chunking, embedding
├── storage/               # Qdrant client wrapper
├── retrieval/             # Hybrid retrieval + reranking
├── generation/            # LLM generation with citation
└── utils/                 # Logger, helpers
```

## Status pengembangan

- [x] Project structure & config
- [x] Ingestion (Unstructured.io)
- [ ] Multimodal summarization
- [ ] Chunking & embedding
- [ ] Qdrant storage
- [ ] Hybrid retrieval & reranking
- [ ] Generation pipeline
- [ ] FastAPI endpoints
- [ ] Evaluation (RAGAS)