"""High-level retrieval orchestrator.

Wires together: Embedder → QdrantStore (hybrid search) → Reranker.

This is the single entry point for the generation layer; it returns
final reranked chunks ready to inject into the LLM prompt.

Pipeline:
    query
      ↓ Embedder.embed_query()
    (dense_vec, sparse_vec)
      ↓ QdrantStore.search()  (hybrid RRF if sparse enabled)
    top_k candidates with payload
      ↓ Reranker.rerank()
    top_n final results sorted by relevance
"""
from __future__ import annotations

from src.config import settings
from src.indexing.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.storage.qdrant_store import QdrantStore
from src.tenancy import require_tenant_id
from src.utils.logger import logger


def _dedupe_identical_text(candidates: list[dict]) -> list[dict]:
    """Buang kandidat yang teksnya identik dengan kandidat berperingkat lebih tinggi.

    Materi yang sama sah terindeks di lebih dari satu cakupan — deck yang dipakai
    ulang di dua minggu, atau di dua mata kuliah (lihat keputusan cakupan di
    src/ingestion/dedup.py). Saat kueri tidak dipersempit ke satu materi, salinan
    itu ikut terambil bersama dan berebut lima slot bukti: terukur rata-rata hanya
    3,6 dari 5 slot berisi potongan yang benar-benar berbeda. Setiap salinan juga
    menambah ±1,96 detik rerank CPU tanpa menambah informasi apa pun.

    Pembanding memakai teks yang spasinya dinormalkan, bukan `chunk_id`: titik
    kembar punya ID berbeda. Urutan tahap pertama dipertahankan, sehingga yang
    bertahan adalah salinan berperingkat tertinggi.
    """
    terlihat: set[str] = set()
    hasil: list[dict] = []
    for c in candidates:
        teks = " ".join(((c.get("payload") or {}).get("text") or "").split())
        if teks and teks in terlihat:
            continue
        if teks:
            terlihat.add(teks)
        hasil.append(c)
    return hasil


class HybridRetriever:
    """End-to-end retrieval: embed → vector search → rerank.

    All three sub-components can be injected for testing.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._store = store or QdrantStore()
        self._reranker = reranker or Reranker()

    async def retrieve(
        self,
        query: str,
        retrieval_top_k: int | None = None,
        rerank_top_k: int | None = None,
        source_filter: str | None = None,
        content_id: str | None = None,
        course_id: str | None = None,
        weeks: list[int] | None = None,
        *,
        tenant_id: str,
    ) -> list[dict]:
        """Run the full retrieval pipeline.

        Args:
            query: User question (boleh berupa query hasil pengayaan).
            retrieval_top_k: Initial vector-search candidates. Defaults to
                settings.retrieval_top_k. More = better recall, slower rerank.
            rerank_top_k: Final results returned to caller. Defaults to
                settings.rerank_top_k.
            source_filter: Optional source filename to restrict to one doc.
            tenant_id: WAJIB. Seluruh penyaring lain di atas bersifat opsional —
                mahasiswa yang bertanya bebas tanpa memilih materi membuat
                semuanya `None`. Dulu keadaan itu berarti pencarian menyapu
                seluruh koleksi; dengan `tenant_id` wajib, cakupan terluas yang
                mungkin terjadi adalah seluruh materi milik kampus itu sendiri.

        Catatan: query yang sama dipakai untuk pencarian vektor DAN reranking.
        Sempat dicoba memisahkannya — reranking memakai pertanyaan asli mahasiswa
        dengan dugaan query panjang mengencerkan sinyal — tetapi pengukuran
        menunjukkan sebaliknya: skor rerank justru turun dari 0,009 ke 0,001.
        Jangan diubah tanpa data pembanding kualitas urutan, bukan sekadar skor.

        Returns:
            List of dicts with keys: chunk_id, score (Qdrant), rerank_score,
            payload (contains text, raw_html, image_base64, metadata).
        """
        tenant_id = require_tenant_id(tenant_id, operation="retrieve")
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Empty query")

        retrieval_top_k = retrieval_top_k or settings.retrieval_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        logger.info(
            "Retrieving for query (len={}): retrieval_top_k={}, rerank_top_k={}, tenant={}",
            len(cleaned_query),
            retrieval_top_k,
            rerank_top_k,
            tenant_id,
        )

        dense, sparse = await self._embedder.embed_query(cleaned_query)

        candidates = await self._store.search(
            dense_vector=dense,
            sparse_vector=sparse,
            top_k=retrieval_top_k,
            source_filter=source_filter,
            content_id=content_id,
            course_id=course_id,
            weeks=weeks,
            tenant_id=tenant_id,
        )
        if not candidates:
            logger.warning("Vector search returned no candidates")
            return []

        # Qdrant bisa jatuh ke dense-only ketika indeks sparse-nya belum siap,
        # dan itu terjadi tanpa galat. Tanpa catatan di sini, sistem dapat
        # berjalan berminggu-minggu tanpa jalur leksikal sama sekali sementara
        # metriknya tampak wajar — mahal justru untuk kueri berbahasa Indonesia,
        # yang paling bergantung pada pencocokan istilah.
        mode = candidates[0].get("retrieval_mode")
        if mode and mode != "hybrid":
            logger.warning(
                "Temu-kembali berjalan mode '{}', bukan hibrida — jalur sparse "
                "tidak ikut menentukan {} kandidat ini",
                mode, len(candidates),
            )

        unik = _dedupe_identical_text(candidates)
        if len(unik) < len(candidates):
            logger.info(
                "Membuang {} kandidat bertek identik sebelum rerank ({} → {})",
                len(candidates) - len(unik), len(candidates), len(unik),
            )
        candidates = unik

        reranked = await self._reranker.rerank(
            query=cleaned_query,
            candidates=candidates,
            top_k=rerank_top_k,
        )
        return reranked