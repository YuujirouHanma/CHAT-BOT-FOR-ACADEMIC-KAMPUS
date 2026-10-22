"""Buang saran pertanyaan yang materinya tidak akan bisa menjawab.

MENGAPA INI PERLU
-----------------
`generate_followup` menyusun pertanyaan lanjutan dari topik, konsep kunci, dan
jawaban yang baru diberikan — TANPA pernah melihat materinya. Lihat promptnya:
ia menerima `[PERTANYAAN AWAL]`, `[TOPIK]`, `[KONSEP KUNCI]`, dan `[JAWABAN
FINAL]`, dan tidak satu pun potongan dokumen. Jadi model menebak apa yang
PANTAS ditanyakan berikutnya, bukan apa yang MAMPU dijawab — dan tebakannya
sering benar secara akademis tetapi tidak ada di berkas yang diunggah dosen.

Akibatnya bisa diperkirakan, dan memang dilaporkan dua kali di rapat: mahasiswa
mengklik saran yang disodorkan sistem sendiri, lalu menerima "tidak ada bagian
materi yang menjawab pertanyaan itu". Dari sisi mahasiswa, sistemnya membantah
dirinya sendiri dalam dua klik — dan yang rusak bukan hanya satu jawaban,
melainkan kepercayaan pada seluruh saran berikutnya.

Pertanyaan pembuka (`generate_starter_questions`) lebih baik keadaannya karena
dibuat DARI teks materi, tetapi tetap tidak diperiksa: teks materi dirakit dari
seluruh berkas, sementara yang menjawab pertanyaan nanti adalah pencarian dengan
penyaring minggu yang sedang aktif. Keduanya tidak selalu sepakat.

CARA: GERBANG YANG SAMA DENGAN JAWABAN, BUKAN ATURAN BARU
---------------------------------------------------------
Satu-satunya ukuran yang jujur untuk "apakah saran ini bisa dijawab" adalah
menjalankan pencarian yang SAMA dengan yang akan dijalankan kalau mahasiswa
benar-benar mengkliknya, lalu menanyakan pertanyaan yang sama yang ditanyakan
`src/retrieval/relevansi.py` sebelum memanggil model: apakah ada setidaknya satu
potongan yang benar-benar menjawab?

Karena itu modul ini tidak membawa ambang sendiri-sendiri dan tidak memakai
kemiripan kata. Ia memakai `skor_tertinggi()` yang sama, dengan asasnya:
**kita hanya menyarankan apa yang memang akan kita jawab.** Kalau kedua ambang
dibiarkan berbeda jauh, yang muncul justru keadaan yang sedang diperbaiki —
saran lolos, lalu jawabannya ditolak.

PENYARINGNYA HARUS IKUT, TERMASUK MINGGUNYA
-------------------------------------------
Pemeriksaan dijalankan dengan `content_id`, `course_id`, dan `weeks` yang sama
dengan pertanyaan yang melahirkan saran itu. Ini sengaja: kalau mahasiswa sedang
membatasi pencarian ke minggu 1, saran yang hanya terjawab di minggu 3 memang
akan ditolak saat diklik, jadi menyaringnya di sini sudah benar. Memeriksa tanpa
penyaring akan meloloskan saran yang gagal — persis gejala yang dikeluhkan.

BIAYANYA, DAN MENGAPA MASIH MURAH
---------------------------------
Tiap kandidat berarti satu pencarian tambahan. Tiga hal menahan biayanya:

1. Semuanya dijalankan BERSAMAAN (`asyncio.gather`), jadi tambahan waktunya
   sekitar satu pencarian, bukan tiga.
2. `rerank_top_k=1` — yang dibutuhkan hanya skor tertinggi, bukan daftarnya.
3. `retrieval_top_k` dipersempit lewat `settings.suggestion_check_top_k`
   (bawaan 10, bukan 20), karena yang dicari cuma ada-tidaknya satu potongan
   yang cocok.

Embedding dan reranker berjalan lokal, jadi tidak ada satu pun token LLM
tambahan yang terpakai di sini. Untuk pertanyaan pembuka biayanya dibayar
sekali saja: hasilnya disimpan `gen_cache` SETELAH disaring.

KALAU PEMERIKSAANNYA SENDIRI GAGAL, SARANNYA DIPERTAHANKAN
----------------------------------------------------------
Kegagalan infrastruktur (Qdrant tak terjangkau, reranker tumbang) bukan bukti
bahwa sarannya buruk. Pada keadaan itu kandidatnya diloloskan apa adanya dan
dicatat sebagai peringatan. Memilih sebaliknya berarti satu gangguan Qdrant
menghapus seluruh saran di seluruh kampus tanpa jejak yang terlihat pengguna —
kemunduran yang lebih buruk daripada gejala yang sedang diperbaiki.

Sebaliknya, pemeriksaan yang BERHASIL dan menolak semuanya mengembalikan daftar
kosong, bukan daftar asli. Tidak menampilkan saran lebih baik daripada
menampilkan saran yang kita sudah tahu akan ditolak.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from loguru import logger

from src.config import settings
from src.retrieval.relevansi import skor_tertinggi

# Panjang maksimum satu kandidat yang masih diperiksa. Saran dari model
# panjangnya satu kalimat; nilai yang jauh lebih panjang berarti parsernya
# meloloskan paragraf, dan mengirim paragraf ke embedder tidak mengukur apa pun
# yang berguna.
_MAKS_PANJANG_KANDIDAT = 300

# Batas jumlah kandidat yang diperiksa dalam satu panggilan. `generate_followup`
# mengembalikan 3 dan starter 5, jadi ini hanya pagar terhadap parser yang
# mengembalikan daftar panjang - bukan batas yang diharapkan tersentuh.
_MAKS_KANDIDAT = 8


def _layak_diperiksa(teks: str) -> bool:
    """Kandidat yang kosong atau kepanjangan dilewati, bukan diperiksa.

    `retrieve()` melempar ValueError pada query kosong, jadi tanpa penjagaan ini
    satu entri kosong dari parser akan menjadi peringatan di log dan membuat
    seluruh kandidat lain diloloskan tanpa diperiksa.
    """
    return bool(teks.strip()) and len(teks) <= _MAKS_PANJANG_KANDIDAT


async def saring_saran(
    kandidat: Sequence[str],
    *,
    retriever: Any,
    content_id: str | None = None,
    course_id: str | None = None,
    weeks: Sequence[int] | None = None,
    source_filter: str | None = None,
    tenant_id: str,
) -> list[str]:
    """Kembalikan hanya saran yang materinya benar-benar bisa menjawab.

    Urutan aslinya dipertahankan: model menyusun saran dari yang paling dekat
    dengan jawaban tadi ke yang paling jauh, dan urutan itu informasi.

    `settings.suggestion_min_score` bernilai 0 mematikan pemeriksaan ini
    seluruhnya — daftar asli dikembalikan tanpa satu pun pencarian dijalankan.
    Disediakan dengan alasan yang sama seperti `relevance_min_score`:
    pemasangan yang rerankernya belum siap tidak boleh ikut kehilangan sarannya.
    """
    ambang = settings.suggestion_min_score
    if ambang <= 0 or not kandidat:
        return list(kandidat)

    diperiksa = [t for t in list(kandidat)[:_MAKS_KANDIDAT] if _layak_diperiksa(t)]
    dilewati = [
        t for t in list(kandidat)[:_MAKS_KANDIDAT] if not _layak_diperiksa(t)
    ]
    if not diperiksa:
        return list(kandidat)

    hasil = await asyncio.gather(
        *(
            retriever.retrieve(
                query=teks,
                retrieval_top_k=settings.suggestion_check_top_k,
                # Hanya skor tertinggi yang dipakai memutuskan, jadi tidak ada
                # gunanya membayar reranker untuk lima potongan.
                rerank_top_k=1,
                content_id=content_id,
                course_id=course_id,
                weeks=list(weeks) if weeks else None,
                source_filter=source_filter,
                tenant_id=tenant_id,
            )
            for teks in diperiksa
        ),
        return_exceptions=True,
    )

    lolos: list[str] = []
    dibuang: list[tuple[str, float]] = []
    for teks, keluaran in zip(diperiksa, hasil, strict=True):
        if isinstance(keluaran, BaseException):
            # Lihat catatan modul: gangguan infrastruktur tidak dipakai sebagai
            # bukti bahwa sarannya buruk.
            logger.warning(
                "Validasi saran gagal untuk '{}', saran dipertahankan: {}",
                teks[:60], keluaran,
            )
            lolos.append(teks)
            continue

        skor = skor_tertinggi(keluaran)
        if skor >= ambang:
            lolos.append(teks)
        else:
            dibuang.append((teks, skor))

    if dibuang:
        logger.info(
            "Saran dibuang karena tak terjawab materi ({} dari {}): {}",
            len(dibuang), len(diperiksa),
            "; ".join(f"'{t[:40]}' skor={s:.3f}" for t, s in dibuang),
        )

    # Kandidat yang dilewati pemeriksaan (kosong/kepanjangan) tidak ikut
    # diloloskan: yang kosong tidak berguna ditampilkan, dan yang kepanjangan
    # pertanda parser bermasalah, bukan saran.
    if dilewati:
        logger.debug("Kandidat saran dilewati pemeriksaan: {}", len(dilewati))

    return lolos
