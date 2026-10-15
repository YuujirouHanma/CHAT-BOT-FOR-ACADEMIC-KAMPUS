"""Putuskan apakah hasil pencarian cukup relevan untuk dijawab.

MENGAPA KEPUTUSAN INI HARUS DI KODE
-----------------------------------
Sebelum modul ini, tidak ada ambang relevansi di mana pun. `rerank_score`
dihitung dan diteruskan sampai ke balasan, tetapi tidak pernah dipakai
memutuskan apa pun. Satu-satunya penolakan terjadi ketika pencarian
mengembalikan NOL hasil — dan dengan penyaring kampus yang selalu aktif, itu
hampir tidak pernah terjadi: yang kembali adalah potongan paling tidak buruk,
serelevan apa pun ia sebenarnya.

Akibatnya keputusan "menjawab atau menolak" diserahkan ke prompt: model disuruh
"katakan dengan jujur bila konteks tidak cukup" DAN disuruh menjawab. Ia menuruti
keduanya — menyanggah di kalimat pertama lalu menjawab panjang sesudahnya. Itu
keluhan yang dilaporkan sebagai "jawaban tidak konsisten", dan ia tidak dapat
diperbaiki dengan memperbaiki prompt: yang salah bukan kalimatnya, melainkan
siapa yang mengambil keputusan.

Dan selama pertanyaan di luar materi tetap "dijawab", sumbernya ikut dilampirkan
— lima berkas dengan skor 0,000 — sehingga mahasiswa meragukan seluruh sitasi,
termasuk yang benar.

ANGKA YANG MENDASARI AMBANGNYA
------------------------------
Diukur pada materi SBD minggu 1-3, skor rerank TERTINGGI per pertanyaan:

    di dalam materi   0,937  0,948  0,960
    di luar materi    0,000  0,001

Celahnya 0,936 — sangat lebar, jadi nilai mana pun di antaranya bekerja.
Bawaannya diambil RENDAH (lihat `settings.relevance_min_score`) supaya
penolakan hanya terjadi ketika pencarian benar-benar kosong dari relevansi;
keluhan lain yang dilaporkan justru "aturannya terlalu ketat".

MEMAKAI SKOR TERTINGGI, BUKAN MENYARING TIAP POTONGAN
-----------------------------------------------------
Skor reranker sangat berekor. Pada pertanyaan yang jelas terjawab materi,
potongan terbaik 0,960 sementara potongan kelima 0,011 — padahal keduanya dari
berkas yang sama dan sama-sama sah dipakai. Menyaring per potongan dengan
ambang ini akan membuang hampir seluruh konteks pertanyaan yang BENAR.

Jadi yang dinilai satu hal saja: apakah ada SETIDAKNYA SATU potongan yang
benar-benar menjawab. Kalau ada, seluruh konteks diteruskan dan model yang
memilih; kalau tidak ada, tidak ada gunanya memanggil model sama sekali.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.config import settings


def skor_tertinggi(hasil: Sequence[dict]) -> float:
    """Skor rerank terbaik di antara hasil pencarian; 0.0 bila tidak ada.

    Hasil tanpa `rerank_score` dihitung 0.0, bukan dilewati: itu terjadi ketika
    reranker dimatikan atau gagal, dan pada keadaan itu kita tidak punya dasar
    untuk menyatakan apa pun relevan.
    """
    if not hasil:
        return 0.0

    return max(float(h.get("rerank_score") or 0.0) for h in hasil)


def cukup_relevan(hasil: Sequence[dict]) -> bool:
    """Apakah ada materi yang benar-benar menjawab pertanyaannya.

    Ambang 0 berarti pemeriksaan ini dimatikan — seluruh hasil dianggap layak.
    Disediakan supaya pemasangan yang rerankernya belum siap tidak ikut menolak
    semua pertanyaan.
    """
    ambang = settings.relevance_min_score
    if ambang <= 0:
        return bool(hasil)

    return skor_tertinggi(hasil) >= ambang
