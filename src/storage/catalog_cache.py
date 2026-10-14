"""Simpanan hasil daftar katalog, supaya satu klik tidak memindai semua materi.

MASALAHNYA
----------
`list_courses`, `list_weeks`, dan `list_materials*` menjawab pertanyaan yang
sangat kecil - "mata kuliah apa saja yang ada", "minggu berapa saja" - tetapi
cara menjawabnya menggulung SELURUH titik milik kampus itu di Qdrant, lalu
mengambil nilai uniknya di Python. Biayanya sebanding dengan besar materi,
padahal jawabannya hanya beberapa baris.

Uji beban memperlihatkan akibatnya: dengan ~2.500 titik, satu `GET
/catalog/courses` memakan ~90 ms, dan throughput-nya datar di ~10 permintaan per
detik di SEMUA tingkat keserentakan - 1 maupun 64. Datar seperti itu berarti
tidak ada yang berjalan berbarengan; menambah pengguna hanya memanjangkan
antrean. Pada keserentakan 64, p50 sudah 5,7 detik untuk membuka satu dropdown.

Dan jalur ini bukan jalur pinggir: chat terpandu memanggilnya di SETIAP langkah
(pilih mata kuliah, lalu minggu, lalu materi), jadi biaya itu dibayar beberapa
kali untuk satu percakapan.

MENGAPA DI-CACHE, BUKAN DIPERCEPAT KUERINYA
-------------------------------------------
Qdrant tidak menyediakan "distinct"; mengumpulkan nilai unik memang berarti
membaca semuanya. Yang bisa diubah adalah SEBERAPA SERING itu dilakukan - dan
jawabannya berubah hanya ketika materi ditambah atau dihapus, yaitu beberapa
kali sehari, bukan beberapa kali sedetik.

MENGAPA PENANDA DI BERKAS, BUKAN TTL
------------------------------------
Produksi berjalan dengan beberapa pekerja uvicorn (UVICORN_WORKERS=4), masing
masing proses terpisah dengan memorinya sendiri. Dengan TTL, dosen yang baru
mengunggah materi akan melihatnya muncul atau tidak muncul tergantung pekerja
mana yang melayani - dan itu terbaca sebagai unggahan gagal.

Penanda ditulis ke satu berkas kecil yang dibagi semua pekerja (direktori data
sudah di-bind-mount di compose). Penulis memperbaruinya, pembaca membandingkan
isinya. Satu pembacaan berkas belasan byte menggantikan pemindaian 90 ms.

Polanya sama dengan `src/storage/tenant_store.py`, yang juga menyegarkan
cache-nya berdasarkan perubahan berkas di disk, bukan berdasarkan waktu.

KALAU BERKASNYA TIDAK DAPAT DITULIS
-----------------------------------
Cache dimatikan untuk kampus itu - setiap pembacaan kembali memindai. Lambat,
tetapi benar. Menyimpan tanpa penanda berarti menyajikan katalog basi tanpa
batas waktu, dan katalog basi membuat materi yang sudah diunggah seolah hilang.
"""
from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from src.config import PROJECT_ROOT
from src.utils.logger import logger

DIR_PENANDA = PROJECT_ROOT / "data" / "catalog"

# Batas jumlah entri per kampus. Kuncinya berasal dari parameter pemanggil
# (mata kuliah, daftar minggu), jadi jumlahnya terbatas pada kombinasi yang
# benar-benar dibuka - tetapi batas ini menutup kemungkinan satu klien
# mengirim ribuan kombinasi minggu dan menggelembungkan memori pekerja.
_MAKS_ENTRI = 256

# Dilindungi kunci karena beberapa permintaan dapat mencapainya bersamaan di
# event loop yang sama; isinya dict biasa, dan mengubah ukurannya saat ada yang
# mengiterasi akan melempar.
_kunci = threading.Lock()
_simpanan: dict[str, dict[tuple, Any]] = {}
_penanda_terlihat: dict[str, str] = {}


def _berkas(tenant_id: str) -> Any:
    return DIR_PENANDA / f"{tenant_id}.stamp"


def _baca_penanda(tenant_id: str) -> str | None:
    """Penanda perubahan kampus ini; None berarti "jangan pakai simpanan".

    Berkas yang belum ada bukan galat: kampus yang belum pernah menulis apa pun
    memang belum punya penanda, dan keadaan itu diwakili string kosong sehingga
    hasilnya tetap dapat disimpan.

    Yang harus dibedakan - dan dulu TIDAK dibedakan - adalah "berkasnya belum
    ada" dari "tempatnya tidak dapat dipakai". Di Windows, jalur yang salah satu
    induknya berupa berkas juga melempar FileNotFoundError, sehingga disk yang
    bermasalah terbaca sebagai kampus baru: penanda tidak pernah dapat ditulis,
    nilainya selamanya "", dan pekerja ini menyajikan katalog basi tanpa batas
    waktu. Karena itu direktorinya diperiksa - dibuat kalau perlu - sebelum
    "belum ada" diterima sebagai jawaban.
    """
    try:
        return _berkas(tenant_id).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Penanda katalog {} tidak terbaca: {}", tenant_id, exc)
        return None

    try:
        DIR_PENANDA.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Direktori penanda katalog tidak dapat dipakai ({}); daftar "
            "katalog dihitung ulang setiap kali.", exc,
        )
        return None
    return ""


def tandai_berubah(tenant_id: str) -> None:
    """Catat bahwa materi kampus ini berubah; batalkan simpanan yang ada.

    Dipanggil dari setiap jalur tulis di `QdrantStore`. Nilainya acak, bukan
    waktu: dua penulisan dalam satu tik jam akan menghasilkan penanda yang sama
    bila memakai waktu, dan pembaca akan menyajikan hasil pertama untuk
    perubahan kedua.
    """
    if not tenant_id:
        return

    with _kunci:
        _simpanan.pop(tenant_id, None)
        _penanda_terlihat.pop(tenant_id, None)

    baru = uuid.uuid4().hex
    try:
        DIR_PENANDA.mkdir(parents=True, exist_ok=True)
        berkas = _berkas(tenant_id)
        # Tulis ke berkas sementara lalu ganti namanya: pembaca di pekerja lain
        # tidak boleh sampai membaca berkas yang baru separuh tertulis.
        sementara = berkas.with_suffix(f".{os.getpid()}.tmp")
        sementara.write_text(baru, encoding="utf-8")
        sementara.replace(berkas)
    except OSError as exc:
        # Gagal menulis penanda membuat pekerja LAIN tidak tahu ada perubahan.
        # Dicatat sebagai galat, bukan peringatan: akibatnya terlihat pengguna
        # sebagai materi yang sudah diunggah tetapi tidak muncul di daftar.
        logger.error(
            "Penanda katalog {} gagal ditulis ({}); daftar di pekerja lain "
            "dapat tertinggal sampai ada penulisan berikutnya.", tenant_id, exc,
        )


def ambil(tenant_id: str, kunci: tuple) -> Any | None:
    """Hasil tersimpan, atau None bila belum ada / sudah kedaluwarsa.

    None juga berarti "jangan pakai cache" ketika penanda tidak dapat dibaca.
    """
    penanda = _baca_penanda(tenant_id)
    if penanda is None:
        return None

    with _kunci:
        if _penanda_terlihat.get(tenant_id) != penanda:
            # Ada pekerja lain yang menulis. Seluruh entri kampus ini dibuang,
            # bukan hanya yang diminta: penulisan apa pun dapat mengubah daftar
            # mata kuliah, minggu, maupun materi sekaligus.
            _simpanan.pop(tenant_id, None)
            _penanda_terlihat[tenant_id] = penanda
            return None
        return _simpanan.get(tenant_id, {}).get(kunci)


def simpan(tenant_id: str, kunci: tuple, nilai: Any) -> None:
    """Simpan hasil untuk kampus ini pada penanda yang sedang berlaku.

    Penanda dibaca ULANG di sini, bukan dibawa dari `ambil`: antara keduanya
    ada pemindaian yang memakan puluhan milidetik, dan bila ada unggahan selesai
    di tengah itu, hasil yang sedang dipegang sudah usang. Menyimpannya dengan
    penanda yang baru akan membuatnya terlihat segar sampai penulisan berikutnya.
    """
    penanda = _baca_penanda(tenant_id)
    if penanda is None:
        return

    with _kunci:
        if _penanda_terlihat.get(tenant_id) != penanda:
            _simpanan.pop(tenant_id, None)
            _penanda_terlihat[tenant_id] = penanda

        milik = _simpanan.setdefault(tenant_id, {})
        if len(milik) >= _MAKS_ENTRI and kunci not in milik:
            milik.clear()
        milik[kunci] = nilai


def kosongkan() -> None:
    """Buang seluruh simpanan di proses ini. Dipakai tes dan CLI."""
    with _kunci:
        _simpanan.clear()
        _penanda_terlihat.clear()
