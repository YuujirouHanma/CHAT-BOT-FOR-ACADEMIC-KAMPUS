"""Penyaring aksara asing pada keluaran model.

Qwen dilatih terutama pada bahasa Tionghoa. Sesekali — pada materi padat, atau
ketika istilah teknisnya tidak punya padanan yang lazim dalam bahasa Indonesia —
ia menyelipkan aksara Han di tengah kalimat Indonesia:

    "Normalisasi 数据库 bertujuan menghilangkan redundansi"

Mahasiswa tidak bisa membaca potongan itu, dan tidak ada yang bisa mereka
lakukan terhadapnya. Instruksi "jawab dalam bahasa Indonesia" di prompt sistem
sudah ada dan tetap tidak cukup, karena kebocoran ini terjadi pada tingkat
sampling, bukan pemahaman instruksi.

Yang dilakukan di sini hanya membuang, bukan menerjemahkan: menerjemahkan
berarti satu panggilan model lagi untuk setiap jawaban, dan hasilnya tetap
tebakan. Membuang satu-dua istilah menyisakan kalimat Indonesia yang utuh —
contoh di atas menjadi "Normalisasi bertujuan menghilangkan redundansi", yang
benar.

Setiap pembuangan DICATAT. Tanpa catatan, penyaring ini menyembunyikan persis
gejala yang dipakai untuk mengukur seberapa sering model gagal menjaga bahasa,
dan angka itu bahan analisis penelitian.
"""
from __future__ import annotations

import re

from src.utils.logger import logger

# Han, Hiragana, Katakana, Hangul, dan bentuk lebar. Aksara Latin beraksen,
# Yunani, dan Arab TIDAK ikut: ketiganya lumrah dalam materi kuliah
# (matematika memakai Yunani, studi Islam memakai Arab), dan membuangnya akan
# merusak jawaban yang benar.
_CJK = re.compile(
    "["
    "぀-ゟ"   # Hiragana
    "゠-ヿ"   # Katakana
    "㐀-䶿"   # Han, perluasan A
    "一-鿿"   # Han
    "豈-﫿"   # Han, bentuk kompatibilitas
    "＀-￯"   # Bentuk lebar dan setengah lebar
    "가-힯"   # Hangul
    "　-〿"   # Tanda baca CJK
    "]+"
)

# Sisa setelah pembuangan: spasi ganda, dan spasi menggantung sebelum tanda baca.
_SPASI_GANDA = re.compile(r"[ \t]{2,}")
_SPASI_SEBELUM_TANDA = re.compile(r" +([,.;:!?)\]])")
_SPASI_SESUDAH_BUKA = re.compile(r"([(\[]) +")

# Di atas ambang ini, yang terjadi bukan "satu istilah bocor" melainkan model
# menjawab dalam bahasa yang salah. Jawabannya tidak dapat diselamatkan dengan
# membuang aksara, jadi dicatat lebih keras supaya terlihat saat dipantau.
_AMBANG_GAWAT = 0.05


def buang_cjk(teks: str) -> tuple[str, int]:
    """Buang aksara CJK dan rapikan sisa spasinya.

    Mengembalikan (teks bersih, jumlah aksara yang dibuang). Tidak mencatat
    apa pun — pemanggil yang memutuskan cara melaporkannya, karena jalur alir
    (streaming) harus mencatat sekali di akhir, bukan sekali per potongan.
    """
    if not teks:
        return teks, 0
    dibuang = sum(len(m) for m in _CJK.findall(teks))
    if not dibuang:
        return teks, 0

    bersih = _CJK.sub("", teks)
    bersih = _SPASI_GANDA.sub(" ", bersih)
    bersih = _SPASI_SEBELUM_TANDA.sub(r"\1", bersih)
    bersih = _SPASI_SESUDAH_BUKA.sub(r"\1", bersih)
    return bersih, dibuang


def catat_kebocoran(asli: str, dibuang: int, *, dari: str) -> None:
    """Catat satu kebocoran bahasa, dengan tingkat sesuai parahnya."""
    if not dibuang:
        return
    porsi = dibuang / max(len(asli), 1)
    contoh = "".join(_CJK.findall(asli))[:60]
    if porsi >= _AMBANG_GAWAT:
        logger.error(
            "{}: model menjawab sebagian besar dalam aksara non-Latin "
            "({} dari {} aksara, {:.0%}) — jawaban ini tidak dapat dipakai "
            "mahasiswa. Contoh: {!r}",
            dari, dibuang, len(asli), porsi, contoh,
        )
    else:
        logger.warning(
            "{}: {} aksara CJK dibuang dari jawaban ({:.1%}). Contoh: {!r}",
            dari, dibuang, porsi, contoh,
        )


def bersihkan_jawaban(teks: str, *, dari: str) -> str:
    """Bentuk siap pakai untuk jalur yang menghasilkan jawaban sekaligus."""
    bersih, dibuang = buang_cjk(teks)
    catat_kebocoran(teks, dibuang, dari=dari)
    return bersih
