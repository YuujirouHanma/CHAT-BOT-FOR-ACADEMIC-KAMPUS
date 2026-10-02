"""Pembantu penamaan berkas unduhan.

Fungsi ini sebelumnya ada DUA KALI dengan nama yang persis sama: satu di
`src/generation/notebook.py` yang selalu menghasilkan `.ipynb`, satu lagi di
`src/generation/exercise.py` yang menerima ekstensi sebagai argumen. Keduanya
membersihkan judul dengan aturan yang sama, hanya berbeda nilai cadangannya.

Dua salinan untuk satu aturan penamaan berarti perbaikan pada salah satunya —
misalnya menangani nama berkas yang dilarang Windows — tidak pernah sampai ke
yang lain, dan tidak ada yang akan menyadarinya sampai ada mahasiswa yang
unduhannya gagal.
"""
from __future__ import annotations

import re


def safe_filename(judul: str, ekstensi: str = "ipynb", *,
                  cadangan: str = "catatan") -> str:
    """Nama berkas yang aman diunduh di semua sistem berkas.

    Membuang tanda baca, merapatkan spasi menjadi tanda hubung, dan memotong
    pada 60 huruf. `cadangan` dipakai bila judulnya kosong atau habis
    seluruhnya oleh pembersihan — misalnya judul yang isinya hanya tanda baca.
    """
    bersih = re.sub(r"[^\w\s-]", "", judul or "", flags=re.UNICODE).strip()
    bersih = re.sub(r"[\s_]+", "-", bersih).strip("-").lower()
    if not bersih:
        bersih = cadangan
    return f"{bersih[:60]}.{ekstensi}"
