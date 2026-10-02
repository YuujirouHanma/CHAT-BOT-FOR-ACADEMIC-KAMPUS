"""Ubah jawaban bergaya praktik menjadi berkas notebook (.ipynb).

Gaya belajar "praktik" meminta LLM menulis kode dalam blok berpagar. Dari situ
notebook dibentuk secara DETERMINISTIK — tanpa panggilan LLM tambahan, jadi
tidak menambah biaya maupun waktu tunggu, dan tidak ada risiko model mengarang
isi yang berbeda dari jawaban yang sudah dibaca mahasiswa.

Teks di luar blok kode menjadi sel markdown, sehingga penjelasan dan kodenya
tetap berdampingan seperti di layar chat.
"""
from __future__ import annotations

import json
import re

from src.utils.berkas import safe_filename as _safe_filename

# Blok kode berpagar beserta penanda bahasanya.
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)

# Bahasa yang masuk akal dijalankan di dalam notebook Python.
_RUNNABLE = {"python", "py", "python3", ""}


def sel_markdown(sumber, sel_id: str | None = None) -> dict:
    """Satu sel markdown nbformat 4.

    `sel_id` diisi untuk nbformat 4.5 ke atas, yang mewajibkan setiap sel punya
    pengenal. Dibiarkan kosong pada pemanggil lama supaya keluarannya tidak
    berubah bentuk.
    """
    sel = {"cell_type": "markdown", "metadata": {}, "source": sumber}
    if sel_id is not None:
        sel["id"] = sel_id
    return sel


def sel_kode(sumber, sel_id: str | None = None) -> dict:
    """Satu sel kode nbformat 4."""
    sel = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sumber,
    }
    if sel_id is not None:
        sel["id"] = sel_id
    return sel


# Nama lama dipertahankan sebagai pembungkus: keduanya memecah teks menjadi
# daftar baris lebih dulu, bentuk yang dipakai pemanggil di berkas ini.
def _markdown_cell(text: str) -> dict:
    return sel_markdown(text.splitlines(True))


def _code_cell(code: str) -> dict:
    return sel_kode(code.splitlines(True))



def has_code(answer: str) -> bool:
    """True bila jawaban memuat blok kode yang layak dijadikan notebook."""
    return any(
        (bahasa or "").lower() in _RUNNABLE and isi.strip()
        for bahasa, isi in _FENCE_RE.findall(answer or "")
    )


def build_notebook(answer: str, title: str = "Catatan Belajar") -> str | None:
    """Bentuk notebook dari sebuah jawaban; None bila tidak ada kode di dalamnya.

    Blok berbahasa selain Python (mis. ```sql) tetap disimpan sebagai markdown
    supaya isinya tidak hilang, tetapi tidak dijadikan sel kode yang dieksekusi.
    """
    teks = answer or ""
    if not has_code(teks):
        return None

    cells: list[dict] = [_markdown_cell(f"# {title}\n")]
    posisi = 0
    for m in _FENCE_RE.finditer(teks):
        sebelum = teks[posisi:m.start()].strip()
        if sebelum:
            cells.append(_markdown_cell(sebelum + "\n"))
        bahasa = (m.group(1) or "").lower()
        isi = m.group(2).rstrip()
        if isi:
            if bahasa in _RUNNABLE:
                cells.append(_code_cell(isi))
            else:
                cells.append(_markdown_cell(f"```{bahasa}\n{isi}\n```\n"))
        posisi = m.end()

    sisa = teks[posisi:].strip()
    if sisa:
        cells.append(_markdown_cell(sisa + "\n"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1)


def safe_filename(base: str) -> str:
    """Nama berkas notebook yang aman di semua sistem berkas.

    Pembungkus tipis atas `src.utils.berkas.safe_filename`, dipertahankan
    agar pemanggil lama tidak perlu ikut diubah.
    """
    return _safe_filename(base, "ipynb", cadangan="catatan")
