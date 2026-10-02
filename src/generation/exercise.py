"""Pembuat tugas praktik untuk mahasiswa bertipe kinestetik.

Gaya belajar 'visual' sudah menghasilkan sesuatu yang bisa dilihat (diagram
Mermaid). Modul ini melengkapi sisi 'kinesthetic': bukan penjelasan lain, tapi
berkas latihan yang benar-benar bisa diunduh dan dikerjakan.

Formatnya TIDAK dipatok ke notebook. Model memilih sendiri berdasarkan materi
dan nama mata kuliah — notebook Python untuk Dasar Pemrograman masuk akal,
untuk Kewirausahaan tidak. Model mengembalikan JSON, lalu modul ini yang
merakitnya jadi berkas (.ipynb atau .md), sehingga struktur berkasnya dijamin
benar oleh kode, bukan diserahkan ke model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.generation.notebook import sel_kode as _sel_kode
from src.generation.notebook import sel_markdown as _sel_markdown
from src.utils.berkas import safe_filename as _safe_filename
from src.utils.logger import logger

# Format yang didukung. Model harus memilih salah satu dari daftar ini.
FORMAT_NOTEBOOK = "notebook"
FORMAT_LEMBAR_KERJA = "lembar_kerja"
_FORMAT_VALID = {FORMAT_NOTEBOOK, FORMAT_LEMBAR_KERJA}

# Bahasa yang punya kernel Jupyter di lingkungan mahasiswa pada umumnya.
# Di luar ini, notebook tidak akan bisa dijalankan, jadi turunkan ke lembar kerja.
_BAHASA_BERKERNEL = {"python"}

MAX_TUGAS = 5


EXERCISE_SYSTEM_PROMPT = """Anda perancang latihan praktik untuk mahasiswa yang belajar \
paling baik dengan MENGERJAKAN, bukan membaca.

Buat latihan berdasarkan MATERI yang diberikan. Jangan mengarang materi di luar konteks.

Pertama tentukan formatnya sendiri dari sifat mata kuliahnya:
- "notebook" — HANYA jika materinya benar-benar tentang pemrograman/komputasi dan \
mahasiswa perlu menulis serta menjalankan kode.
- "lembar_kerja" — untuk semua yang lain: analisis, perancangan, studi kasus, \
perhitungan manual, kewirausahaan, teori.

Jika ragu, pilih "lembar_kerja". Notebook berisi soal non-pemrograman lebih \
membingungkan daripada membantu.

Balas HANYA dengan JSON valid, tanpa penjelasan di luar JSON, dengan bentuk persis:
{
  "format": "notebook" | "lembar_kerja",
  "bahasa": "python" | null,
  "judul": "Judul latihan yang spesifik",
  "pengantar": "1-2 kalimat: apa yang akan dikuasai mahasiswa setelah mengerjakan ini.",
  "tugas": [
    {
      "instruksi": "Perintah yang jelas dan bisa dikerjakan.",
      "kode_awal": "Kerangka kode dengan komentar TODO. Kosongkan ('') untuk lembar_kerja.",
      "petunjuk": "Satu petunjuk yang menuntun tanpa membocorkan jawaban."
    }
  ]
}

Aturan:
- Buat 3 sampai 5 tugas, urut dari yang paling mudah.
- Tiap tugas harus bisa dikerjakan hanya dengan materi yang diberikan.
- "kode_awal" berisi kerangka dengan TODO, JANGAN berisi jawaban lengkap.
- Tulis semua teks dalam Bahasa Indonesia.
- "bahasa" diisi null bila formatnya lembar_kerja."""


@dataclass
class Tugas:
    instruksi: str
    kode_awal: str = ""
    petunjuk: str = ""


@dataclass
class Latihan:
    """Spesifikasi latihan yang sudah divalidasi dan siap dirakit jadi berkas."""

    format: str
    judul: str
    pengantar: str
    tugas: list[Tugas] = field(default_factory=list)
    bahasa: str | None = None

    @property
    def ekstensi(self) -> str:
        return "ipynb" if self.format == FORMAT_NOTEBOOK else "md"

    @property
    def mime(self) -> str:
        return "application/x-ipynb+json" if self.format == FORMAT_NOTEBOOK else "text/markdown"


def build_exercise_prompt(topic: str, class_name: str, context_block: str) -> str:
    """Susun prompt pengguna untuk pembuatan latihan."""
    return (
        f"[MATA KULIAH]\n{class_name}\n\n"
        f"[TOPIK]\n{topic}\n\n"
        f"[MATERI]\n{context_block}\n\n"
        "Rancang latihan praktik untuk topik di atas."
    )


def _ambil_json(raw: str) -> dict[str, Any] | None:
    """Ambil objek JSON dari keluaran model, toleran terhadap pagar kode."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        # Model kadang menambahi kalimat pembuka; ambil objek pertama yang utuh.
        match = re.search(r"\{[\s\S]*\}", clean)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


def parse_exercise_json(raw: str, topic: str) -> Latihan | None:
    """Ubah keluaran model jadi Latihan yang valid, atau None bila tidak terselamatkan.

    Sengaja mengembalikan None alih-alih latihan kosong: berkas berisi tugas
    palsu lebih merugikan mahasiswa daripada pesan galat yang jujur.
    """
    data = _ambil_json(raw)
    if not data:
        # Cuplikan keluaran ikut dicatat: tanpa itu kegagalan parse hanya terlihat
        # sebagai "gagal" dan tidak bisa didiagnosis setelah kejadian.
        logger.warning(
            "Keluaran latihan bukan JSON yang bisa dibaca | panjang={} awal={!r}",
            len(raw), raw[:200],
        )
        return None

    tugas: list[Tugas] = []
    for item in data.get("tugas") or []:
        if not isinstance(item, dict):
            continue
        instruksi = str(item.get("instruksi") or "").strip()
        if not instruksi:
            continue
        tugas.append(
            Tugas(
                instruksi=instruksi,
                kode_awal=str(item.get("kode_awal") or "").strip(),
                petunjuk=str(item.get("petunjuk") or "").strip(),
            )
        )
        if len(tugas) >= MAX_TUGAS:
            break

    if not tugas:
        logger.warning("Latihan tidak punya satu pun tugas yang bisa dipakai")
        return None

    fmt = str(data.get("format") or "").strip().lower()
    if fmt not in _FORMAT_VALID:
        fmt = FORMAT_LEMBAR_KERJA

    bahasa = data.get("bahasa")
    bahasa = str(bahasa).strip().lower() if bahasa else None

    # Notebook tanpa kernel yang tersedia tidak bisa dijalankan mahasiswa —
    # lebih baik diturunkan jadi lembar kerja daripada memberi berkas mati.
    if fmt == FORMAT_NOTEBOOK and bahasa not in _BAHASA_BERKERNEL:
        logger.info("Format notebook diturunkan ke lembar kerja (bahasa={})", bahasa)
        fmt = FORMAT_LEMBAR_KERJA
        bahasa = None

    return Latihan(
        format=fmt,
        judul=str(data.get("judul") or f"Latihan: {topic}").strip(),
        pengantar=str(data.get("pengantar") or "").strip(),
        tugas=tugas,
        bahasa=bahasa,
    )



def render_notebook(latihan: Latihan) -> str:
    """Rakit .ipynb versi 4 yang sah."""
    # Sejak nbformat_minor 5 setiap sel WAJIB punya id; tanpa itu nbformat
    # sekarang hanya memperingatkan, tapi versi mendatang menolaknya sebagai galat.
    sel: list[dict[str, Any]] = [
        _sel_markdown(f"# {latihan.judul}\n\n{latihan.pengantar}", "judul")
    ]

    for nomor, t in enumerate(latihan.tugas, 1):
        blok = f"## Tugas {nomor}\n\n{t.instruksi}"
        if t.petunjuk:
            blok += f"\n\n> **Petunjuk:** {t.petunjuk}"
        sel.append(_sel_markdown(blok, f"tugas-{nomor}-soal"))
        sel.append(_sel_kode(t.kode_awal or "# Tulis jawaban Anda di sini\n",
                             f"tugas-{nomor}-jawab"))

    notebook = {
        "cells": sel,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1)


def render_lembar_kerja(latihan: Latihan) -> str:
    """Rakit lembar kerja Markdown."""
    bagian = [f"# {latihan.judul}", ""]
    if latihan.pengantar:
        bagian += [latihan.pengantar, ""]

    for nomor, t in enumerate(latihan.tugas, 1):
        bagian += [f"## Tugas {nomor}", "", t.instruksi, ""]
        if t.petunjuk:
            bagian += [f"> **Petunjuk:** {t.petunjuk}", ""]
        if t.kode_awal:
            bagian += ["```", t.kode_awal, "```", ""]
        bagian += ["**Jawaban Anda:**", "", "_(tulis di sini)_", "", "---", ""]

    return "\n".join(bagian)


def render_exercise(latihan: Latihan) -> str:
    """Rakit berkas sesuai formatnya."""
    if latihan.format == FORMAT_NOTEBOOK:
        return render_notebook(latihan)
    return render_lembar_kerja(latihan)


def safe_filename(judul: str, ekstensi: str) -> str:
    """Nama berkas latihan yang aman diunduh.

    Pembungkus tipis atas `src.utils.berkas.safe_filename`; hanya nilai
    cadangannya yang berbeda dari pemakaian di notebook.
    """
    return _safe_filename(judul, ekstensi, cadangan="latihan")
