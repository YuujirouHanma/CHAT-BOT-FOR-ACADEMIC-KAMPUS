"""Penyusunan peta konsep tingkat minggu dari chunk yang sudah terindeks.

Berbeda dari ``/guided/topic-hints`` yang mengambil 8 chunk sembarang lalu
menebak topiknya, modul ini membaca SELURUH chunk milik satu
``(content_id, section_id)``. Akibatnya hasilnya mencakup semua berkas minggu
itu dan sama tiap kali dijalankan — syarat yang harus dipenuhi sebelum apa pun
bisa dilacak per konsep.

Sengaja membaca dari Qdrant, bukan menempel di jalur unggah:

- materi yang sudah terindeks tidak perlu diindeks ulang;
- satu minggu bisa berisi beberapa berkas, dan membaca per minggu membuat
  penggabungan lintas berkas terjadi dengan sendirinya — tidak perlu langkah
  konsolidasi terpisah;
- bisa dijalankan ulang kapan saja setelah dosen menambah berkas.

Urutan pengajaran tidak ditebak model: ia diturunkan dari ``page_number`` yang
sudah tersimpan di payload sejak indexing, karena dosen memang sudah menyusun
materinya secara berurutan.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src import usage
from src.generation.llm import LLMGenerator
from src.storage.qdrant_store import QdrantStore
from src.utils.logger import logger

# Berapa banyak konsep yang boleh keluar per minggu. Sengaja kasar: konsep
# sebesar "topik yang ditulis dosen di rencana pembelajaran", bukan istilah
# tunggal. Butiran halus membuat penguasaan mustahil diukur tanpa membanjiri
# mahasiswa dengan pemeriksaan.
MIN_KONSEP = 3
MAKS_KONSEP = 5

# Anggaran konteks. Minggu tebal bisa punya ratusan chunk; yang tidak boleh
# dilakukan adalah memotong ke N pertama, karena itu persis kesalahan yang
# sedang diperbaiki. Bila melebihi anggaran, potongan diambil MERATA di seluruh
# rentang materi sehingga cakupannya tetap utuh meski jumlahnya dibatasi.
MAKS_CHUNK_KONTEKS = 60
MAKS_KARAKTER_PER_CHUNK = 900

_PAYLOAD_FIELDS = (
    "text",
    "source_file",
    "page_number",
    "chunk_index",
    "element_type",
)


class SumberKonsep(BaseModel):
    """Di mana sebuah konsep diajarkan, dalam satuan berkas dan halaman."""

    source_file: str
    halaman_awal: int | None = None
    halaman_akhir: int | None = None


class Konsep(BaseModel):
    nama: str
    deskripsi: str = ""
    urutan: int = 0
    sumber: list[SumberKonsep] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class PetaKonsep(BaseModel):
    """Hasil satu kali penyusunan, lengkap dengan angka untuk menilai cakupan."""

    content_id: str
    section_id: int | None = None
    konsep: list[Konsep] = Field(default_factory=list)
    jumlah_chunk: int = 0
    chunk_dipakai: int = 0
    berkas: list[str] = Field(default_factory=list)
    # chunk_id yang disodorkan ke model, dikelompokkan per berkas. Ini yang
    # memungkinkan cakupan dihitung per berkas: berkas yang nol persen tercakup
    # berarti dianggap tidak relevan, dan hanya manusia yang bisa memutuskan
    # apakah itu benar (berkas nyasar) atau salah (materi terlewat).
    chunk_per_berkas: dict[str, list[str]] = Field(default_factory=dict)
    # Jumlah potongan SEBENARNYA per berkas, dihitung dari seluruh materi minggu
    # itu — bukan dari yang disodorkan ke model.
    #
    # Keduanya perlu ada karena `chunk_per_berkas` sudah tersaring oleh
    # `pilih_merata`: begitu satu minggu punya lebih dari MAKS_CHUNK_KONTEKS
    # potongan, isinya hanya cuplikan. Dulu cakupan dihitung dari peta itu saja,
    # sehingga berkas 120 potongan dilaporkan bertotal 60 — angka yang terlihat
    # masuk akal, tidak pernah memicu galat, dan menyembunyikan tepat satu hal
    # yang perlu diketahui: separuh materinya tidak pernah dibaca model.
    total_per_berkas: dict[str, int] = Field(default_factory=dict)
    keluaran_mentah: str = ""


_SISTEM_PROMPT = (
    "Kamu adalah asisten kurikulum yang menyusun peta konsep dari materi "
    "kuliah. Kamu menjawab HANYA dengan JSON valid, tanpa kalimat pembuka, "
    "tanpa penutup, dan tanpa blok kode."
)

_PROMPT = """Potongan berikut adalah SELURUH materi yang terindeks untuk {section_name} pada mata kuliah {class_name}. Tiap potongan diberi nomor beserta berkas dan halaman asalnya.

Tugasmu: tentukan {min_konsep}-{maks_konsep} KONSEP UTAMA yang diajarkan pada minggu ini.

Aturan:
- Konsep utama adalah topik yang akan ditulis dosen di rencana pembelajaran, bukan istilah tunggal atau detail kecil.
- Gabungkan potongan dari berkas yang berbeda bila keduanya membahas konsep yang sama.
- Bila ada potongan yang jelas tidak berhubungan dengan mata kuliah ini, abaikan saja. Jangan memaksakannya menjadi konsep.
- Setiap konsep harus menyebutkan nomor potongan yang mengajarkannya.
- Tulis dalam bahasa Indonesia.

Balas HANYA dengan JSON berbentuk:
[
  {{"nama": "Nama konsep", "deskripsi": "Satu kalimat tentang apa yang dipelajari mahasiswa di sini", "potongan": [1, 4, 7]}}
]

[MATERI]
{materi}"""


async def ambil_chunk_minggu(
    store: QdrantStore,
    content_id: str,
    section_id: int | None = None,
    *,
    tenant_id: str,
) -> list[dict]:
    """Ambil seluruh chunk satu minggu, terurut seperti urutan materi diajarkan.

    Pengurutan memakai (berkas, halaman, indeks chunk) supaya nomor potongan
    yang dilihat model naik seiring materi berjalan — itu yang membuat urutan
    konsep bisa diturunkan tanpa bertanya ke model.
    """
    mentah = await store.scroll_all(
        content_id=content_id,
        section_ids=[section_id] if section_id is not None else None,
        payload_fields=_PAYLOAD_FIELDS,
        tenant_id=tenant_id,
    )

    chunks: list[dict] = []
    for item in mentah:
        payload = item["payload"]
        teks = (payload.get("text") or "").strip()
        if not teks:
            continue
        chunks.append(
            {
                "chunk_id": item["chunk_id"],
                "text": teks,
                "source_file": payload.get("source_file") or "(tanpa berkas)",
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index") or 0,
                "element_type": payload.get("element_type") or "text",
            }
        )

    chunks.sort(
        key=lambda c: (
            c["source_file"],
            c["page_number"] if c["page_number"] is not None else 10**6,
            c["chunk_index"],
        )
    )
    return chunks


def pilih_merata(chunks: Sequence[dict], maks: int = MAKS_CHUNK_KONTEKS) -> list[dict]:
    """Ambil paling banyak `maks` potongan, tersebar merata di seluruh materi.

    Dipakai hanya bila materi minggu itu melebihi anggaran konteks. Mengambil
    yang pertama akan membuat konsep di paruh akhir materi tidak pernah
    terlihat; mengambil merata menjaga cakupan seluruh rentang halaman.
    """
    jumlah = len(chunks)
    if jumlah <= maks:
        return list(chunks)

    langkah = jumlah / maks
    return [chunks[min(int(i * langkah), jumlah - 1)] for i in range(maks)]


def _susun_materi(chunks: Sequence[dict]) -> str:
    bagian: list[str] = []
    for nomor, c in enumerate(chunks, start=1):
        teks = c["text"]
        if len(teks) > MAKS_KARAKTER_PER_CHUNK:
            teks = teks[:MAKS_KARAKTER_PER_CHUNK] + " […]"
        halaman = c["page_number"]
        letak = f"{c['source_file']}, hal. {halaman}" if halaman else c["source_file"]
        jenis = "" if c["element_type"] == "text" else f", {c['element_type']}"
        bagian.append(f"[{nomor}] ({letak}{jenis})\n{teks}")
    return "\n\n".join(bagian)


def _urai_json(mentah: str) -> list[dict]:
    """Ambil daftar konsep dari keluaran model, toleran terhadap bungkus tambahan."""
    bersih = mentah.replace("```json", "").replace("```", "").strip()

    kandidat = [bersih]
    cocok = re.search(r"\[[\s\S]*\]", bersih)
    if cocok:
        kandidat.append(cocok.group(0))

    for teks in kandidat:
        try:
            data = json.loads(teks)
        except Exception:
            continue
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

    return []


def _bangun_sumber(chunks: Sequence[dict]) -> list[SumberKonsep]:
    """Rangkum potongan sebuah konsep menjadi rentang halaman per berkas."""
    per_berkas: dict[str, list[int]] = {}
    for c in chunks:
        per_berkas.setdefault(c["source_file"], [])
        if c["page_number"] is not None:
            per_berkas[c["source_file"]].append(int(c["page_number"]))

    return [
        SumberKonsep(
            source_file=berkas,
            halaman_awal=min(halaman) if halaman else None,
            halaman_akhir=max(halaman) if halaman else None,
        )
        for berkas, halaman in per_berkas.items()
    ]


def _rakit_konsep(data: Sequence[dict], chunks: Sequence[dict]) -> list[Konsep]:
    """Ubah keluaran model jadi Konsep, lalu urutkan mengikuti urutan materi."""
    terkumpul: list[tuple[int, Konsep]] = []

    for butir in data:
        nama = str(butir.get("nama") or "").strip()
        if not nama:
            continue

        # Nomor di luar jangkauan diabaikan diam-diam: model kadang menyebut
        # potongan yang tidak ada, dan itu tidak boleh menggagalkan sisanya.
        posisi: list[int] = []
        for n in butir.get("potongan") or []:
            try:
                idx = int(n) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(chunks):
                posisi.append(idx)

        milik = [chunks[i] for i in sorted(set(posisi))]
        # Konsep tanpa potongan yang sah didorong ke belakang, bukan dibuang:
        # namanya masih berguna, hanya letaknya di materi yang tidak diketahui.
        awal = min(posisi) if posisi else 10**6
        terkumpul.append(
            (
                awal,
                Konsep(
                    nama=nama,
                    deskripsi=str(butir.get("deskripsi") or "").strip(),
                    sumber=_bangun_sumber(milik),
                    chunk_ids=[c["chunk_id"] for c in milik],
                ),
            )
        )

    terkumpul.sort(key=lambda p: p[0])
    hasil = [k for _, k in terkumpul[:MAKS_KONSEP]]
    for nomor, konsep in enumerate(hasil, start=1):
        konsep.urutan = nomor
    return hasil


async def susun_peta_konsep(
    store: QdrantStore,
    generator: LLMGenerator,
    content_id: str,
    section_id: int | None = None,
    class_name: str = "Mata Kuliah",
    section_name: str = "Minggu Ini",
    *,
    tenant_id: str,
) -> PetaKonsep:
    """Susun peta konsep satu minggu dari chunk yang sudah ada di Qdrant.

    Mengembalikan PetaKonsep kosong (tanpa galat) bila minggu itu belum punya
    chunk sama sekali — belum ada materi bukan kondisi luar biasa.
    """
    chunks = await ambil_chunk_minggu(store, content_id, section_id, tenant_id=tenant_id)
    if not chunks:
        logger.info(
            "Peta konsep dilewati: tidak ada chunk untuk content_id={} section_id={}",
            content_id, section_id,
        )
        return PetaKonsep(content_id=content_id, section_id=section_id)

    dipakai = pilih_merata(chunks)
    berkas = sorted({c["source_file"] for c in chunks})

    logger.info(
        "Menyusun peta konsep | content_id={} section_id={} chunk={} dipakai={} berkas={}",
        content_id, section_id, len(chunks), len(dipakai), len(berkas),
    )

    prompt = _PROMPT.format(
        section_name=section_name,
        class_name=class_name,
        min_konsep=MIN_KONSEP,
        maks_konsep=MAKS_KONSEP,
        materi=_susun_materi(dipakai),
    )

    pesan: list[Any] = [
        {"role": "system", "content": _SISTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # reasoning=False: ini panggilan bantu di luar jalur jawaban mahasiswa, dan
    # token berpikir berbagi jatah dengan keluaran JSON-nya — lihat
    # src/generation/llm.py::_reasoning_body.
    usage.pasang_tujuan("peta-konsep")
    mentah = await generator._call_with_retry(
        pesan, temperature=0.2, max_tokens=1500, reasoning=False
    )

    per_berkas: dict[str, list[str]] = {}
    for c in dipakai:
        per_berkas.setdefault(c["source_file"], []).append(c["chunk_id"])

    total_berkas: dict[str, int] = {}
    for c in chunks:
        total_berkas[c["source_file"]] = total_berkas.get(c["source_file"], 0) + 1

    konsep = _rakit_konsep(_urai_json(mentah), dipakai)
    if not konsep:
        logger.warning(
            "Model tidak menghasilkan konsep yang bisa dipakai untuk {} minggu {}",
            content_id, section_id,
        )

    return PetaKonsep(
        content_id=content_id,
        section_id=section_id,
        konsep=konsep,
        jumlah_chunk=len(chunks),
        chunk_dipakai=len(dipakai),
        berkas=berkas,
        chunk_per_berkas=per_berkas,
        total_per_berkas=total_berkas,
        keluaran_mentah=mentah,
    )
