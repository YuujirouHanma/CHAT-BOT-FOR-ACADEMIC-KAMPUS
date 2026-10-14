"""Penyusunan bank soal pilihan ganda, satu berkas per konsep.

Soal dibuat PER KONSEP, bukan per minggu. Alasannya sama dengan alasan peta
konsep disimpan: yang mau diukur adalah penguasaan sebuah konsep, dan skor
minggu tidak bisa dipecah kembali menjadi skor konsep setelah tercampur.

Materi tiap soal diambil dari potongan yang mengajarkan konsep itu — berkas dan
rentang halaman yang sudah dicatat di ``concept_source`` saat peta disusun, dan
sudah dilihat dosen. Jadi soal tidak lahir dari seluruh minggu lalu diberi label
konsep; ia lahir dari bagian materi yang memang menjadi konsep tersebut.

Tiap konsep menghasilkan dua soal dengan tingkat yang sengaja berbeda:

- ``ingatan`` — mahasiswa mengingat dan memahami isi materi;
- ``penerapan`` — mahasiswa memakai konsep pada situasi yang tidak ada di materi.

Pembedaan ini gratis (jumlah soalnya sama) tapi membuat kenaikan skor bisa
dibaca lebih tajam: naik di ingatan saja berarti mahasiswa menghafal lebih
banyak, naik di penerapan berarti ia benar-benar bisa memakainya.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from src import usage
from src.generation.llm import LLMGenerator
from src.indexing.concepts import _susun_materi, ambil_chunk_minggu, pilih_merata
from src.storage.qdrant_store import QdrantStore
from src.utils.logger import logger

# Dua tipe soal: ingatan dan penerapan (lihat SoalKonsep.tipe). Tiap soal yang
# dihasilkan model ditandai salah satu dari keduanya; jumlahnya sendiri
# sekarang bisa diatur dosen, lihat SOAL_PER_KONSEP_DEFAULT/_MAKS di bawah.
TIPE_SOAL: tuple[str, ...] = ("ingatan", "penerapan")

# Bawaan bila dosen tidak menyebutkan jumlah: satu ingatan, satu penerapan.
SOAL_PER_KONSEP_DEFAULT = len(TIPE_SOAL)

# Batas atas jumlah soal per konsep per panggilan. Angkanya bukan selera —
# 3-5 konsep dikali dua bawaan menghasilkan 6-10 soal, yang masih selesai
# dalam 5-8 menit. Uji yang lebih panjang dikerjakan asal-asalan, dan data
# asal-asalan lebih merugikan daripada data yang sedikit. Dosen yang memang
# ingin lebih banyak masih bisa memanggil "tambah soal" berkali-kali, yang
# menumpuk di atas soal lama alih-alih memaksa satu panggilan model membuat
# terlalu banyak soal sekaligus dari materi yang sama.
SOAL_PER_KONSEP_MAKS = 10

# Empat pilihan: peluang tebak 25%. Lima pilihan menurunkannya jadi 20% tapi
# menambah satu pengecoh yang harus tetap masuk akal, dan pengecoh yang dipaksa
# ada justru membuat soal lebih mudah ditebak lewat eliminasi.
JUMLAH_OPSI = 4

# Anggaran konteks per konsep. Jauh lebih kecil daripada anggaran peta konsep
# karena di sini materinya sudah dipersempit ke satu konsep, bukan satu minggu.
MAKS_CHUNK_PER_KONSEP = 14


class OpsiSoal(BaseModel):
    teks: str
    benar: bool = False


class SoalKonsep(BaseModel):
    concept_id: int | None = None
    nama_konsep: str = ""
    tipe: str = "ingatan"
    pertanyaan: str
    opsi: list[OpsiSoal] = Field(default_factory=list)
    pembahasan: str = ""
    urutan: int = 0


class KonsepMasukan(BaseModel):
    """Konsep yang sudah tersimpan dan (semoga) sudah ditinjau dosen.

    ``sumber`` boleh kosong: konsep yang ditambahkan dosen sendiri lewat panel
    peta konsep tidak punya baris ``concept_source``. Kasus itu ditangani
    dengan mundur ke materi minggu, bukan dengan menolak konsepnya.
    """

    concept_id: int | None = None
    nama: str
    deskripsi: str = ""
    sumber: list[dict] = Field(default_factory=list)


class BankSoal(BaseModel):
    content_id: str
    section_id: int | None = None
    soal: list[SoalKonsep] = Field(default_factory=list)
    # Konsep yang tidak menghasilkan satu soal sah pun. Dilaporkan, bukan
    # didiamkan: konsep tanpa soal berarti ada bagian minggu yang tidak akan
    # pernah terukur, dan hanya dosen yang bisa memutuskan apakah itu penting.
    konsep_gagal: list[str] = Field(default_factory=list)


_SISTEM_PROMPT = (
    "Kamu adalah penyusun soal ujian mata kuliah. Kamu menjawab HANYA dengan "
    "JSON valid, tanpa kalimat pembuka, tanpa penutup, dan tanpa blok kode."
)

_PROMPT = """Potongan materi berikut diambil dari {section_name} pada mata kuliah {class_name}, dan merupakan bagian yang mengajarkan konsep "{nama_konsep}".{deskripsi}

Tugasmu: buat TEPAT {jumlah_soal} soal pilihan ganda untuk mengukur penguasaan mahasiswa atas konsep ini.

Tiap soal bertipe salah satu dari dua ini:
- "ingatan": menguji apakah mahasiswa mengingat dan memahami apa yang diajarkan materi.
- "penerapan": menyajikan situasi BARU yang tidak disebutkan di materi, lalu menguji apakah mahasiswa bisa memakai konsep ini untuk memutuskan sesuatu.

Campur kedua tipe itu — jangan seluruh soal bertipe sama — supaya kenaikan skor bisa dibaca lebih tajam: naik di ingatan saja berarti mahasiswa menghafal lebih banyak, naik di penerapan berarti ia benar-benar bisa memakainya.

Aturan:
- Tiap soal punya TEPAT {jumlah_opsi} pilihan dan TEPAT SATU jawaban benar.
- Ketiga pengecoh harus masuk akal bagi mahasiswa yang belum paham. Pengecoh yang jelas-jelas salah membuat soal tidak mengukur apa pun.
- Panjang semua pilihan harus mirip. Jawaban yang paling panjang atau paling rinci mudah ditebak tanpa memahami materi.
- Dilarang memakai pilihan seperti "semua benar", "tidak ada yang benar", atau "A dan B benar".
- Jawaban harus bisa ditentukan dari materi di bawah saja, tanpa pengetahuan dari luar.
- Pembahasan berisi satu sampai dua kalimat: kenapa kuncinya benar, dan kenapa pengecoh yang paling menggoda itu salah.
- Di dalam pembahasan, JANGAN menyebut pilihan dengan huruf atau nomor ("pilihan A", "opsi 2"). Sebut isinya, misalnya "pilihan yang menyatakan hasilnya dibulatkan ke bawah". Urutan pilihan diacak sebelum sampai ke mahasiswa, jadi huruf apa pun yang kamu sebut akan menunjuk ke pilihan yang salah.
- Jangan menyebut nomor potongan, nama berkas, atau nomor halaman di dalam soal.
- Setiap soal harus menguji sisi berbeda dari konsep ini — jangan menulis ulang soal yang sama dengan kalimat berbeda.
- Tulis dalam bahasa Indonesia.

Balas HANYA dengan JSON array berisi TEPAT {jumlah_soal} objek, masing-masing berbentuk:
{{"tipe": "ingatan" | "penerapan", "pertanyaan": "...", "opsi": ["...", "...", "...", "..."], "kunci": 0, "pembahasan": "..."}}

"kunci" adalah nomor urut jawaban benar di dalam "opsi", dihitung mulai dari 0.

[MATERI]
{materi}"""


def _dalam_rentang(halaman: Any, awal: Any, akhir: Any) -> bool:
    """Apakah sebuah halaman jatuh di dalam rentang yang dicatat konsep.

    Rentang yang tidak diketahui (mis. berkas .txt yang tidak punya halaman)
    dianggap mencakup seluruh berkas. Menolaknya akan membuang materi yang sah
    hanya karena formatnya tidak bernomor halaman.
    """
    if awal is None and akhir is None:
        return True
    if halaman is None:
        return True
    try:
        h = int(halaman)
    except (TypeError, ValueError):
        return True
    if awal is not None and h < int(awal):
        return False
    if akhir is not None and h > int(akhir):
        return False
    return True


def saring_chunk_konsep(chunks: Sequence[dict], sumber: Sequence[dict]) -> list[dict]:
    """Ambil potongan minggu yang jatuh di dalam sumber sebuah konsep.

    Mengembalikan daftar kosong bila konsep tidak punya sumber sama sekali —
    pemanggil yang memutuskan apa artinya.
    """
    if not sumber:
        return []

    rentang: dict[str, list[tuple[Any, Any]]] = {}
    for s in sumber:
        berkas = str(s.get("source_file") or "").strip()
        if not berkas:
            continue
        rentang.setdefault(berkas, []).append((s.get("halaman_awal"), s.get("halaman_akhir")))

    if not rentang:
        return []

    terpilih: list[dict] = []
    for c in chunks:
        for awal, akhir in rentang.get(c["source_file"], []):
            if _dalam_rentang(c["page_number"], awal, akhir):
                terpilih.append(c)
                break

    return terpilih


def _urai_json(mentah: str) -> list[dict]:
    """Ambil daftar soal dari keluaran model, toleran terhadap bungkus tambahan."""
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


def _rakit_soal(
    data: Sequence[dict],
    konsep: KonsepMasukan,
    jumlah_soal: int = SOAL_PER_KONSEP_DEFAULT,
    acak: bool = True,
) -> list[SoalKonsep]:
    """Ubah keluaran model jadi soal yang sah, buang yang cacat.

    Soal cacat dibuang diam-diam alih-alih menggagalkan seluruh konsep: satu
    soal penerapan yang rusak tidak boleh ikut membuang soal ingatan yang sudah
    benar. Yang dilaporkan hanyalah konsep yang tidak menyisakan soal apa pun.

    Urutan pilihan diacak. Model punya kecenderungan menaruh jawaban benar di
    posisi pertama, dan bank soal yang kuncinya selalu di A bisa dikerjakan
    tanpa membaca soalnya sama sekali.
    """
    hasil: list[SoalKonsep] = []

    for butir in data:
        pertanyaan = str(butir.get("pertanyaan") or "").strip()
        opsi_mentah = butir.get("opsi")
        if not pertanyaan or not isinstance(opsi_mentah, list):
            continue

        teks_opsi = [str(o).strip() for o in opsi_mentah]
        if len(teks_opsi) != JUMLAH_OPSI or any(not t for t in teks_opsi):
            continue
        # Pilihan kembar membuat soal tidak punya satu jawaban benar, apa pun
        # kuncinya. Perbandingan tanpa memandang huruf besar-kecil karena
        # "Fotosintesis" dan "fotosintesis" tetap satu pilihan yang sama.
        if len({t.casefold() for t in teks_opsi}) != JUMLAH_OPSI:
            continue

        try:
            kunci = int(butir.get("kunci"))
        except (TypeError, ValueError):
            continue
        if not 0 <= kunci < JUMLAH_OPSI:
            continue

        tipe = str(butir.get("tipe") or "").strip().lower()
        if tipe not in TIPE_SOAL:
            # Alih-alih menumpuk semuanya ke tipe terakhir, alternasikan by
            # posisi — dengan jumlah soal yang sekarang bisa lebih dari dua,
            # capping ke tipe terakhir akan membuat mayoritas soal bertipe
            # sama persis saat model lupa menyebut tipenya.
            tipe = TIPE_SOAL[len(hasil) % len(TIPE_SOAL)]

        opsi = [OpsiSoal(teks=t, benar=(i == kunci)) for i, t in enumerate(teks_opsi)]
        if acak:
            random.shuffle(opsi)

        hasil.append(
            SoalKonsep(
                concept_id=konsep.concept_id,
                nama_konsep=konsep.nama,
                tipe=tipe,
                pertanyaan=pertanyaan,
                opsi=opsi,
                pembahasan=str(butir.get("pembahasan") or "").strip(),
            )
        )

    # Lebih dari yang diminta dipangkas: prompt meminta tepat jumlah_soal, dan
    # kelebihannya berarti model mengarang di luar permintaan.
    hasil = hasil[:jumlah_soal]
    for nomor, soal in enumerate(hasil, start=1):
        soal.urutan = nomor
    return hasil


async def _soal_satu_konsep(
    generator: LLMGenerator,
    konsep: KonsepMasukan,
    chunks: Sequence[dict],
    class_name: str,
    section_name: str,
    jumlah_soal: int = SOAL_PER_KONSEP_DEFAULT,
    acak: bool = True,
) -> list[SoalKonsep]:
    if not chunks:
        return []

    deskripsi = f" Konsep ini dijelaskan sebagai: {konsep.deskripsi}" if konsep.deskripsi else ""
    prompt = _PROMPT.format(
        section_name=section_name,
        class_name=class_name,
        nama_konsep=konsep.nama,
        deskripsi=deskripsi,
        jumlah_opsi=JUMLAH_OPSI,
        jumlah_soal=jumlah_soal,
        materi=_susun_materi(chunks),
    )

    pesan: list[Any] = [
        {"role": "system", "content": _SISTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # reasoning=False dengan alasan yang sama seperti penyusunan peta konsep:
    # ini panggilan bantu di luar jalur jawaban mahasiswa, dan token berpikir
    # berbagi jatah dengan keluaran JSON-nya — lihat llm.py::_reasoning_body.
    # max_tokens diskalakan dengan jumlah_soal: ~900 token/soal (empat pilihan
    # + pembahasan) plus sedikit ruang tetap, diukur dari bawaan 1800 token
    # untuk dua soal.
    usage.pasang_tujuan("bank-soal")
    mentah = await generator._call_with_retry(
        pesan, temperature=0.3, max_tokens=200 + 900 * jumlah_soal, reasoning=False
    )

    return _rakit_soal(_urai_json(mentah), konsep, jumlah_soal=jumlah_soal, acak=acak)


async def susun_bank_soal(
    store: QdrantStore,
    generator: LLMGenerator,
    content_id: str,
    konsep: Sequence[KonsepMasukan],
    section_id: int | None = None,
    class_name: str = "Mata Kuliah",
    section_name: str = "Minggu Ini",
    jumlah_soal: int = SOAL_PER_KONSEP_DEFAULT,
    acak_opsi: bool = True,
    *,
    tenant_id: str,
) -> BankSoal:
    """Susun soal untuk seluruh konsep satu minggu.

    Satu panggilan LLM per konsep, dijalankan bersamaan. Menggabungkan semua
    konsep ke dalam satu panggilan memang lebih murah, tapi keluarannya jadi
    panjang dan gampang terpotong di tengah — dan soal yang terpotong tidak
    bisa diselamatkan, sementara satu konsep yang gagal masih menyisakan
    sisanya.

    ``jumlah_soal`` berlaku sama untuk SETIAP konsep dalam satu panggilan ini.
    Diklem ke [1, SOAL_PER_KONSEP_MAKS] di sini juga, bukan hanya di skema API
    — fungsi ini dipanggil dari lebih dari satu jalur.
    """
    if not konsep:
        return BankSoal(content_id=content_id, section_id=section_id)

    jumlah_soal = max(1, min(jumlah_soal, SOAL_PER_KONSEP_MAKS))

    semua = await ambil_chunk_minggu(store, content_id, section_id, tenant_id=tenant_id)
    if not semua:
        logger.info(
            "Bank soal dilewati: tidak ada chunk untuk content_id={} section_id={}",
            content_id, section_id,
        )
        return BankSoal(
            content_id=content_id,
            section_id=section_id,
            konsep_gagal=[k.nama for k in konsep],
        )

    # Materi cadangan untuk konsep yang tidak punya sumber, yaitu konsep yang
    # ditambahkan dosen sendiri. Diambil merata di seluruh minggu supaya
    # cadangannya tetap mewakili keseluruhan, bukan bagian awalnya saja.
    cadangan = pilih_merata(semua, MAKS_CHUNK_PER_KONSEP)

    bahan: list[Sequence[dict]] = []
    for k in konsep:
        milik = saring_chunk_konsep(semua, k.sumber)
        if not milik:
            logger.info(
                "Konsep '{}' tidak punya potongan sendiri — memakai materi minggu",
                k.nama,
            )
            milik = cadangan
        bahan.append(pilih_merata(milik, MAKS_CHUNK_PER_KONSEP))

    logger.info(
        "Menyusun bank soal | content_id={} section_id={} konsep={} chunk={}",
        content_id, section_id, len(konsep), len(semua),
    )

    hasil = await asyncio.gather(
        *(
            _soal_satu_konsep(
                generator, k, potongan, class_name, section_name,
                jumlah_soal=jumlah_soal, acak=acak_opsi,
            )
            for k, potongan in zip(konsep, bahan, strict=True)
        ),
        return_exceptions=True,
    )

    soal: list[SoalKonsep] = []
    gagal: list[str] = []
    for k, keluaran in zip(konsep, hasil, strict=True):
        if isinstance(keluaran, BaseException):
            logger.warning("Penyusunan soal gagal untuk konsep '{}': {}", k.nama, keluaran)
            gagal.append(k.nama)
            continue
        if not keluaran:
            logger.warning("Model tidak menghasilkan soal yang bisa dipakai untuk '{}'", k.nama)
            gagal.append(k.nama)
            continue
        soal.extend(keluaran)

    return BankSoal(
        content_id=content_id,
        section_id=section_id,
        soal=soal,
        konsep_gagal=gagal,
    )


__all__ = [
    "JUMLAH_OPSI",
    "MAKS_CHUNK_PER_KONSEP",
    "SOAL_PER_KONSEP_DEFAULT",
    "SOAL_PER_KONSEP_MAKS",
    "TIPE_SOAL",
    "BankSoal",
    "KonsepMasukan",
    "OpsiSoal",
    "SoalKonsep",
    "saring_chunk_konsep",
    "susun_bank_soal",
]
