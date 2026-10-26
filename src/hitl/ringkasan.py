"""Ringkasan riwayat chat mahasiswa, untuk dibaca dosen.

APA YANG SEBENARNYA KURANG
--------------------------
`teach/chat_history.php` sudah menampilkan riwayat chat apa adanya: satu baris
per interaksi, terbaru di atas. Itu berguna untuk memeriksa SATU jawaban, dan
tidak berguna sama sekali untuk pertanyaan yang sebenarnya dimiliki dosen —
"kelas saya tidak paham bagian mana?" Menjawabnya dari tampilan mentah menuntut
membaca ratusan baris dan menghitung sendiri.

Masukan dari workshop menyebutnya langsung: riwayat chat direkam dari sisi
mahasiswa dan diamati dosen. Yang terpasang baru bagian "direkam".

ANGKANYA DIHITUNG DI KODE, NARASINYA DITULIS MODEL
--------------------------------------------------
Pembagian ini disengaja, dan sama dengan yang dipakai gerbang relevansi: apa
pun yang bisa dihitung, dihitung — tidak diserahkan ke model. Model tidak bisa
menghitung dengan benar dari ratusan baris, dan kalaupun hasilnya kebetulan
tepat, tidak ada cara memeriksanya.

Jadi modul ini menghitung sendiri: jumlah interaksi, jumlah mahasiswa, topik
yang paling sering muncul, berkas yang paling sering jadi sumber, pertanyaan
yang ditanyakan lebih dari satu mahasiswa, dan — yang paling berharga —
pertanyaan yang DITOLAK karena tidak terjawab materi. Model hanya diminta
menuliskan apa artinya bagi pengajaran minggu depan.

Akibatnya angka di balasan selalu dapat dipertanggungjawabkan: dosen yang
meragukan "12 pertanyaan tidak terjawab" bisa membuka 12 contohnya, karena
contohnya ikut dikirim.

PERTANYAAN YANG DITOLAK ADALAH SINYAL TERBAIK
---------------------------------------------
Sejak gerbang relevansi dipasang, pertanyaan yang materinya tidak mencakup
dijawab dengan penolakan baku, bukan karangan. Itu berarti log sekarang memuat
daftar yang sebelumnya tidak mungkin disusun: hal-hal yang DICARI mahasiswa dan
TIDAK ADA di materi yang diunggah. Untuk dosen, itu daftar kerja — bukan keluhan
tentang chatbot.

Penolakan dikenali dari penanda kalimatnya, bukan dari `sources` yang kosong:
jawaban navigasi guided ("pilih mata kuliah dulu") juga tidak punya sumber, dan
menghitungnya sebagai kesenjangan materi akan melaporkan angka yang salah.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Penanda bahwa sebuah jawaban adalah PENOLAKAN karena materi tidak mencakup.
# Diambil dari kalimat yang benar-benar disusun `_pesan_tak_terjawab()` dan
# `_pesan_tanpa_materi()` di src/pipeline.py. Kalau kalimat di sana diubah,
# daftar ini harus ikut — karena itu diuji di tests/test_ringkasan.py dengan
# memanggil pembuat pesannya langsung, bukan dengan teks yang disalin tangan.
_PENANDA_TAK_TERJAWAB: tuple[str, ...] = (
    "tidak ada bagian materi yang menjawab",
    "tidak mencakup informasi tentang",
    "di minggu yang sedang kamu aktifkan",
    "belum ada materi terindeks",
)

# Kata yang terlalu umum untuk membedakan dua pertanyaan. Tanpa ini, "apa itu
# X" dan "apa itu Y" dikelompokkan sebagai pertanyaan yang sama karena berbagi
# tiga kata dari empat.
_KATA_UMUM: frozenset[str] = frozenset({
    "apa", "itu", "yang", "untuk", "dari", "dengan", "pada", "dan", "atau",
    "bagaimana", "mengapa", "kenapa", "jelaskan", "sebutkan", "adalah",
    "tolong", "saya", "bisa", "dalam", "ada", "akan", "ini", "aku",
    "jadi", "kalau", "tentang", "contoh", "maksud", "beda", "cara",
})

# Panjang maksimum kutipan pertanyaan yang ikut di balasan. Pertanyaan mahasiswa
# bisa panjang; yang perlu dilihat dosen inti kalimatnya.
_MAKS_KUTIPAN = 160

# Berapa contoh yang ikut per kategori. Cukup untuk diperiksa, tidak sampai
# membuat balasannya sebesar riwayat mentahnya.
_MAKS_CONTOH = 12


def _tak_terjawab(jawaban: str, pertanyaan: str = "") -> bool:
    """Apakah ini KESENJANGAN MATERI — hal yang dicari mahasiswa dan tidak ada.

    Permintaan ringkasan yang ditolak TIDAK dihitung, walaupun kalimat
    penolakannya sama. Alasannya: ringkasan tidak pernah bisa menjadi
    kesenjangan materi — yang diminta justru materi yang sudah dipilih. Kalau ia
    ditolak, itu kegagalan sistem, bukan kekurangan bahan ajar.

    Terukur saat dicoba pada log sungguhan: "ringkas materi minggu 1" dan
    "ringkas materi kuliah ini" muncul di daftar "tidak ada di materi yang
    diunggah" — keduanya penolakan dari perilaku LAMA, sebelum pengecualian
    ikhtisar dipasang. Dosen yang membaca daftar itu akan menyimpulkan
    mahasiswa mencari sesuatu yang tidak ia unggah, padahal materinya ada dan
    yang salah adalah sistemnya.

    Dikenali dengan pengenal yang SAMA seperti gerbang relevansi
    (`guided.minta_ikhtisar`), bukan daftar kata kedua yang akan berbeda
    pelan-pelan dari yang pertama.
    """
    teks = (jawaban or "").lower()
    if not any(p in teks for p in _PENANDA_TAK_TERJAWAB):
        return False

    from src import guided

    return not guided.minta_ikhtisar(pertanyaan or "")


def _kunci_pertanyaan(teks: str) -> str:
    """Kunci pengelompokan untuk pertanyaan yang berbeda kata tetapi sama isi.

    "apa itu primary key" dan "primary key itu apa" harus dihitung sebagai satu
    pertanyaan yang ditanyakan dua kali, karena bagi dosen keduanya menunjukkan
    kebingungan yang sama. Kata kunci diurutkan, jadi urutan kalimat tidak
    berpengaruh; kata umum dibuang supaya yang tersisa benar-benar isinya.

    Mengembalikan "" bila tidak ada kata bermakna yang tersisa — pemanggil
    melewatinya, karena mengelompokkan seluruh pertanyaan pendek menjadi satu
    kelompok kosong akan melaporkan "ditanyakan 40 kali" untuk sesuatu yang
    tidak pernah ditanyakan.
    """
    kata = re.findall(r"[a-z0-9]+", (teks or "").lower())
    bermakna = sorted({k for k in kata if len(k) >= 4 and k not in _KATA_UMUM})
    return " ".join(bermakna)


def _potong(teks: str) -> str:
    bersih = re.sub(r"\s+", " ", (teks or "").strip())
    return bersih if len(bersih) <= _MAKS_KUTIPAN else bersih[:_MAKS_KUTIPAN] + "…"


@dataclass
class FaktaRiwayat:
    """Semua yang bisa diketahui dari log TANPA memanggil model."""

    total: int = 0
    mahasiswa_unik: int = 0
    sejak: str | None = None
    hingga: str | None = None
    tak_terjawab: int = 0
    topik_teratas: list[tuple[str, int]] = field(default_factory=list)
    materi_teratas: list[tuple[str, int]] = field(default_factory=list)
    tipe_pertanyaan: list[tuple[str, int]] = field(default_factory=list)
    contoh_tak_terjawab: list[str] = field(default_factory=list)
    pertanyaan_berulang: list[tuple[str, int]] = field(default_factory=list)

    @property
    def persen_tak_terjawab(self) -> float:
        return (self.tak_terjawab / self.total * 100) if self.total else 0.0


def hitung_fakta(records: list[dict[str, Any]]) -> FaktaRiwayat:
    """Ringkas log percakapan menjadi angka, tanpa satu pun panggilan model.

    `records` adalah keluaran `read_conversation_logs()` — sudah tersaring per
    kampus dan per kelas oleh pemanggil. Fungsi ini tidak menyaring apa pun
    lagi: menyaring dua kali adalah cara paling mudah melaporkan angka satu
    kelas sebagai angka kelas lain.
    """
    fakta = FaktaRiwayat(total=len(records))
    if not records:
        return fakta

    stempel = sorted(r.get("timestamp") or "" for r in records)
    stempel = [s for s in stempel if s]
    if stempel:
        fakta.sejak, fakta.hingga = stempel[0], stempel[-1]

    # Mahasiswa dihitung dari user_id, bukan student_name: namanya teks bebas
    # yang bisa diganti sendiri mahasiswa, jadi menghitungnya akan melaporkan
    # satu orang yang mengganti nama dua kali sebagai dua mahasiswa.
    fakta.mahasiswa_unik = len({
        r.get("user_id") for r in records if r.get("user_id") is not None
    })

    topik: Counter[str] = Counter()
    materi: Counter[str] = Counter()
    tipe: Counter[str] = Counter()
    berulang: dict[str, tuple[str, set[Any], int]] = {}

    for r in records:
        dq = r.get("decomposition") or {}
        if t := (dq.get("topik_utama") or "").strip():
            topik[t] += 1
        if tp := (dq.get("tipe_pertanyaan") or "").strip():
            tipe[tp] += 1

        # Satu berkas dihitung sekali per interaksi walaupun tiga potongannya
        # terpakai: yang diukur "berapa pertanyaan bersandar pada berkas ini",
        # bukan berapa potongan terambil.
        for nama in {
            (s or {}).get("source_file")
            for s in (r.get("sources") or [])
            if (s or {}).get("source_file")
        }:
            materi[nama] += 1

        pertanyaan = r.get("question") or ""
        if _tak_terjawab(r.get("answer") or "", pertanyaan):
            fakta.tak_terjawab += 1
            if len(fakta.contoh_tak_terjawab) < _MAKS_CONTOH:
                fakta.contoh_tak_terjawab.append(_potong(pertanyaan))

        if kunci := _kunci_pertanyaan(pertanyaan):
            contoh, siapa, jumlah = berulang.get(kunci, (_potong(pertanyaan), set(), 0))
            if r.get("user_id") is not None:
                siapa.add(r["user_id"])
            berulang[kunci] = (contoh, siapa, jumlah + 1)

    fakta.topik_teratas = topik.most_common(10)
    fakta.materi_teratas = materi.most_common(10)
    fakta.tipe_pertanyaan = tipe.most_common()

    # Hanya yang ditanyakan oleh LEBIH DARI SATU mahasiswa. Satu mahasiswa yang
    # menanyakan hal yang sama lima kali sedang kesulitan sendiri; dua mahasiswa
    # yang menanyakan hal sama menunjukkan materinya yang kurang jelas — dan
    # hanya yang kedua dapat ditindaklanjuti dosen di depan kelas.
    fakta.pertanyaan_berulang = sorted(
        ((contoh, len(siapa)) for contoh, siapa, _ in berulang.values() if len(siapa) >= 2),
        key=lambda x: x[1],
        reverse=True,
    )[:_MAKS_CONTOH]

    return fakta


def susun_bahan(fakta: FaktaRiwayat) -> str:
    """Bahan ringkas untuk model: angka yang sudah dihitung, bukan log mentah.

    Mengirim ratusan baris percakapan utuh akan menghabiskan jendela konteks dan
    membuat biayanya naik sebanding jumlah chat, padahal yang perlu dibaca model
    hanya hasil hitungannya. Pada 200 interaksi, bahan ini kira-kira 1-2 kB.
    """
    baris = [
        f"Jumlah interaksi: {fakta.total}",
        f"Jumlah mahasiswa: {fakta.mahasiswa_unik}",
        f"Rentang waktu: {fakta.sejak or '-'} sampai {fakta.hingga or '-'}",
        f"Pertanyaan yang TIDAK terjawab materi: {fakta.tak_terjawab}"
        f" ({fakta.persen_tak_terjawab:.0f}%)",
    ]

    if fakta.topik_teratas:
        baris.append(
            "Topik paling sering ditanyakan: "
            + ", ".join(f"{t} ({n}x)" for t, n in fakta.topik_teratas)
        )
    if fakta.tipe_pertanyaan:
        baris.append(
            "Jenis pertanyaan: "
            + ", ".join(f"{t} ({n})" for t, n in fakta.tipe_pertanyaan)
        )
    if fakta.materi_teratas:
        baris.append(
            "Materi paling sering dipakai menjawab: "
            + ", ".join(f"{m} ({n}x)" for m, n in fakta.materi_teratas)
        )
    if fakta.pertanyaan_berulang:
        baris.append("Ditanyakan beberapa mahasiswa berbeda:")
        baris += [
            f"  - \"{t}\" — {n} mahasiswa" for t, n in fakta.pertanyaan_berulang
        ]
    if fakta.contoh_tak_terjawab:
        baris.append("Contoh pertanyaan yang materinya tidak mencakup:")
        baris += [f"  - \"{t}\"" for t in fakta.contoh_tak_terjawab]

    return "\n".join(baris)
