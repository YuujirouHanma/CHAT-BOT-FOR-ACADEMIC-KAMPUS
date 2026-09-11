"""Deteksi berkas rangkap sebelum sebuah materi masuk ke index.

MASALAH YANG DIATASI
--------------------
Jalur unggah tidak pernah memeriksa apakah isi berkas sudah pernah diindeks.
Akibatnya satu berkas yang sama diunggah berulang kali — pada korpus contoh,
72 berkas hanya berisi 5 isi yang berbeda — dan setiap salinan menghasilkan
potongan yang identik. Karena penyaringan akhir hanya menyediakan lima slot
bukti, salinan-salinan itu saling berebut slot: rata-rata hanya 3,6 dari 5 slot
berisi potongan yang benar-benar berbeda. Sepertiga jatah bukti terbuang untuk
mengulang kalimat yang sama, dan jawaban kehilangan sudut pandang yang
sebenarnya tersedia di materi lain.

KEPUTUSAN 1 — HASH ISI BERKAS, BUKAN HASH ISI POTONGAN
------------------------------------------------------
Yang di-hash adalah byte berkasnya (SHA-256), bukan teks tiap potongan.

- Ia menangkap persis kasus yang terukur: salinan yang identik byte per byte.
- Ia menolak SEBELUM parsing, peringkasan gambar oleh LLM, dan embedding —
  tiga langkah termahal di seluruh pipeline. Hash potongan baru bisa dihitung
  SESUDAH ongkos itu dibayar, jadi ia menghemat ruang index tetapi tidak
  menghemat sepeser pun biaya pemrosesan.
- Hash potongan menuntut daftar hash lintas dokumen yang hidup di lapisan
  indexing/penyimpanan, bukan di ingestion.

Sisanya yang belum tertangkap: dua berkas dengan teks sama tetapi byte berbeda
(mis. PDF yang disimpan ulang sehingga metadatanya bergeser). Itu perlu
penyaringan setingkat potongan di lapisan index, dan sengaja tidak dikerjakan
di sini.

KEPUTUSAN 2 — CAKUPAN DUPLIKAT ADALAH SATU DIREKTORI MATERI
-----------------------------------------------------------
Berkas yang sama diunggah untuk mata kuliah atau minggu BERBEDA BUKAN duplikat.
Slide yang dipakai ulang di dua minggu memang harus terindeks dua kali: kueri
mahasiswa disaring per `course_id`/`week`, jadi menolak unggahan kedua membuat
minggu itu tidak punya materi sama sekali — bukan menghemat, tetapi menghapus
jawaban yang seharusnya ada.

Pembedanya tidak perlu dicari jauh: unggahan sudah dipisah secara fisik ke
`storage/{tenant_id}/{content_id}/`, dan `content_id` justru berbentuk
"sbd-minggu-2" — satu tenant, satu mata kuliah, satu minggu. Jadi cakupan
duplikat = direktori tempat berkas itu berada. Tenant lain, mata kuliah lain,
atau minggu lain otomatis berada di direktori lain dan tidak pernah saling
dianggap rangkap, tanpa perlu menaruh negara-bagian baru di mana pun.

BATASNYA: unggahan TANPA `content_id` semuanya jatuh ke satu direktori
`_uploads/`, sehingga mata kuliahnya tidak lagi terbaca dari path dan dua
mata kuliah berbeda akan dianggap satu cakupan. Itu diterima dengan sadar —
pesan galatnya menyebutkan jalan keluarnya (kirim `content_id`), sehingga
kasus ini gagal dengan terang, bukan diam-diam.

KEPUTUSAN 3 — YANG PALING TUA YANG BERTAHAN
--------------------------------------------
Bila dua berkas beradu, yang dianggap sah adalah yang lebih dulu ada
(mtime, lalu nama sebagai pemutus seri). Tanpa aturan yang tegas, pengindeksan
massal atas 67 salinan akan membuat setiap salinan melihat salinan lain sebagai
duplikat — dan TIDAK SATU PUN terindeks. Aturan "yang tertua bertahan" juga
sudah benar untuk jalur unggah: berkas yang baru ditulis selalu yang termuda,
jadi salinan barulah yang ditolak, bukan materi yang sudah terlanjur diindeks.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils.logger import logger

# Dibaca per blok, bukan sekaligus: berkas unggahan boleh sampai 100 MB dan
# membacanya utuh ke memori membuat beberapa unggahan bersamaan cukup untuk
# menjatuhkan proses.
_HASH_BLOCK = 1024 * 1024


def hash_file(path: Path) -> str:
    """SHA-256 isi sebuah berkas, dibaca bertahap."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            blok = f.read(_HASH_BLOCK)
            if not blok:
                break
            h.update(blok)
    return h.hexdigest()


def _kunci_umur(path: Path) -> tuple[float, str]:
    """Kunci urut "siapa lebih dulu ada": waktu ubah, lalu nama sebagai pemutus seri.

    Nama ikut karena dua berkas dapat memiliki mtime yang persis sama — pada
    beberapa berkas sistem resolusinya hanya satu detik, dan penyalinan massal
    memang menghasilkan mtime kembar. Tanpa pemutus seri, urutannya bergantung
    pada urutan pembacaan direktori dan hasil pemeriksaan menjadi tidak tetap.
    """
    return (path.stat().st_mtime, path.name)


def find_duplicate(path: Path) -> Path | None:
    """Berkas lebih tua di direktori yang sama dengan isi persis sama, bila ada.

    Mengembalikan None bila `path` justru yang tertua — pemanggil boleh
    melanjutkan, karena berkas inilah salinan yang sah.

    Kandidat disaring lebih dulu berdasarkan UKURAN: ukuran berbeda mustahil
    berisi byte yang sama, dan `stat()` jauh lebih murah daripada membaca
    seluruh berkas. Tanpa penyaringan itu, satu unggahan ke direktori berisi
    ratusan materi harus membaca semuanya dari disk.
    """
    try:
        ukuran = path.stat().st_size
        kunci_diri = _kunci_umur(path)
    except OSError:
        return None

    # Berkas kosong ditangani `validate_file` sebagai galat tersendiri; di sini
    # ia hanya akan membuat setiap berkas kosong tampak rangkap satu sama lain.
    if ukuran == 0:
        return None

    kandidat: list[Path] = []
    try:
        isi_direktori = list(path.parent.iterdir())
    except OSError:
        return None

    for lain in isi_direktori:
        if lain.name == path.name:
            continue
        try:
            if not lain.is_file() or lain.stat().st_size != ukuran:
                continue
            if _kunci_umur(lain) >= kunci_diri:
                continue
        except OSError:
            # Berkas yang lenyap atau tak terbaca di tengah pemindaian bukan
            # alasan menggagalkan unggahan — lewati saja.
            continue
        kandidat.append(lain)

    if not kandidat:
        return None

    sidik = hash_file(path)
    for lain in sorted(kandidat, key=_kunci_umur):
        try:
            if hash_file(lain) == sidik:
                logger.warning(
                    "Berkas '{}' identik dengan '{}' yang sudah ada di {}",
                    path.name, lain.name, path.parent.name,
                )
                return lain
        except OSError:
            continue
    return None
