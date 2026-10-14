"""Uji beban: berapa banyak mahasiswa yang benar-benar dapat memakai layanan serentak.

    python -m scripts.load_test --lapis kuota --kunci <KUNCI>
    python -m scripts.load_test --lapis web   --kunci <KUNCI> --serentak 1,4,8,16,32
    python -m scripts.load_test --lapis penuh --kunci <KUNCI> --berbayar

MENGAPA ADA
-----------
Seluruh uji otomatis yang ada memanggil satu permintaan pada satu waktu. Itu
membuktikan setiap jalur BENAR, tetapi tidak menjawab pertanyaan yang diajukan
saat ujicoba kampus: kalau satu kelas memakainya bersamaan, yang jebol lebih
dulu apa. Pertanyaan itu hanya terjawab dengan mengirim permintaan serentak dan
melihat angkanya.

TIGA LAPIS, SENGAJA DIPISAH
---------------------------
`web`   - endpoint katalog: autentikasi, kuota, dan pembacaan penyimpanan,
          TANPA memanggil model. Gratis, jadi boleh diulang sesering perlu.
`kuota` - mengirim ledakan permintaan sampai ditolak, lalu melaporkan lapis
          mana yang menolak (per-IP atau per-kampus) dan berapa yang lolos.
          Inilah batas yang sesungguhnya dirasakan pengguna.
`penuh` - `/ask` sungguhan. BERBAYAR: satu pertanyaan memanggil model tiga kali
          (penguraian, jawaban, saran lanjutan), jadi lapis ini menuntut
          `--berbayar` secara eksplisit dan jumlah permintaannya dibatasi.

Dipisah karena mencampurnya menghasilkan angka yang tidak dapat ditafsirkan:
latensi `/ask` didominasi penyedia model di luar kendali kita, sehingga
pelambatan di lapisan kita sendiri tenggelam di dalamnya.

CARA MEMBACA ANGKANYA
---------------------
Yang dilaporkan p50/p95/p99, bukan rata-rata. Rata-rata menyembunyikan ekor:
dengan 40 permintaan, satu yang tertahan 8 detik hampir tidak menggeser
rata-rata, padahal justru mahasiswa itu yang mengeluh.

Angka dari mesin pengembangan TIDAK dapat dipakai menaksir produksi apa adanya:
di sini model embedding dan reranker berjalan lokal, sedangkan compose produksi
memakai OpenRouter untuk keduanya dan Qdrant server, bukan mode lokal. Yang
berpindah utuh ke produksi adalah hasil lapis `kuota` - batasnya ditentukan
konfigurasi, bukan perangkat keras.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

# Endpoint termurah yang masih menuntut autentikasi penuh: ia melewati seluruh
# lapis yang dibagi semua permintaan (kuota IP, verifikasi kunci, kuota kampus)
# tanpa menyentuh model sama sekali.
JALUR_MURAH = "/catalog/courses"

# Pertanyaan diputar bergilir supaya tidak satu pun terlayani dari cache
# embedding kueri (lihat _query_cache di src/indexing/embedder.py). Memakai satu
# pertanyaan yang sama akan membuat uji ini mengukur cache, bukan layanannya.
PERTANYAAN = [
    "Apa itu normalisasi basis data?",
    "Jelaskan perbedaan stack dan queue.",
    "Bagaimana cara kerja indeks B-tree?",
    "Apa fungsi primary key?",
    "Kapan sebaiknya memakai denormalisasi?",
    "Terangkan konsep transaksi ACID.",
]


@dataclass
class Hasil:
    """Hasil satu tingkat keserentakan."""

    serentak: int
    latensi: list[float] = field(default_factory=list)
    kode: dict[int, int] = field(default_factory=dict)
    galat: dict[str, int] = field(default_factory=dict)
    durasi: float = 0.0

    @property
    def sukses(self) -> int:
        return sum(n for k, n in self.kode.items() if 200 <= k < 300)

    @property
    def total(self) -> int:
        return sum(self.kode.values()) + sum(self.galat.values())

    @property
    def tak_wajar(self) -> dict[int, int]:
        return {k: v for k, v in self.kode.items() if not 200 <= k < 300}

    def persentil(self, p: float) -> float:
        if not self.latensi:
            return 0.0
        urut = sorted(self.latensi)
        # Indeks dibatasi supaya p99 pada 10 sampel tidak keluar dari daftar.
        i = min(len(urut) - 1, int(round(p / 100 * len(urut))) - 1)
        return urut[max(0, i)]


async def _satu(
    klien: httpx.AsyncClient,
    metode: str,
    jalur: str,
    badan: dict | None,
    hasil: Hasil,
) -> httpx.Response | None:
    """Kirim satu permintaan dan bukukan hasilnya. Tidak pernah melempar.

    Galat jaringan dibukukan sebagai galat, bukan dibiarkan menggagalkan seluruh
    tingkat: justru pada keserentakan tinggi itulah ia muncul, dan kehilangan
    seluruh pengukuran karena satu koneksi tertolak membuang semua datanya.
    """
    t0 = time.perf_counter()
    try:
        if metode == "GET":
            r = await klien.get(jalur)
        else:
            r = await klien.post(jalur, json=badan or {})
    except Exception as exc:  # noqa: BLE001 - kegagalan jaringan adalah datanya
        nama = type(exc).__name__
        hasil.galat[nama] = hasil.galat.get(nama, 0) + 1
        return None

    hasil.latensi.append((time.perf_counter() - t0) * 1000)
    hasil.kode[r.status_code] = hasil.kode.get(r.status_code, 0) + 1
    return r


async def _tingkat(
    basis: str,
    kunci: str,
    metode: str,
    jalur: str,
    badan_dari: Callable[[int], dict | None],
    serentak: int,
    jumlah: int,
    batas_waktu: float,
) -> Hasil:
    """Jalankan `jumlah` permintaan dengan paling banyak `serentak` berjalan."""
    hasil = Hasil(serentak=serentak)
    gerbang = asyncio.Semaphore(serentak)

    limits = httpx.Limits(
        max_connections=serentak + 4, max_keepalive_connections=serentak + 4,
    )
    async with httpx.AsyncClient(
        base_url=basis,
        headers={"X-API-Key": kunci},
        timeout=batas_waktu,
        limits=limits,
    ) as klien:

        async def terbatas(i: int) -> None:
            async with gerbang:
                await _satu(klien, metode, jalur, badan_dari(i), hasil)

        t0 = time.perf_counter()
        await asyncio.gather(*(terbatas(i) for i in range(jumlah)))
        hasil.durasi = time.perf_counter() - t0

    return hasil


def _cetak_tabel(hasil: list[Hasil]) -> None:
    kepala = (f"  {'serentak':>8}  {'kirim':>6}  {'sukses':>6}  {'req/s':>7}"
              f"  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  catatan")
    print(kepala)
    print("  " + "-" * 86)
    for h in hasil:
        rps = h.total / h.durasi if h.durasi else 0.0
        catatan = ", ".join(
            [f"{k}x{n}" for k, n in sorted(h.tak_wajar.items())]
            + [f"{k}x{n}" for k, n in sorted(h.galat.items())]
        )
        print(f"  {h.serentak:>8}  {h.total:>6}  {h.sukses:>6}  {rps:>7.1f}"
              f"  {h.persentil(50):>8.0f}  {h.persentil(95):>8.0f}"
              f"  {h.persentil(99):>8.0f}  {catatan or '-'}")


# Balasan 429 SENGAJA tidak menyebutkan lapis penolaknya kepada pemanggil:
# memberitahunya hanya menolong penyerang menakar pertahanan. Lapisnya tercatat
# di jejak audit. Untuk alat ini, pesan yang berbeda sudah cukup membedakan,
# dan pemetaan di sini menerjemahkannya menjadi setelan yang harus diubah.
_LAPIS_DARI_PESAN = {
    "Terlalu banyak permintaan": "per-IP (RATE_LIMIT_IP_PER_MINUTE)",
    "Kuota pemakaian terlampaui": "per-kampus (quota.requests_per_minute)",
}


def _lingkup_penolakan(r: httpx.Response) -> str:
    """Lapis mana yang menolak permintaan ini, dalam istilah setelannya."""
    try:
        badan: Any = r.json()
    except ValueError:
        badan = {}
    if isinstance(badan, dict):
        for jalan in (badan.get("scope"), badan.get("lingkup")):
            if jalan:
                return str(jalan)
        detail = badan.get("detail")
        if isinstance(detail, dict):
            for jalan in (detail.get("scope"), detail.get("lingkup")):
                if jalan:
                    return str(jalan)
            pesan = str(detail.get("message") or detail.get("pesan") or "")
            if pesan:
                return _LAPIS_DARI_PESAN.get(pesan, pesan[:70])
        if detail:
            return _LAPIS_DARI_PESAN.get(str(detail), str(detail)[:70])
    return f"retry-after={r.headers.get('retry-after', '?')}"


async def _pastikan_siap(a: argparse.Namespace) -> str | None:
    """Pastikan satu permintaan tunggal berhasil sebelum mengirim ribuan.

    Ada karena versi pertama alat ini pernah mengirim 200 permintaan yang
    SELURUHNYA dijawab 404 - jalurnya salah - lalu melaporkan "belum tersentuh
    batas". Laporan itu menenangkan dan sepenuhnya salah. Uji beban yang tidak
    memeriksa bahwa ia benar-benar memanggil sesuatu lebih buruk daripada tidak
    ada uji beban.

    Mengembalikan None bila siap, atau keterangan masalahnya.
    """
    async with httpx.AsyncClient(
        base_url=a.basis, headers={"X-API-Key": a.kunci}, timeout=a.batas_waktu,
    ) as klien:
        try:
            r = await klien.get(JALUR_MURAH)
        except Exception as exc:  # noqa: BLE001
            return f"tidak dapat menghubungi {a.basis}: {type(exc).__name__}: {exc}"

    if r.status_code == 429:
        return ("sudah kena batas laju sebelum uji dimulai; tunggu satu menit "
                "lalu ulangi")
    if not 200 <= r.status_code < 300:
        return (f"GET {JALUR_MURAH} menjawab {r.status_code}, bukan 2xx "
                f"({r.text[:160]})")
    return None


async def lapis_web(a: argparse.Namespace) -> int:
    """Autentikasi + kuota + pembacaan penyimpanan, tanpa model."""
    print(f"\nLapis WEB - GET {JALUR_MURAH} (tanpa panggilan model)")
    print("Yang diukur: autentikasi, pemeriksaan kuota, pembacaan registri "
          "kampus, dan\nkemampuan event loop melayani permintaan berbarengan.\n")

    hasil: list[Hasil] = []
    for s in a.serentak:
        h = await _tingkat(a.basis, a.kunci, "GET", JALUR_MURAH,
                           lambda i: None, s, a.per_tingkat, a.batas_waktu)
        hasil.append(h)
        if h.kode.get(429):
            _cetak_tabel(hasil)
            print(f"\n  [!] 429 muncul pada serentak={s}, jadi yang terukur "
                  f"sejak titik ini\n      adalah pembatas laju, bukan kemampuan "
                  f"layanan. Untuk mengukur\n      kemampuannya, jalankan "
                  f"ulang server dengan RATE_LIMIT_ENABLED=false.")
            return 0
    _cetak_tabel(hasil)
    return 0


async def lapis_kuota(a: argparse.Namespace) -> int:
    """Cari batas yang sesungguhnya dirasakan pengguna, dan lapis mana penolaknya."""
    print(f"\nLapis KUOTA - ledakan GET {JALUR_MURAH} sampai ditolak")
    print("Yang dijawab: berapa permintaan yang lolos dalam satu menit, dan "
          "oleh lapis\nmana sisanya ditolak - per alamat IP atau per kampus.\n")

    hasil = Hasil(serentak=a.ledakan)
    alasan: dict[str, int] = {}

    async with httpx.AsyncClient(
        base_url=a.basis,
        headers={"X-API-Key": a.kunci},
        timeout=a.batas_waktu,
        limits=httpx.Limits(max_connections=a.ledakan + 4),
    ) as klien:
        t0 = time.perf_counter()

        async def satu(i: int) -> None:
            r = await _satu(klien, "GET", JALUR_MURAH, None, hasil)
            if r is not None and r.status_code == 429:
                lingkup = _lingkup_penolakan(r)
                alasan[lingkup] = alasan.get(lingkup, 0) + 1

        await asyncio.gather(*(satu(i) for i in range(a.ledakan)))
        hasil.durasi = time.perf_counter() - t0

    print(f"  dikirim        : {hasil.total}")
    print(f"  diterima (2xx) : {hasil.sukses}")
    print(f"  ditolak (429)  : {hasil.kode.get(429, 0)}")
    lain = {k: v for k, v in hasil.tak_wajar.items() if k != 429}
    print(f"  kode lain      : {lain or '-'}")
    print(f"  galat jaringan : {hasil.galat or '-'}")
    print(f"  lama           : {hasil.durasi:.2f}s")
    if alasan:
        print(f"  alasan 429     : {alasan}")

    if not hasil.kode.get(429):
        print(f"\n  Belum tersentuh batas. Naikkan --ledakan di atas {a.ledakan}.")
    else:
        print(f"\n  Artinya: dalam satu menit, seluruh pemanggil dari alamat ini "
              f"hanya dapat\n  dilayani {hasil.sukses} permintaan. Bandingkan "
              f"dengan jumlah mahasiswa satu kelas\n  dikali berapa kali mereka "
              f"bertanya per menit.")
    return 0


async def lapis_penuh(a: argparse.Namespace) -> int:
    """`/ask` sungguhan. Berbayar, jadi jumlahnya kecil dan harus diminta eksplisit."""
    if not a.berbayar:
        print("\nLapis PENUH memanggil model berbayar: satu pertanyaan = TIGA "
              "panggilan\n(penguraian kueri, jawaban, saran lanjutan). Tambahkan "
              "--berbayar bila\nmemang ingin membelanjakannya.\n")
        return 2

    jumlah = min(a.per_tingkat, a.batas_berbayar)
    print(f"\nLapis PENUH - POST /ask, {jumlah} pertanyaan per tingkat (BERBAYAR)")
    print(f"Perkiraan panggilan model: {jumlah * len(a.serentak) * 3}. "
          f"Biaya sebenarnya dapat dilihat\nsetelahnya dengan: "
          f"python -m scripts.usage_report --per tujuan\n")

    hasil: list[Hasil] = []
    for s in a.serentak:
        h = await _tingkat(
            a.basis, a.kunci, "POST", "/ask",
            lambda i: {"question": PERTANYAAN[i % len(PERTANYAAN)]},
            s, jumlah, a.batas_waktu,
        )
        hasil.append(h)
    _cetak_tabel(hasil)
    print("\n  Ingat: di mesin ini embedding dan reranker berjalan LOKAL, "
          "sedangkan compose\n  produksi memakai OpenRouter untuk keduanya. "
          "Latensi di sini bukan taksiran\n  latensi produksi.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lapis", default="kuota", choices=["web", "kuota", "penuh"])
    p.add_argument("--basis", default="http://127.0.0.1:8000")
    p.add_argument("--kunci", required=True, help="kunci API kampus")
    p.add_argument("--serentak", default="1,2,4,8,16,32",
                   help="daftar tingkat keserentakan, dipisah koma")
    p.add_argument("--per-tingkat", type=int, default=40,
                   help="jumlah permintaan di setiap tingkat")
    p.add_argument("--ledakan", type=int, default=200,
                   help="lapis kuota: berapa permintaan dikirim sekaligus")
    p.add_argument("--batas-waktu", type=float, default=120.0)
    p.add_argument("--berbayar", action="store_true",
                   help="izinkan lapis penuh memanggil model berbayar")
    p.add_argument("--batas-berbayar", type=int, default=12,
                   help="pagar atas jumlah permintaan berbayar per tingkat")
    a = p.parse_args(argv)

    try:
        a.serentak = [int(x) for x in a.serentak.split(",") if x.strip()]
    except ValueError:
        print("--serentak harus berupa angka dipisah koma, mis. 1,4,8")
        return 2
    if not a.serentak:
        print("--serentak kosong")
        return 2

    jalan = {"web": lapis_web, "kuota": lapis_kuota, "penuh": lapis_penuh}[a.lapis]

    async def jalankan() -> int:
        masalah = await _pastikan_siap(a)
        if masalah:
            print(f"Tidak dapat memulai uji: {masalah}")
            return 2
        return await jalan(a)

    return asyncio.run(jalankan())


if __name__ == "__main__":
    raise SystemExit(main())
