"""Cap `tenant_id` pada titik index warisan yang belum punya pemilik.

    python -m scripts.migrate_tenant_id <tenant_id>            # uji-kering
    python -m scripts.migrate_tenant_id <tenant_id> --dry-run  # sama, eksplisit
    python -m scripts.migrate_tenant_id <tenant_id> --apply    # baru menulis

MENGAPA SKRIP INI ADA
---------------------
Titik yang diindeks sebelum sistem menjadi multi-tenant tidak membawa field
`tenant_id` sama sekali. Sejak `QdrantStore._build_filter()` SELALU menyisipkan
kondisi `tenant_id` — dan itu perbaikan keamanan yang tidak boleh dilonggarkan —
titik tanpa pemilik tidak lagi cocok dengan kueri mana pun. Mereka tetap memakan
disk tetapi tidak dapat dijangkau siapa pun. Pada instalasi contoh, 34 dari 44
titik berada dalam keadaan itu: korpus efektif tiap tenant menyusut menjadi lima
potongan.

Jalan keluarnya adalah memberi mereka pemilik, bukan melemahkan filternya.

TIGA ATURAN YANG DITEGAKKAN DI SINI
-----------------------------------
1. TENANT SASARAN HARUS DISEBUT. Tidak ada nilai bawaan dan tidak ada tebakan
   dari isi payload. Menebak salah berarti memindahkan materi satu kampus ke
   kampus lain — kebocoran data yang justru sedang kita cegah. Tenant-nya juga
   harus sudah terdaftar; kalau belum, skrip berhenti dan menunjuk `tenantctl`.
2. UJI-KERING ADALAH BAWAAN. Tanpa `--apply` skrip hanya membaca dan melapor.
3. TITIK YANG SUDAH PUNYA `tenant_id` TIDAK DISENTUH sama sekali — id-nya
   dikumpulkan saat pemindaian dan hanya id titik tanpa pemilik yang dikirim ke
   `set_payload`. Aman dijalankan berulang kali; jalan kedua tidak menemukan apa
   pun untuk diubah.

Skrip ini memakai `QdrantClient` mentah, bukan `QdrantStore`. Itu disengaja:
seluruh metode `QdrantStore` menyaring per tenant, sehingga justru tidak dapat
melihat titik yang belum punya tenant. Pengecualian ini berhenti di berkas ini
dan tidak mengubah apa pun pada penyaringan saat kueri.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

from src.config import settings  # noqa: E402
from src.storage import tenant_store  # noqa: E402
from src.storage.qdrant_store import TENANT_FIELD  # noqa: E402
from src.tenancy import is_valid_tenant_id  # noqa: E402

# Ukuran halaman pemindaian sekaligus potongan penulisan.
_BATCH = 256

# Field payload yang ikut ditampilkan agar operator dapat mengenali materi apa
# yang akan dicap sebelum menyetujui perubahannya.
_FIELD_LAPORAN = ["source_file", "content_id", "course_id", "week", TENANT_FIELD]


def _buat_klien() -> QdrantClient:
    """Klien Qdrant sesuai setelan — mode lokal (on-disk) atau server."""
    if settings.qdrant_mode == "server":
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return QdrantClient(path=settings.qdrant_path)


def tanpa_tenant(payload: dict) -> bool:
    """True bila titik belum punya pemilik.

    Field yang hilang, `None`, dan string kosong diperlakukan sama: ketiganya
    tidak akan pernah cocok dengan `MatchValue` pada `tenant_id`, jadi ketiganya
    sama-sama tak terjangkau dan sama-sama perlu dicap.
    """
    return not str(payload.get(TENANT_FIELD) or "").strip()


def pindai(
    client: QdrantClient, collection: str,
) -> tuple[list[Any], Counter, Counter]:
    """Baca seluruh koleksi sekali jalan.

    Mengembalikan (id titik tanpa tenant, sebaran tenant, rincian materi yatim).
    """
    yatim: list[Any] = []
    sebaran: Counter = Counter()
    rincian: Counter = Counter()
    offset: Any = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=_BATCH,
            offset=offset,
            with_payload=_FIELD_LAPORAN,
            with_vectors=False,
        )
        for titik in batch:
            payload = dict(titik.payload or {})
            if tanpa_tenant(payload):
                sebaran["(tanpa tenant_id)"] += 1
                yatim.append(titik.id)
                rincian[(
                    payload.get("content_id"),
                    payload.get("source_file"),
                    payload.get("course_id"),
                    payload.get("week"),
                )] += 1
            else:
                sebaran[str(payload[TENANT_FIELD])] += 1
        if offset is None:
            break

    return yatim, sebaran, rincian


def _laporkan(collection: str, sebaran: Counter, rincian: Counter, tenant_id: str) -> None:
    total = sum(sebaran.values())
    print(f"Koleksi '{collection}': {total} titik")
    for nama, jumlah in sorted(sebaran.items()):
        print(f"  {jumlah:6}  {nama}")

    if not rincian:
        return

    print(f"\nAkan dicap tenant_id='{tenant_id}':")
    for kunci, jumlah in sorted(rincian.items(), key=lambda item: [str(x) for x in item[0]]):
        content_id, source_file, course_id, week = kunci
        print(
            f"  {jumlah:6}  {source_file or '(tanpa nama berkas)'}"
            f"  content_id={content_id}  course_id={course_id}  minggu={week}"
        )


def cap_tenant(
    client: QdrantClient, collection: str, ids: list[Any], tenant_id: str,
) -> int:
    """Tulis `tenant_id` pada titik-titik yang disebutkan, per potongan.

    `set_payload` hanya menambahkan kunci yang diberikan — sisa payload (teks,
    nomor halaman, course_id) tetap utuh. Sasarannya adalah daftar id yang
    eksplisit, bukan sebuah filter: daftar itu disusun saat pemindaian dan tidak
    mungkin ikut menyentuh titik yang sudah bertenant.
    """
    ditulis = 0
    for awal in range(0, len(ids), _BATCH):
        potongan = ids[awal : awal + _BATCH]
        client.set_payload(
            collection_name=collection,
            payload={TENANT_FIELD: tenant_id},
            points=potongan,
            wait=True,
        )
        ditulis += len(potongan)
    return ditulis


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="migrate_tenant_id",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("tenant_id", help="tenant yang akan memiliki titik warisan")
    p.add_argument(
        "--apply", action="store_true",
        help="benar-benar menulis; tanpa ini skrip hanya melapor",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="hanya melapor (bawaan); ditolak bila digabung dengan --apply",
    )
    p.add_argument(
        "--collection", default=None,
        help=f"koleksi Qdrant (bawaan: {settings.qdrant_collection})",
    )
    args = p.parse_args(argv)

    if args.apply and args.dry_run:
        print("Gagal: --apply dan --dry-run saling bertentangan.", file=sys.stderr)
        return 2

    if not is_valid_tenant_id(args.tenant_id):
        print(f"Gagal: '{args.tenant_id}' bukan tenant_id yang sah.", file=sys.stderr)
        return 1

    # Tenant harus sudah terdaftar. Salah ketik satu huruf akan menjadikan
    # seluruh materi warisan milik tenant hantu yang tak seorang pun punya
    # kuncinya — tidak merusak apa pun, tetapi sama tak terjangkaunya dengan
    # keadaan sebelum migrasi, dan jauh lebih sulit dikenali.
    if tenant_store.get(args.tenant_id) is None:
        print(
            f"Gagal: tenant '{args.tenant_id}' belum terdaftar. Buat dulu dengan:\n"
            f"  python -m scripts.tenantctl create {args.tenant_id}",
            file=sys.stderr,
        )
        return 1

    collection = args.collection or settings.qdrant_collection
    client = _buat_klien()
    try:
        if not client.collection_exists(collection):
            print(f"Gagal: koleksi '{collection}' tidak ada.", file=sys.stderr)
            return 1

        yatim, sebaran, rincian = pindai(client, collection)
        _laporkan(collection, sebaran, rincian, args.tenant_id)

        if not yatim:
            print("\nTidak ada titik tanpa tenant_id. Tidak ada yang perlu diubah.")
            return 0

        if not args.apply:
            print(
                f"\nUJI-KERING: {len(yatim)} titik AKAN dicap tenant_id="
                f"'{args.tenant_id}'. Tidak ada yang diubah.\n"
                f"Jalankan ulang dengan --apply untuk benar-benar menulis."
            )
            return 0

        ditulis = cap_tenant(client, collection, yatim, args.tenant_id)
        print(f"\n{ditulis} titik dicap tenant_id='{args.tenant_id}'.")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
