"""Buang titik index yang tidak memiliki `tenant_id`.

    python -m scripts.purge_orphan_points            # uji-kering
    python -m scripts.purge_orphan_points --apply    # baru menghapus

MENGAPA SKRIP INI ADA
---------------------
Titik yang diindeks sebelum sistem menjadi multi-tenant tidak membawa
`tenant_id`. Karena `QdrantStore._build_filter()` selalu menyisipkan kondisi
tenant, titik itu tidak terjangkau kueri mana pun: ia memakan disk tanpa dapat
dibaca siapa pun.

Ada dua jalan keluar, dan keduanya sah untuk keadaan yang berbeda:

1. `scripts/migrate_tenant_id.py` — mencapkan pemilik pada titik yang sudah ada.
   Tepat bila titik itu masih layak dipakai apa adanya.
2. Skrip ini — membuangnya, SESUDAH materinya diindeks ulang dari berkas asli.
   Tepat bila titik lama dibuat oleh versi parser terdahulu, sehingga mencapnya
   berarti mengawetkan potongan yang sudah diketahui cacat: nomor halaman
   seluruh dokumen tercatat 1, dan indeks ulang tanpa pembersihan meninggalkan
   potongan kembar.

PENJAGAAN
---------
- Uji-kering adalah bawaan; penghapusan menuntut `--apply` yang eksplisit.
- Hanya titik TANPA `tenant_id` yang tersentuh. Titik bertuan tidak pernah
  masuk daftar hapus, bahkan bila isinya sama persis.
- Skrip menolak berjalan bila TIDAK ADA satu pun titik bertuan di koleksi itu.
  Keadaan demikian berarti index memang belum dimigrasi sama sekali, dan
  menghapus seluruh isinya hampir pasti bukan yang dimaksud.
- Materi asli tetap ada di `storage/`, sehingga penghapusan ini dapat dibatalkan
  dengan mengindeks ulang — bukan operasi yang kehilangan data sumber.

Seperti skrip migrasi, di sini dipakai `QdrantClient` mentah: seluruh metode
`QdrantStore` menyaring per tenant, sehingga justru tidak dapat melihat titik
yang belum punya tenant.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client import models as qm  # noqa: E402

from src.config import settings  # noqa: E402
from src.storage.qdrant_store import TENANT_FIELD  # noqa: E402

_BATCH = 256
_FIELD_LAPORAN = ["source_file", "content_id", "course_id", "week"]


def _client() -> QdrantClient:
    if settings.qdrant_mode == "local":
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _pindai(client: QdrantClient, koleksi: str) -> tuple[list[Any], Counter, int]:
    """Kumpulkan id titik yatim, ringkasan isinya, dan cacah titik bertuan."""
    yatim: list[Any] = []
    ringkas: Counter = Counter()
    bertuan = 0
    offset = None
    while True:
        titik, offset = client.scroll(
            collection_name=koleksi,
            limit=_BATCH,
            offset=offset,
            with_payload=_FIELD_LAPORAN + [TENANT_FIELD],
            with_vectors=False,
        )
        for t in titik:
            payload = t.payload or {}
            if payload.get(TENANT_FIELD):
                bertuan += 1
                continue
            yatim.append(t.id)
            ringkas[(
                payload.get("course_id"),
                payload.get("content_id"),
                payload.get("week"),
                payload.get("source_file"),
            )] += 1
        if offset is None:
            break
    return yatim, ringkas, bertuan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--apply", action="store_true",
        help="benar-benar menghapus; tanpa ini skrip hanya melapor",
    )
    p.add_argument(
        "--collection", default=settings.qdrant_collection,
        help="koleksi Qdrant yang diperiksa",
    )
    args = p.parse_args()

    client = _client()
    koleksi = args.collection
    yatim, ringkas, bertuan = _pindai(client, koleksi)

    print(f"Koleksi           : {koleksi}")
    print(f"Titik bertuan     : {bertuan}")
    print(f"Titik tanpa tenant: {len(yatim)}")
    if not yatim:
        print("\nTidak ada yang perlu dibuang.")
        return 0

    print("\nRincian titik yang akan dibuang:")
    for (course, content_id, week, source), n in sorted(ringkas.items(), key=lambda x: -x[1]):
        print(f"  {n:4} titik | course={course} | content_id={content_id} | "
              f"minggu={week} | {source}")

    if bertuan == 0:
        print("\nDIBATALKAN: tidak ada satu pun titik bertuan di koleksi ini.")
        print("Index tampaknya belum dimigrasi sama sekali; menghapus seluruh")
        print("isinya hampir pasti bukan yang Anda maksud. Periksa dulu, atau")
        print("pakai scripts/migrate_tenant_id.py untuk mencapkan pemilik.")
        return 1

    if not args.apply:
        print(f"\nUJI-KERING. Tidak ada yang dihapus. Jalankan ulang dengan --apply")
        print(f"untuk benar-benar membuang {len(yatim)} titik.")
        return 0

    for i in range(0, len(yatim), _BATCH):
        client.delete(
            collection_name=koleksi,
            points_selector=qm.PointIdsList(points=yatim[i:i + _BATCH]),
        )
    print(f"\n{len(yatim)} titik dibuang. Titik bertuan tidak tersentuh ({bertuan}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
