"""Laporan pemakaian token: berapa yang terbakar, oleh siapa, untuk apa.

    python -m scripts.usage_report                 # bulan berjalan
    python -m scripts.usage_report --bulan 2026-09
    python -m scripts.usage_report --per model
    python -m scripts.usage_report --csv > pemakaian.csv

Dibuat sebagai skrip, bukan endpoint, karena pembacanya pengelola — bukan
aplikasi. Menjadikannya endpoint berarti menambah satu jalur ber-hak akses yang
harus dijaga, untuk angka yang cukup dibaca sebulan sekali.

Biaya hanya dijumlahkan dari panggilan yang penyedianya MELAPORKAN biaya.
Jumlah panggilan yang tidak melaporkannya ditampilkan terpisah, supaya angka
rupiah tidak terbaca sebagai total padahal hanya sebagian.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime

from src import usage


def _tabel(ringkasan: dict[str, dict], judul_kolom: str) -> None:
    if not ringkasan:
        print("  (belum ada pemakaian tercatat)")
        return

    lebar = max(len(judul_kolom), max(len(k) for k in ringkasan))
    print(f"  {judul_kolom:<{lebar}}  {'panggilan':>9}  {'token masuk':>12}"
          f"  {'token keluar':>12}  {'penalaran':>10}  {'biaya USD':>11}")
    print("  " + "-" * (lebar + 62))

    for kunci, d in sorted(ringkasan.items(), key=lambda x: -x[1]["keluar"]):
        # Biaya ditandai bintang bila sebagian panggilan tidak melaporkannya,
        # supaya tidak dikira jumlah yang utuh.
        tanda = "" if d["biaya_diketahui"] == d["panggilan"] else "*"
        biaya = f"{d['biaya_usd']:.4f}{tanda}" if d["biaya_diketahui"] else "-"
        print(f"  {kunci:<{lebar}}  {d['panggilan']:>9,}  {d['masuk']:>12,}"
              f"  {d['keluar']:>12,}  {d['penalaran']:>10,}  {biaya:>11}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bulan", help="format 2026-10; kosong = bulan berjalan")
    p.add_argument("--per", default="tenant_id",
                   choices=["tenant_id", "model", "tujuan"],
                   help="dikelompokkan menurut apa")
    p.add_argument("--csv", action="store_true", help="keluarkan sebagai CSV")
    a = p.parse_args(argv)

    bulan = a.bulan or f"{datetime.now(UTC):%Y-%m}"
    baris = usage.baca(bulan)

    if a.csv:
        w = csv.writer(sys.stdout, lineterminator="\n")
        w.writerow([a.per, "panggilan", "token_masuk", "token_keluar",
                    "penalaran", "biaya_usd", "biaya_diketahui"])
        for kunci, d in sorted(usage.ringkas(baris, a.per).items()):
            w.writerow([kunci, d["panggilan"], d["masuk"], d["keluar"],
                        d["penalaran"], f"{d['biaya_usd']:.6f}",
                        d["biaya_diketahui"]])
        return 0

    print(f"\nPemakaian token — {bulan}, dikelompokkan per {a.per}")
    print(f"Sumber: data/usage/{bulan}.jsonl ({len(baris):,} panggilan)\n")
    _tabel(usage.ringkas(baris, a.per), a.per)

    if baris:
        total = usage.ringkas(baris, "model")
        tak_berbiaya = sum(d["panggilan"] - d["biaya_diketahui"] for d in total.values())
        if tak_berbiaya:
            print(f"\n  * {tak_berbiaya:,} panggilan tidak melaporkan biaya; "
                  f"angka USD hanya mencakup sisanya.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
