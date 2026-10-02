"""CLI penyediaan tenant dan kunci API.

    python -m scripts.tenantctl create <tenant_id> [--name "Kampus A"]
    python -m scripts.tenantctl issue-key <tenant_id> [--label "BE produksi"]
                                          [--scopes chat:ask,catalog:read]
                                          [--courses sbd,kka]
    python -m scripts.tenantctl list
    python -m scripts.tenantctl schedule <tenant_id>
    python -m scripts.tenantctl schedule <tenant_id> --set "4:kuis:1-4,7:ets:1-7"
    python -m scripts.tenantctl schedule <tenant_id> --reset
    python -m scripts.tenantctl revoke <tenant_id> <key_id>
    python -m scripts.tenantctl suspend <tenant_id>
    python -m scripts.tenantctl activate <tenant_id>
    python -m scripts.tenantctl verify-audit <tenant_id>

Nilai mentah kunci HANYA ditampilkan sekali, saat diterbitkan — yang tersimpan
hanya hash-nya. Kalau hilang, cabut kunci itu dan terbitkan yang baru; tidak ada
cara memulihkannya, dan memang itu tujuannya.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluation  # noqa: E402
from src.security import audit  # noqa: E402
from src.storage import tenant_store  # noqa: E402
from src.tenancy import ALL_SCOPES, READER_SCOPES, TenantQuota  # noqa: E402


def cmd_create(args: argparse.Namespace) -> int:
    kuota = TenantQuota(
        requests_per_minute=args.rpm,
        requests_per_day=args.rpd,
        max_upload_mb=args.max_upload_mb,
        max_storage_mb=args.max_storage_mb,
    )
    try:
        t = tenant_store.create_tenant(args.tenant_id, name=args.name or "", quota=kuota)
    except ValueError as exc:
        print(f"Gagal: {exc}", file=sys.stderr)
        return 1
    print(f"Tenant dibuat: {t.tenant_id} ({t.name})")
    print(f"  Kuota: {kuota.as_dict()}")
    print("\nTerbitkan kunci dengan:")
    print(f"  python -m scripts.tenantctl issue-key {t.tenant_id}")
    return 0


def cmd_issue_key(args: argparse.Namespace) -> int:
    if args.scopes:
        scopes = {s.strip() for s in args.scopes.split(",") if s.strip()}
    elif args.admin:
        scopes = set(ALL_SCOPES)
    else:
        scopes = set(READER_SCOPES)

    # Diurutkan dan dibuang kembarannya: nilainya jadi baku apa pun urutan
    # ketik operator, sehingga dua kunci dengan cakupan sama tersimpan sama
    # persis dan perubahan berkas tenant mudah dibaca.
    courses = (
        sorted({c.strip() for c in args.courses.split(",") if c.strip()})
        if args.courses else None
    )

    try:
        k = tenant_store.issue_key(
            args.tenant_id, label=args.label or "",
            scopes=scopes, allowed_courses=courses,
        )
    except ValueError as exc:
        print(f"Gagal: {exc}", file=sys.stderr)
        return 1

    print(f"Kunci diterbitkan untuk tenant '{args.tenant_id}'")
    print(f"  key_id : {k.key_id}")
    print(f"  hak    : {', '.join(sorted(scopes))}")
    if courses:
        print(f"  matkul : {', '.join(courses)}")
    print("\n  KUNCI (disimpan sekarang juga — tidak ditampilkan lagi):\n")
    print(f"    {k.raw}\n")
    print("  Kirimkan lewat kanal yang aman. Server hanya menyimpan hash-nya.")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    tenants = tenant_store.list_tenants()
    if not tenants:
        print("Belum ada tenant.")
        return 0
    for t in tenants:
        aktif = sum(1 for k in t.api_keys if k.is_active)
        print(f"{t.tenant_id:20} {t.status:10} kunci_aktif={aktif:<3} {t.name}")
        for k in t.api_keys:
            tanda = " " if k.is_active else "✗"
            print(f"   {tanda} {k.key_id}  {k.label or '(tanpa label)':24} "
                  f"[{', '.join(k.scopes)}]")
    return 0


def _parse_schedule_spec(spec: str) -> evaluation.EvaluationSchedule:
    """Ubah "4:kuis:1-4,7:ets:1-7" menjadi jadwal yang sudah tervalidasi.

    Bentuk teks ini hanya kulit CLI; keabsahannya diputuskan oleh
    `EvaluationSchedule.from_json` yang sama dengan yang dipakai saat berkas
    tenant dimuat — supaya tidak ada jadwal yang lolos lewat CLI tetapi ditolak
    saat server membacanya (atau sebaliknya).
    """
    entri: list[dict[str, object]] = []
    for bagian in spec.split(","):
        bagian = bagian.strip()
        if not bagian:
            continue
        potong = bagian.split(":")
        if len(potong) != 3:
            raise ValueError(
                f"Bentuk '{bagian}' tidak dikenali; pakai minggu:jenis:awal-akhir"
            )
        minggu, jenis, rentang = (p.strip() for p in potong)
        if "-" not in rentang:
            raise ValueError(f"Rentang '{rentang}' harus ditulis awal-akhir")
        awal, _, akhir = rentang.partition("-")
        try:
            angka = [int(minggu), int(awal), int(akhir)]
        except ValueError as exc:
            raise ValueError(f"Angka minggu tidak sah pada '{bagian}'") from exc
        entri.append({"week": angka[0], "kind": jenis, "covers": angka[1:]})
    return evaluation.EvaluationSchedule.from_json(entri)


def _print_schedule(tenant: tenant_store.Tenant) -> None:
    jadwal = tenant.evaluation_schedule
    asal = "sendiri" if jadwal else "bawaan (tenant belum menentukan)"
    print(f"Jadwal evaluasi '{tenant.tenant_id}' — {asal}:")
    rencana = evaluation.plans_for(jadwal)
    for w in sorted(rencana):
        p = rencana[w]
        print(f"  minggu {w:<3} {p.label:28} menguji {p.range_text:16} "
              f"{p.blueprint.total} soal {p.blueprint.counts}")


def cmd_schedule(args: argparse.Namespace) -> int:
    """Tampilkan, tetapkan, atau kembalikan kalender evaluasi sebuah tenant."""
    tenant = tenant_store.get(args.tenant_id)
    if tenant is None:
        print("Tenant tidak ditemukan.", file=sys.stderr)
        return 1

    if args.set and args.reset:
        print("Pilih salah satu: --set atau --reset.", file=sys.stderr)
        return 1

    if args.set or args.reset:
        try:
            baru = _parse_schedule_spec(args.set) if args.set else None
        except ValueError as exc:
            print(f"Jadwal tidak sah: {exc}", file=sys.stderr)
            return 1
        tenant_store.set_schedule(args.tenant_id, baru)
        tenant = tenant_store.get(args.tenant_id)
        assert tenant is not None

    _print_schedule(tenant)
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    if tenant_store.revoke_key(args.tenant_id, args.key_id):
        print(f"Kunci {args.key_id} dicabut.")
        return 0
    print("Kunci tidak ditemukan atau sudah dicabut.", file=sys.stderr)
    return 1


def cmd_status(args: argparse.Namespace, status: str) -> int:
    if tenant_store.set_status(args.tenant_id, status):
        print(f"Tenant {args.tenant_id} → {status}")
        return 0
    print("Tenant tidak ditemukan.", file=sys.stderr)
    return 1


def cmd_verify_audit(args: argparse.Namespace) -> int:
    """Periksa keutuhan rantai jejak audit satu tenant."""
    direktori = audit.AUDIT_DIR / args.tenant_id
    if not direktori.exists():
        print(f"Tidak ada jejak audit untuk '{args.tenant_id}'.", file=sys.stderr)
        return 1
    rusak = 0
    for berkas in sorted(direktori.glob("*.jsonl")):
        utuh, jumlah, ket = audit.verify_chain(berkas)
        tanda = "OK " if utuh else "GAGAL"
        print(f"[{tanda}] {berkas.name}  {jumlah} baris  {ket}")
        if not utuh:
            rusak += 1
    return 1 if rusak else 0


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI.

    `argv` diterima sebagai argumen, bukan dibaca dari sys.argv, supaya
    perintahnya dapat diuji langsung tanpa menjalankan proses terpisah.
    """
    p = argparse.ArgumentParser(prog="tenantctl", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="Buat tenant baru")
    c.add_argument("tenant_id")
    c.add_argument("--name", default="")
    c.add_argument("--rpm", type=int, default=60, help="permintaan per menit")
    c.add_argument("--rpd", type=int, default=5000, help="permintaan per hari")
    c.add_argument("--max-upload-mb", type=int, default=100)
    c.add_argument("--max-storage-mb", type=int, default=20000)
    c.set_defaults(fn=cmd_create)

    k = sub.add_parser("issue-key", help="Terbitkan kunci API")
    k.add_argument("tenant_id")
    k.add_argument("--label", default="")
    k.add_argument("--scopes", default="", help="dipisah koma; kosong = hak pembaca")
    k.add_argument("--admin", action="store_true", help="beri seluruh hak")
    k.add_argument("--courses", default="", help="batasi ke mata kuliah tertentu")
    k.set_defaults(fn=cmd_issue_key)

    sub.add_parser("list", help="Daftar tenant & kunci").set_defaults(fn=cmd_list)

    j = sub.add_parser("schedule", help="Lihat/atur kalender evaluasi tenant")
    j.add_argument("tenant_id")
    j.add_argument(
        "--set", default="",
        help="jadwal kampus, mis. \"4:kuis:1-4,7:ets:1-7,11:kuis:8-11,14:eas:8-14\"",
    )
    j.add_argument(
        "--reset", action="store_true", help="kembalikan ke jadwal bawaan",
    )
    j.set_defaults(fn=cmd_schedule)

    r = sub.add_parser("revoke", help="Cabut satu kunci")
    r.add_argument("tenant_id")
    r.add_argument("key_id")
    r.set_defaults(fn=cmd_revoke)

    s = sub.add_parser("suspend", help="Bekukan tenant")
    s.add_argument("tenant_id")
    s.set_defaults(fn=lambda a: cmd_status(a, tenant_store.STATUS_SUSPENDED))

    a = sub.add_parser("activate", help="Aktifkan kembali tenant")
    a.add_argument("tenant_id")
    a.set_defaults(fn=lambda x: cmd_status(x, tenant_store.STATUS_ACTIVE))

    v = sub.add_parser("verify-audit", help="Periksa keutuhan jejak audit")
    v.add_argument("tenant_id")
    v.set_defaults(fn=cmd_verify_audit)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
