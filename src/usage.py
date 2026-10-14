"""Buku pemakaian token: berapa yang terbakar, oleh kampus mana, untuk apa.

Sebelum ini jumlah token hanya muncul sebagai teks di log server. Itu cukup
untuk menelusuri satu permintaan, tetapi tidak dapat dijumlahkan: tidak ada cara
menjawab "bulan ini kampus A menghabiskan berapa" tanpa mengurai berkas log
dengan regex, dan log dirotasi.

Padahal pertanyaan itulah yang menentukan model bisnis — model mana yang
sanggup dipakai pada harga berapa, dan apakah satu kampus memakai jauh lebih
banyak daripada yang lain.

BENTUK PENYIMPANAN
------------------
Satu baris JSON per panggilan model, ditambahkan ke berkas per bulan:

    data/usage/2026-10.jsonl

Dipilih JSONL, bukan basis data, karena tiga alasan. Penulisannya hanya
menambah di akhir berkas sehingga tidak pernah mengunci pembaca; berkasnya
dapat dijumlahkan dengan skrip sederhana maupun dibuka langsung saat
menelusuri; dan rotasi per bulan membuat berkas lama dapat diarsipkan tanpa
menyentuh yang berjalan.

TENANT DARI MANA
----------------
Lewat `ContextVar`, bukan parameter. Panggilan model terjadi jauh di dalam
pipeline, dan menambahkan `tenant_id` ke setiap tanda tangan di antaranya akan
menyentuh puluhan fungsi yang sama sekali tidak peduli siapa kampusnya.
`ContextVar` aman pada asyncio: tiap permintaan berjalan di task-nya sendiri,
dan nilainya tidak bocor antar permintaan yang berjalan bersamaan.

PENCATATAN TIDAK PERNAH MENGGAGALKAN PERMINTAAN
-----------------------------------------------
Seluruh kegagalan penulisan ditelan dan dicatat sebagai peringatan. Mahasiswa
yang sudah menunggu dua menit tidak boleh kehilangan jawabannya hanya karena
disk penuh saat pembukuan.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from src.config import PROJECT_ROOT
from src.utils.logger import logger

DIR_PEMAKAIAN = PROJECT_ROOT / "data" / "usage"

# Diisi saat kunci API diverifikasi, dibaca saat model dipanggil.
_tenant_sekarang: ContextVar[str | None] = ContextVar("tenant_pemakaian", default=None)
_tujuan_sekarang: ContextVar[str | None] = ContextVar("tujuan_pemakaian", default=None)


def pasang_tenant(tenant_id: str | None) -> None:
    """Tandai kampus pemilik permintaan yang sedang berjalan."""
    _tenant_sekarang.set(tenant_id)


def pasang_tujuan(tujuan: str | None) -> None:
    """Tandai untuk apa model dipanggil, mis. 'chat' atau 'kuis'.

    Dipisah dari tenant karena satu permintaan dapat memanggil model beberapa
    kali dengan tujuan berbeda — menguraikan pertanyaan, lalu menyusun jawaban.
    Tanpa pemisahan ini, biaya tahap penguraian tidak dapat dibedakan dari biaya
    jawabannya, dan justru itu yang perlu diketahui saat menekan biaya.
    """
    _tujuan_sekarang.set(tujuan)


def _angka(sumber: Any, *nama: str) -> int | None:
    """Ambil satu bilangan dari objek usage, apa pun bentuknya.

    SDK penyedia mengembalikan objek pada satu versi dan dict pada versi lain;
    keduanya ditemui pada pustaka yang sama.
    """
    for n in nama:
        nilai = getattr(sumber, n, None)
        if nilai is None and isinstance(sumber, dict):
            nilai = sumber.get(n)
        if nilai is not None:
            try:
                return int(nilai)
            except (TypeError, ValueError):
                return None
    return None


def catat(usage: Any, *, model: str, tujuan: str | None = None) -> None:
    """Bukukan satu panggilan model. Tidak pernah melempar.

    `usage` adalah objek pemakaian dari balasan penyedia. Panggilan tanpa data
    pemakaian dilewati diam-diam: mencatat baris kosong hanya akan mengotori
    penjumlahan tanpa menambah keterangan apa pun.
    """
    if usage is None:
        return

    try:
        masuk = _angka(usage, "prompt_tokens", "input_tokens")
        keluar = _angka(usage, "completion_tokens", "output_tokens")
        if masuk is None and keluar is None:
            return

        rincian = getattr(usage, "completion_tokens_details", None)
        if rincian is None and isinstance(usage, dict):
            rincian = usage.get("completion_tokens_details")
        penalaran = _angka(rincian, "reasoning_tokens") if rincian else None

        # `cost` hanya dikirim sebagian penyedia (OpenRouter mengirimnya).
        # Disimpan apa adanya; yang tidak mengirim dibiarkan null, bukan nol —
        # nol akan terbaca sebagai "gratis" saat dijumlahkan.
        biaya = getattr(usage, "cost", None)
        if biaya is None and isinstance(usage, dict):
            biaya = usage.get("cost")

        baris = {
            "pada": datetime.now(UTC).isoformat(timespec="seconds"),
            "tenant_id": _tenant_sekarang.get(),
            "model": model,
            "tujuan": tujuan or _tujuan_sekarang.get(),
            "masuk": masuk,
            "keluar": keluar,
            "penalaran": penalaran,
            "biaya_usd": float(biaya) if biaya is not None else None,
        }

        DIR_PEMAKAIAN.mkdir(parents=True, exist_ok=True)
        berkas = DIR_PEMAKAIAN / f"{datetime.now(UTC):%Y-%m}.jsonl"
        with berkas.open("a", encoding="utf-8") as f:
            f.write(json.dumps(baris, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - pembukuan tidak boleh menggagalkan jawaban
        logger.warning("Pemakaian token gagal dicatat: {}", exc)


def baca(bulan: str | None = None) -> list[dict]:
    """Seluruh baris pemakaian satu bulan; kosong bila belum ada.

    `bulan` berbentuk "2026-10"; kosong berarti bulan berjalan.
    """
    nama = bulan or f"{datetime.now(UTC):%Y-%m}"
    berkas = DIR_PEMAKAIAN / f"{nama}.jsonl"
    if not berkas.exists():
        return []

    hasil: list[dict] = []
    for baris in berkas.read_text(encoding="utf-8").splitlines():
        baris = baris.strip()
        if not baris:
            continue
        try:
            hasil.append(json.loads(baris))
        except json.JSONDecodeError:
            # Satu baris rusak (mis. penulisan terpotong saat mati listrik)
            # tidak boleh membuang seluruh bulan.
            continue
    return hasil


def ringkas(baris: list[dict], per: str = "tenant_id") -> dict[str, dict]:
    """Jumlahkan pemakaian menurut satu medan: tenant_id, model, atau tujuan."""
    hasil: dict[str, dict] = {}
    for b in baris:
        kunci = str(b.get(per) or "(tanpa)")
        d = hasil.setdefault(
            kunci, {"panggilan": 0, "masuk": 0, "keluar": 0, "penalaran": 0,
                    "biaya_usd": 0.0, "biaya_diketahui": 0},
        )
        d["panggilan"] += 1
        for medan in ("masuk", "keluar", "penalaran"):
            d[medan] += int(b.get(medan) or 0)
        if b.get("biaya_usd") is not None:
            d["biaya_usd"] += float(b["biaya_usd"])
            d["biaya_diketahui"] += 1
    return hasil
