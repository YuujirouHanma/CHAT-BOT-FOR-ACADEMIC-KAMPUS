"""Buku pemakaian token harus bisa dijumlahkan, dan tidak pernah menggagalkan jawaban.

Sebelum modul ini ada, jumlah token hanya muncul sebagai teks di log server.
Cukup untuk menelusuri satu permintaan, tidak cukup untuk menjawab "bulan ini
kampus A menghabiskan berapa" — dan pertanyaan itulah yang menentukan model
bisnis serta pilihan model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import usage


@pytest.fixture(autouse=True)
def _isolasi(tmp_path, monkeypatch):
    """Tulis ke direktori sementara, bukan ke buku pemakaian sungguhan."""
    monkeypatch.setattr(usage, "DIR_PEMAKAIAN", tmp_path / "usage")
    usage.pasang_tenant(None)
    usage.pasang_tujuan(None)
    yield


def _usage(masuk=100, keluar=50, penalaran=None, biaya=None):
    rincian = SimpleNamespace(reasoning_tokens=penalaran) if penalaran else None
    u = SimpleNamespace(prompt_tokens=masuk, completion_tokens=keluar,
                        completion_tokens_details=rincian)
    if biaya is not None:
        u.cost = biaya
    return u


class TestPencatatan:
    def test_satu_panggilan_tercatat(self) -> None:
        usage.pasang_tenant("kampus-a")
        usage.catat(_usage(4120, 890, 310, 0.00042), model="qwen", tujuan="chat")

        b = usage.baca()
        assert len(b) == 1
        assert b[0]["tenant_id"] == "kampus-a"
        assert b[0]["masuk"] == 4120
        assert b[0]["keluar"] == 890
        assert b[0]["penalaran"] == 310
        assert b[0]["biaya_usd"] == pytest.approx(0.00042)

    def test_bentuk_dict_ikut_terbaca(self) -> None:
        """SDK penyedia mengembalikan objek pada satu versi, dict pada versi lain."""
        usage.catat({"prompt_tokens": 900, "completion_tokens": 240,
                     "completion_tokens_details": {"reasoning_tokens": 180},
                     "cost": 0.002}, model="qwen")
        b = usage.baca()
        assert b[0]["masuk"] == 900
        assert b[0]["penalaran"] == 180

    def test_tanpa_data_pemakaian_dilewati(self) -> None:
        """Baris kosong hanya mengotori penjumlahan tanpa menambah keterangan."""
        usage.catat(None, model="qwen")
        usage.catat(SimpleNamespace(), model="qwen")
        assert usage.baca() == []

    def test_biaya_tak_dilaporkan_jadi_null_bukan_nol(self) -> None:
        """Nol akan terbaca sebagai 'gratis' saat dijumlahkan."""
        usage.catat(_usage(biaya=None), model="lokal")
        assert usage.baca()[0]["biaya_usd"] is None


class TestTidakMenggagalkanPermintaan:
    def test_disk_bermasalah_tidak_melempar(self, tmp_path, monkeypatch) -> None:
        """Mahasiswa yang menunggu dua menit tidak boleh kehilangan jawabannya
        hanya karena pembukuan gagal menulis.

        Kegagalan dibuat nyata, bukan ditiru: direktori tujuan ditaruh di dalam
        sebuah BERKAS, sehingga mkdir benar-benar gagal seperti saat jalurnya
        tidak dapat ditulisi.
        """
        penghalang = tmp_path / "bukan-direktori"
        penghalang.write_text("x", encoding="utf-8")
        monkeypatch.setattr(usage, "DIR_PEMAKAIAN", penghalang / "usage")

        usage.catat(_usage(), model="qwen")   # tidak boleh melempar
        assert usage.baca() == []

    def test_baris_rusak_tidak_membuang_sebulan(self) -> None:
        """Penulisan terpotong saat mati listrik hanya merusak satu baris."""
        usage.catat(_usage(100, 50), model="qwen")
        berkas = next(usage.DIR_PEMAKAIAN.glob("*.jsonl"))
        berkas.write_text(berkas.read_text(encoding="utf-8") + '{"rusak\n',
                          encoding="utf-8")
        assert len(usage.baca()) == 1


class TestPenjumlahan:
    def test_dijumlahkan_per_kampus(self) -> None:
        usage.pasang_tenant("kampus-a")
        usage.catat(_usage(100, 50, biaya=0.001), model="qwen")
        usage.catat(_usage(200, 80), model="qwen")
        usage.pasang_tenant("kampus-b")
        usage.catat(_usage(900, 400, biaya=0.004), model="qwen")

        r = usage.ringkas(usage.baca(), "tenant_id")
        assert r["kampus-a"]["panggilan"] == 2
        assert r["kampus-a"]["keluar"] == 130
        assert r["kampus-b"]["masuk"] == 900

    def test_biaya_sebagian_ditandai(self) -> None:
        """Agar angka USD tidak terbaca sebagai total padahal hanya sebagian."""
        usage.pasang_tenant("kampus-a")
        usage.catat(_usage(biaya=0.001), model="qwen")
        usage.catat(_usage(), model="qwen")

        d = usage.ringkas(usage.baca(), "tenant_id")["kampus-a"]
        assert d["panggilan"] == 2
        assert d["biaya_diketahui"] == 1

    def test_dijumlahkan_per_tujuan(self) -> None:
        """Biaya penguraian pertanyaan harus bisa dipisahkan dari biaya jawaban."""
        usage.catat(_usage(800, 120), model="qwen", tujuan="uraikan")
        usage.catat(_usage(4000, 900), model="qwen", tujuan="chat")

        r = usage.ringkas(usage.baca(), "tujuan")
        assert r["uraikan"]["keluar"] == 120
        assert r["chat"]["keluar"] == 900


class TestTenantDariKonteks:
    def test_tenant_terbawa_tanpa_parameter(self) -> None:
        """Panggilan model terjadi jauh di dalam pipeline; menambahkan
        `tenant_id` ke tiap tanda tangan di antaranya akan menyentuh puluhan
        fungsi yang tidak peduli siapa kampusnya."""
        usage.pasang_tenant("kampus-x")
        usage.catat(_usage(), model="qwen")
        assert usage.baca()[0]["tenant_id"] == "kampus-x"

    def test_tanpa_tenant_tetap_tercatat(self) -> None:
        """Panggilan dari skrip CLI tidak punya kampus; tetap dibukukan."""
        usage.catat(_usage(), model="qwen")
        assert usage.baca()[0]["tenant_id"] is None


class TestTujuanTersambungKeJalurModel:
    """Laporan menawarkan `--per tujuan`; tanpa uji ini opsi itu pernah kosong.

    `pasang_tujuan` ada sejak awal tetapi tidak pernah dipanggil dari jalur
    model mana pun, sehingga setiap baris tercatat `tujuan: null` dan
    pengelompokan per tujuan menghasilkan satu kelompok "(tanpa)" berisi
    semuanya. Fitur yang tampak tersedia padahal tidak - jenis cacat yang
    paling lama tidak ketahuan, karena laporannya tetap jalan.
    """

    @pytest.mark.asyncio
    async def test_penguraian_dan_jawaban_terpisah(self) -> None:
        """Satu pertanyaan mahasiswa memanggil model tiga kali dengan tujuan beda.

        Yang diperiksa di sini dua langkah yang paling menentukan biaya:
        penguraian kueri (mekanis, murah) dan jawaban (mahal). Keduanya harus
        muncul sebagai kelompok terpisah, bukan menyatu.
        """
        from src.generation.llm import LLMGenerator
        from src.generation.prompts import FormattedContext

        balasan = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"query_diperkaya": "apa"}'),
                finish_reason="stop",
            )],
            usage=_usage(100, 20),
        )

        async def create(**_: object) -> SimpleNamespace:
            return balasan

        klien = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        gen = LLMGenerator(client=klien)  # type: ignore[arg-type]

        await gen.decompose_query("apa itu primary key?")
        await gen.generate("apa itu primary key?",
                           FormattedContext(text_block="ctx", image_payloads=[]))

        per_tujuan = usage.ringkas(usage.baca(), "tujuan")
        assert set(per_tujuan) == {"penguraian", "jawaban"}, (
            f"tujuan tidak terbawa ke buku pemakaian: {sorted(per_tujuan)}"
        )
        assert per_tujuan["penguraian"]["panggilan"] == 1
        assert per_tujuan["jawaban"]["panggilan"] == 1
