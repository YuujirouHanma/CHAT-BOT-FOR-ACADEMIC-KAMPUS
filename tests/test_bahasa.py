"""Penyaring aksara asing pada keluaran model.

Qwen dilatih terutama pada bahasa Tionghoa dan sesekali menyelipkan aksara Han
ke tengah kalimat Indonesia. Instruksi di prompt sistem sudah ada dan tetap
tidak cukup — kebocorannya terjadi saat sampling, bukan karena instruksinya
tidak dipahami.

Yang dijaga di sini ada dua, dan keduanya sama pentingnya:

1. Aksara CJK hilang, dan kalimat Indonesia di sekitarnya tetap utuh.
2. Aksara non-Latin yang LUMRAH di materi kuliah — Yunani untuk matematika,
   Arab untuk studi Islam, Latin beraksen — tidak ikut terbuang. Penyaring yang
   kelewat rakus merusak jawaban yang benar, dan itu lebih buruk daripada
   kebocoran yang hendak diperbaikinya.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.generation.bahasa import bersihkan_jawaban, buang_cjk


class TestPembuangan:
    def test_istilah_di_tengah_kalimat(self) -> None:
        teks = "Normalisasi 数据库 bertujuan menghilangkan redundansi"
        bersih, dibuang = buang_cjk(teks)
        assert bersih == "Normalisasi bertujuan menghilangkan redundansi"
        assert dibuang == 3

    def test_spasi_ganda_dirapikan(self) -> None:
        """Membuang aksara meninggalkan lubang; lubangnya ikut dirapikan."""
        bersih, _ = buang_cjk("Tabel 表 ini normal")
        assert "  " not in bersih
        assert bersih == "Tabel ini normal"

    def test_spasi_menggantung_sebelum_tanda_baca(self) -> None:
        bersih, _ = buang_cjk("Itu disebut kunci utama 主键.")
        assert bersih == "Itu disebut kunci utama."

    def test_aksara_di_dalam_kurung(self) -> None:
        bersih, _ = buang_cjk("Kunci utama (主键) bersifat unik")
        assert bersih == "Kunci utama () bersifat unik"

    def test_hiragana_katakana_hangul_ikut_terbuang(self) -> None:
        for asing in ("ひらがな", "カタカナ", "한글"):
            bersih, dibuang = buang_cjk(f"Materi {asing} selesai")
            assert dibuang > 0, f"{asing} lolos"
            assert "Materi" in bersih and "selesai" in bersih

    def test_teks_bersih_dikembalikan_apa_adanya(self) -> None:
        """Jalur umum tidak boleh menyentuh apa pun."""
        teks = "Normalisasi bertujuan menghilangkan redundansi data."
        bersih, dibuang = buang_cjk(teks)
        assert bersih is teks
        assert dibuang == 0

    def test_teks_kosong(self) -> None:
        assert buang_cjk("") == ("", 0)


class TestTidakKelewatRakus:
    """Aksara non-Latin yang memang dipakai materi kuliah harus selamat."""

    @pytest.mark.parametrize(
        "teks",
        [
            "Nilai λ dihitung dari Σx dibagi n",          # Yunani — matematika
            "Lafaz بسم الله ditulis di awal",              # Arab — studi Islam
            "Istilah café dan naïve tetap utuh",          # Latin beraksen
            "Suhu 30°C dengan ± 2 derajat",               # simbol
            "Rumus: ∫ f(x) dx = F(x) + C",                # matematika
            "Mata uang Rp 1.000,00 dan $5",               # mata uang
            "Emoji penanda ✓ dan ✗ dipakai di rubrik",    # tanda
        ],
    )
    def test_tidak_ikut_terbuang(self, teks: str) -> None:
        bersih, dibuang = buang_cjk(teks)
        assert dibuang == 0, f"penyaring merusak teks yang sah: {teks!r} -> {bersih!r}"
        assert bersih == teks


class TestPencatatan:
    """Pembuangan tidak boleh terjadi diam-diam.

    Angkanya bahan analisis: tanpa catatan, penyaring ini menyembunyikan persis
    gejala yang dipakai mengukur seberapa sering model gagal menjaga bahasa.

    Dicatat lewat loguru, yang tidak melewati `logging` bawaan — jadi `caplog`
    tidak melihat apa pun di sini. Perekam di bawah menggantikannya, meniru pola
    yang sudah dipakai `tests/test_llm.py`.
    """

    class _Perekam:
        """Menampung baris log; loguru memakai placeholder {} sehingga cukup format()."""

        def __init__(self) -> None:
            self.warning: list[str] = []
            self.error: list[str] = []

    @pytest.fixture
    def log(self, monkeypatch):
        rekam = self._Perekam()
        palsu = SimpleNamespace(
            warning=lambda m, *a: rekam.warning.append(m.format(*a)),
            error=lambda m, *a: rekam.error.append(m.format(*a)),
        )
        monkeypatch.setattr("src.generation.bahasa.logger", palsu)
        return rekam

    def test_kebocoran_kecil_dicatat_sebagai_peringatan(self, log) -> None:
        bersihkan_jawaban(
            "Normalisasi 数据库 " + "bertujuan menghilangkan redundansi " * 10,
            dari="uji",
        )
        assert len(log.warning) == 1, log.warning
        assert not log.error
        assert "3 aksara CJK dibuang" in log.warning[0]

    def test_jawaban_yang_salah_bahasa_dicatat_sebagai_galat(self, log) -> None:
        """Di atas ambang, membuang aksara tidak menyelamatkan jawabannya."""
        bersihkan_jawaban("数据库规范化的目的是消除数据冗余", dari="uji")
        assert len(log.error) == 1, log.error
        assert not log.warning
        assert "tidak dapat dipakai mahasiswa" in log.error[0]

    def test_teks_bersih_tidak_mencatat_apa_pun(self, log) -> None:
        hasil = bersihkan_jawaban("Jawaban yang sepenuhnya Indonesia.", dari="uji")
        assert hasil == "Jawaban yang sepenuhnya Indonesia."
        assert not log.warning and not log.error
