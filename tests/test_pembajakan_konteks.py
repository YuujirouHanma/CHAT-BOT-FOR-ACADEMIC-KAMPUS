"""Regresi: nama mata kuliah di dalam pertanyaan isi tidak boleh menimpa sesi.

Kata umum seperti "data", "struktur", dan "dasar" muncul di hampir semua
pertanyaan materi, sekaligus menjadi bagian nama mata kuliah yang terindex.
Sebelum perbaikan, pertanyaan isi biasa — termasuk pertanyaan template yang
dihasilkan sistem sendiri — mencocoki nama mata kuliah lain lalu memindahkan
konteks sesi tanpa diminta: mahasiswa yang sedang di satu mata kuliah tiba-tiba
dijawab dari mata kuliah lain, tanpa pesan galat apa pun.

Perpindahan yang MEMANG diminta ("ganti ke basis data", "buka mata kuliah X")
harus tetap bekerja — itu satu-satunya jalan pindah bagi mahasiswa.
"""
from __future__ import annotations

import pytest

from src import guided

COURSES = [
    {"course_id": "sbd", "course_name": "Sistem Basis Data"},
    {"course_id": "strukdat", "course_name": "Struktur Data"},
    {"course_id": "dasprog", "course_name": "Dasar Pemrograman"},
]


class TestPertanyaanIsiTidakMemindahkanSesi:
    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "apa itu struktur data?",
            "Apa itu struktur data dan mengapa penting?",
            "bagaimana cara kerja basis data dalam sistem informasi?",
            "jelaskan dasar pemrograman berorientasi objek",
            "sebutkan contoh struktur data yang sering dipakai",
        ],
    )
    def test_nama_mata_kuliah_sebagai_topik_bukan_pilihan(
        self, pertanyaan: str,
    ) -> None:
        assert guided.match_course(pertanyaan, COURSES) is None

    @pytest.mark.parametrize(
        "pertanyaan",
        [
            "apa itu struktur data?",
            "jelaskan dasar pemrograman berorientasi objek",
        ],
    )
    def test_resolve_refs_tidak_mengembalikan_course(self, pertanyaan: str) -> None:
        """`resolve_refs` dipakai pemanggil untuk menimpa konteks sesi.

        Selama ia mengembalikan course_id, pemanggil akan menimpa mata kuliah
        yang sedang dibuka — jadi jaminannya harus ada di lapisan ini.
        """
        assert guided.resolve_refs(pertanyaan, COURSES).course_id is None


class TestPerpindahanYangDimintaTetapJalan:
    @pytest.mark.parametrize(
        ("pesan", "harapan"),
        [
            ("ganti ke basis data", "sbd"),
            ("pindah ke struktur data", "strukdat"),
            ("buka mata kuliah struktur data", "strukdat"),
            ("bisa ganti ke struktur data?", "strukdat"),
            ("saya mau belajar dasar pemrograman", "dasprog"),
            ("sistem basis data", "sbd"),
            ("strukdat", "strukdat"),
            # Klik tombol pilihan mata kuliah: klien mengirim course_id-nya.
            ("sbd", "sbd"),
        ],
    )
    def test_perintah_pindah_eksplisit(self, pesan: str, harapan: str) -> None:
        assert guided.match_course(pesan, COURSES) == harapan

    def test_pertanyaan_dengan_penanda_konteks_tetap_menyaring(self) -> None:
        """"di sbd minggu 3" = penyaringan yang diminta, bukan pembajakan."""
        refs = guided.resolve_refs("apa itu normalisasi di sbd minggu 3?", COURSES)
        assert (refs.course_id, refs.weeks) == ("sbd", [3])
