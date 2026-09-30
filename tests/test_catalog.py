"""Tests for course/week derivation from content_id."""
from __future__ import annotations

import ast
import inspect

import pytest

import src.catalog as catalog
from src.catalog import display_name, humanize_course, parse_content_id, resolve_course_week


class TestParseContentId:
    @pytest.mark.parametrize(
        "content_id,course,week",
        [
            ("sbd-minggu-2", "sbd", 2),
            ("sbd_minggu_2", "sbd", 2),
            ("sbd/minggu_2", "sbd", 2),
            ("algo-w3", "algo", 3),
            ("algo-week-10", "algo", 10),
            ("dasar-pemrograman-minggu-1", "dasar-pemrograman", 1),
        ],
    )
    def test_parses_week_patterns(self, content_id: str, course: str, week: int) -> None:
        assert parse_content_id(content_id) == (course, week)

    @pytest.mark.parametrize(
        "content_id",
        ["class-3", "kka", "sbd", "team-7"],
    )
    def test_no_week_pattern_keeps_whole_id_as_course(self, content_id: str) -> None:
        course, week = parse_content_id(content_id)
        assert course == content_id
        assert week is None

    def test_empty_returns_none(self) -> None:
        assert parse_content_id(None) == (None, None)
        assert parse_content_id("") == (None, None)


class TestHumanizeCourse:
    def test_short_slug_uppercased_as_acronym(self) -> None:
        assert humanize_course("sbd") == "SBD"
        assert humanize_course("kka") == "KKA"

    def test_multiword_title_cased(self) -> None:
        assert humanize_course("dasar-pemrograman") == "Dasar Pemrograman"
        assert humanize_course("struktur_data") == "Struktur Data"

    def test_none(self) -> None:
        assert humanize_course(None) is None


class TestResolveCourseWeek:
    def test_derives_from_content_id(self) -> None:
        # Tanpa course_name eksplisit, nama turunan adalah tebakan mekanis dari slug.
        # Tidak boleh ada nama kurasi milik satu kampus di sini: slug "sbd" bisa berarti
        # mata kuliah berbeda di kampus lain, dan nama itu ikut ke prompt LLM.
        assert resolve_course_week("sbd-minggu-2") == ("sbd", "SBD", 2)

    def test_explicit_values_win(self) -> None:
        result = resolve_course_week(
            "sbd-minggu-2", course_id="fisika", course_name="Fisika Dasar", week=5
        )
        assert result == ("fisika", "Fisika Dasar", 5)

    def test_partial_explicit_merges_with_parsed(self) -> None:
        # explicit course, week still derived from content_id
        assert resolve_course_week("sbd-minggu-2", course_name="Sistem Basis Data") == (
            "sbd",
            "Sistem Basis Data",
            2,
        )


class TestTidakAdaNamaKurasi:
    """Penjaga agar nama mata kuliah khusus satu kampus tidak kembali ke modul ini.

    Satu pemasangan melayani banyak kampus. Kamus nama kurasi di kode pernah membuat
    kampus lain yang memakai slug sama ("kka") melihat nama mata kuliah kampus lain
    pada tombol pilihannya — dan nama itu ikut masuk ke prompt LLM.
    """

    @pytest.mark.parametrize(
        "course_id",
        ["sbd", "kka", "rsbp", "strukdat", "dasprog-rka", "dasprog-if"],
    )
    def test_display_name_hanya_tebakan_mekanis(self, course_id: str) -> None:
        assert display_name(course_id) == humanize_course(course_id)

    def test_display_name_kosong(self) -> None:
        assert display_name(None) is None
        assert display_name("") is None

    def test_modul_tidak_memuat_kamus_nama(self) -> None:
        # Menangkap kamus slug→nama yang muncul lagi dengan nama variabel apa pun.
        kamus = {
            nama: nilai
            for nama, nilai in vars(catalog).items()
            if isinstance(nilai, dict)
            and nilai
            and all(isinstance(k, str) and isinstance(v, str) for k, v in nilai.items())
        }
        assert kamus == {}, f"kamus nama mata kuliah tertanam di kode: {sorted(kamus)}"

    def test_tidak_ada_literal_nama_mata_kuliah(self) -> None:
        # Menangkap nama kurasi yang kembali lewat if/elif atau konstanta lepas,
        # bukan hanya lewat kamus. Docstring dilewati: contoh seperti
        # 'dasar-pemrograman' → 'Dasar Pemrograman' justru menerangkan tebakan mekanis.
        pohon = ast.parse(inspect.getsource(catalog))
        for simpul in ast.walk(pohon):
            if isinstance(simpul, ast.Module | ast.FunctionDef | ast.ClassDef):
                simpul.body = simpul.body[1:] if ast.get_docstring(simpul) else simpul.body
        literal = [
            n.value
            for n in ast.walk(pohon)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        mencurigakan = [t for t in literal if len(t.split()) >= 2 and t[:1].isupper()]
        assert mencurigakan == [], f"nama mata kuliah tertanam di kode: {mencurigakan}"
