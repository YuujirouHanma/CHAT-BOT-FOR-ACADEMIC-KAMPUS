"""Tests for course/week derivation from content_id."""
from __future__ import annotations

import pytest

from src.catalog import humanize_course, parse_content_id, resolve_course_week


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
        # Nama turunan memakai nama kurasi (display_name), bukan akronim mentah —
        # nama inilah yang muncul sebagai tombol pilihan bagi mahasiswa.
        assert resolve_course_week("sbd-minggu-2") == ("sbd", "Sistem Basis Data", 2)

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
