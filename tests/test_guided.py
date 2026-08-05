"""Tests for guided navigation intent resolution.

Pengenal maksud ini dipakai di SETIAP pesan chat, jadi salah tafsir langsung
terasa oleh mahasiswa: pertanyaan sungguhan malah dijawab dengan menu, atau
sebaliknya menu malah dikirim ke retrieval. Karena itu batas antara "ini
pilihan" dan "ini pertanyaan" diuji cukup rinci.
"""
from __future__ import annotations

import pytest

from src import guided

COURSES = [
    {"course_id": "kka", "course_name": "KKA"},
    {"course_id": "sbd", "course_name": "Sistem Basis Data"},
    {"course_id": "dasprog", "course_name": "Dasar Pemrograman"},
]

MATERIALS = [
    {"source_file": "Materi SBD TM9(Materi).pptx", "content_id": "sbd-minggu-3"},
    {"source_file": "Branching and Iteration - MIT.pdf", "content_id": "sbd-minggu-3"},
    {"source_file": "01-Artificial Intelligence.pdf", "content_id": "sbd-minggu-3"},
]


class TestMatchCourse:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("sbd", "sbd"),
            ("saya mau belajar sbd", "sbd"),
            ("sistem basis data", "sbd"),
            ("materi sistem basis data minggu 3", "sbd"),
            ("Dasar Pemrograman", "dasprog"),
            ("kka", "kka"),
        ],
    )
    def test_recognises_indexed_courses(self, text: str, expected: str) -> None:
        assert guided.match_course(text, COURSES) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "apa itu normalisasi?",
            "materi minggu ini apa",   # semuanya stopword
            "minggu 3",
            "",
        ],
    )
    def test_returns_none_when_no_course_named(self, text: str) -> None:
        assert guided.match_course(text, COURSES) is None

    def test_never_invents_a_course(self) -> None:
        """Kosakata hanya dari yang terindex — 'kalkulus' tidak boleh dipaksa cocok."""
        assert guided.match_course("saya mau belajar kalkulus", COURSES) is None

    def test_no_courses_indexed(self) -> None:
        assert guided.match_course("sbd", []) is None


class TestFindWeek:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("minggu 3", 3),
            ("minggu ke 3", 3),
            ("minggu ke-3", 3),
            ("week 5", 5),
            ("pertemuan 9", 9),
            ("tm9", 9),
            ("w3", 3),
            ("minggu 12", 12),
            ("apa itu normalisasi", None),
        ],
    )
    def test_week_patterns(self, text: str, expected: int | None) -> None:
        assert guided.find_week(text) == expected

    def test_bare_number_ignored_by_default(self) -> None:
        """Supaya angka di tengah pertanyaan biasa tidak disangka nomor minggu."""
        assert guided.find_week("3") is None
        assert guided.find_week("ada berapa 3 bentuk normal") is None

    def test_bare_number_accepted_when_asking_week(self) -> None:
        assert guided.find_week("3", allow_bare_number=True) == 3


class TestLooksLikeQuestion:
    @pytest.mark.parametrize(
        "text",
        [
            "apa itu normalisasi?",
            "jelaskan perulangan",
            "bagaimana cara membuat ERD",
            "apa perbedaan primary key dan foreign key",
            "kenapa perlu indexing",
        ],
    )
    def test_real_questions(self, text: str) -> None:
        assert guided.looks_like_question(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "sbd",
            "minggu 3",
            "saya mau mata kuliah sbd minggu 3",
            "halo",
            "",
        ],
    )
    def test_selections_are_not_questions(self, text: str) -> None:
        assert guided.looks_like_question(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "materi sbd minggu 3 apa saja?",
            "apa aja materinya",
            "daftar materi",
            "ada berapa materi",
        ],
    )
    def test_navigation_phrases_are_not_questions(self, text: str) -> None:
        """"apa saja" pada pesan PENDEK minta DAFTAR pilihan, bukan penjelasan."""
        assert guided.looks_like_question(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            # Bunyi khas pertanyaan template yang kita hasilkan sendiri — dulu
            # ini terbaca sebagai navigasi karena memuat "apa saja", sehingga
            # klik pertanyaan template tidak pernah dijawab.
            "Apa itu DML dan perintah apa saja yang termasuk di dalamnya?",
            "Operator apa saja yang dapat digunakan untuk memfilter data?",
            "Apa saja bentuk normal dalam normalisasi basis data dan bedanya?",
        ],
    )
    def test_long_question_containing_nav_phrase_is_still_a_question(
        self, text: str
    ) -> None:
        assert guided.looks_like_question(text) is True


class TestWantsStartAndReset:
    @pytest.mark.parametrize("text", ["halo", "mau belajar", "bingung", "", "menu"])
    def test_wants_start(self, text: str) -> None:
        assert guided.wants_start(text) is True

    @pytest.mark.parametrize(
        "text", ["apa itu normalisasi?", "saya mau sbd minggu 3"]
    )
    def test_does_not_want_start(self, text: str) -> None:
        assert guided.wants_start(text) is False

    @pytest.mark.parametrize(
        "text",
        ["ganti mata kuliah", "ganti materi", "mulai dari awal", "reset", "batal"],
    )
    def test_wants_reset(self, text: str) -> None:
        assert guided.wants_reset(text) is True

    def test_normal_question_is_not_reset(self) -> None:
        assert guided.wants_reset("apa itu normalisasi?") is False

    @pytest.mark.parametrize(
        "text",
        [
            # "ulang" adalah substring dari "perulangan" — topik nyata di materi
            # pemrograman. Dengan pencocokan substring, pertanyaan ini dulu
            # diartikan "mulai ulang" dan tidak pernah dijawab.
            "apa itu perulangan?",
            "jelaskan perulangan bersarang",
            "bagaimana cara mengulangi proses ini",
        ],
    )
    def test_topic_words_containing_reset_phrase_are_not_reset(
        self, text: str
    ) -> None:
        assert guided.wants_reset(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            # "list" adalah substring dari "listrik".
            "apa itu listrik statis?",
            "jelaskan rangkaian listrik",
        ],
    )
    def test_topic_words_containing_nav_phrase_are_still_questions(
        self, text: str
    ) -> None:
        assert guided.looks_like_question(text) is True
        assert guided.wants_reset(text) is False


class TestMatchMaterial:
    def test_full_filename(self) -> None:
        got = guided.match_material("Materi SBD TM9(Materi).pptx", MATERIALS)
        assert got is not None
        assert got["source_file"] == "Materi SBD TM9(Materi).pptx"

    def test_partial_title(self) -> None:
        got = guided.match_material("branching and iteration", MATERIALS)
        assert got is not None
        assert got["source_file"] == "Branching and Iteration - MIT.pdf"

    def test_ordinal_when_asking_material(self) -> None:
        """Pengguna baru sering menjawab '2' saja untuk pilihan kedua."""
        got = guided.match_material("2", MATERIALS, allow_ordinal=True)
        assert got is not None
        assert got["source_file"] == "Branching and Iteration - MIT.pdf"

    def test_ordinal_out_of_range(self) -> None:
        assert guided.match_material("9", MATERIALS, allow_ordinal=True) is None

    def test_ordinal_disabled_by_default(self) -> None:
        assert guided.match_material("2", MATERIALS) is None

    def test_unrelated_text(self) -> None:
        assert guided.match_material("apa itu normalisasi", MATERIALS) is None

    def test_empty_material_list(self) -> None:
        assert guided.match_material("apa pun", []) is None


class TestQuizHelpers:
    OPSI_BERNOMOR = ["A. WHERE x = NULL", "B. WHERE x IS NULL"]
    OPSI_POLOS = ["WHERE x = NULL", "WHERE x IS NULL"]

    def test_option_labels_not_double_numbered(self) -> None:
        """LLM kerap menomori opsinya sendiri — jangan sampai jadi 'A. A. ...'."""
        labels = [c.label for c in guided.quiz_option_choices(self.OPSI_BERNOMOR)]
        assert labels == ["A. WHERE x = NULL", "B. WHERE x IS NULL"]

    def test_option_labels_added_when_missing(self) -> None:
        labels = [c.label for c in guided.quiz_option_choices(self.OPSI_POLOS)]
        assert labels == ["A. WHERE x = NULL", "B. WHERE x IS NULL"]

    def test_choice_value_is_index(self) -> None:
        c = guided.quiz_option_choices(self.OPSI_POLOS)
        assert [x.value for x in c] == ["0", "1"]
        assert all(x.kind == "quiz" for x in c)

    @pytest.mark.parametrize(
        ("jawaban", "harap"),
        [
            ("B", 1), ("b", 1), ("A.", 0), ("2", 1), ("1", 0),
            ("WHERE x IS NULL", 1),
            ("B. WHERE x IS NULL", 1),
        ],
    )
    def test_match_option_accepts_letter_number_and_text(
        self, jawaban: str, harap: int
    ) -> None:
        assert guided.match_quiz_option(jawaban, self.OPSI_POLOS) == harap

    def test_match_option_on_prefixed_options(self) -> None:
        assert guided.match_quiz_option("B. WHERE x IS NULL", self.OPSI_BERNOMOR) == 1

    @pytest.mark.parametrize("jawaban", ["hmm tidak tahu", "", "Z", "9"])
    def test_unrecognised_answer_returns_none(self, jawaban: str) -> None:
        """Harus None, bukan tebakan — jawaban tak jelas tidak boleh dihitung salah."""
        assert guided.match_quiz_option(jawaban, self.OPSI_POLOS) is None

    @pytest.mark.parametrize(
        "teks", ["kuis", "mau kuis dong", "latihan soal", "tes pemahaman"]
    )
    def test_wants_quiz(self, teks: str) -> None:
        assert guided.wants_quiz(teks) is True

    @pytest.mark.parametrize("teks", ["apa itu normalisasi?", "sbd", "minggu 3"])
    def test_does_not_want_quiz(self, teks: str) -> None:
        assert guided.wants_quiz(teks) is False

    def test_result_message_shows_score_and_explanation(self) -> None:
        pesan = guided.quiz_result_message(
            50.0, 1, 2,
            [
                {"question": "Soal A", "options": self.OPSI_POLOS,
                 "your_answer": 0, "correct_answer": 0, "is_correct": True,
                 "explanation": "Karena begitu."},
                {"question": "Soal B", "options": self.OPSI_POLOS,
                 "your_answer": 0, "correct_answer": 1, "is_correct": False,
                 "explanation": "Harus IS NULL."},
            ],
        )
        assert "Skor kamu: 50" in pesan
        assert "Karena begitu." in pesan and "Harus IS NULL." in pesan
        assert "Jawabanmu: A. WHERE x = NULL" in pesan   # hanya untuk yang salah
        assert pesan.count("Jawabanmu") == 1


class TestNextStep:
    @pytest.mark.parametrize(
        ("course", "week", "source", "expected"),
        [
            (None, None, None, guided.STEP_COURSE),
            ("sbd", None, None, guided.STEP_WEEK),
            ("sbd", 3, None, guided.STEP_MATERIAL),
            ("sbd", 3, "bab3.pdf", guided.STEP_QUESTION),
        ],
    )
    def test_progression(
        self, course: str | None, week: int | None,
        source: str | None, expected: str,
    ) -> None:
        assert guided.next_step(
            course_id=course, week=week, source_file=source,
        ) == expected


class TestResolveRefs:
    def test_course_and_week_in_one_message(self) -> None:
        refs = guided.resolve_refs("saya mau mata kuliah sbd, minggu 3", COURSES)
        assert refs.course_id == "sbd"
        assert refs.week == 3
        assert refs.any_found is True

    def test_question_can_carry_context(self) -> None:
        """"apa itu normalisasi di sbd minggu 3?" = pertanyaan DAN filter."""
        text = "apa itu normalisasi di sbd minggu 3?"
        refs = guided.resolve_refs(text, COURSES)
        assert (refs.course_id, refs.week) == ("sbd", 3)
        assert guided.looks_like_question(text) is True

    def test_nothing_found(self) -> None:
        refs = guided.resolve_refs("apa itu normalisasi?", COURSES)
        assert refs.any_found is False

    def test_awaiting_week_allows_bare_number(self) -> None:
        refs = guided.resolve_refs("3", COURSES, awaiting=guided.STEP_WEEK)
        assert refs.week == 3

    def test_material_matched_when_list_given(self) -> None:
        refs = guided.resolve_refs(
            "branching and iteration", COURSES, materials=MATERIALS,
        )
        assert refs.source_file == "Branching and Iteration - MIT.pdf"
        assert refs.content_id == "sbd-minggu-3"


class TestPrompts:
    @pytest.mark.parametrize(
        "step",
        [guided.STEP_COURSE, guided.STEP_WEEK,
         guided.STEP_MATERIAL, guided.STEP_QUESTION],
    )
    def test_every_step_has_a_prompt(self, step: str) -> None:
        text = guided.prompt_for(
            step, course_name="Sistem Basis Data", week=3, source_file="bab3.pdf",
        )
        assert text.strip()

    def test_answer_step_has_no_prompt(self) -> None:
        assert guided.prompt_for(guided.STEP_ANSWER) == ""

    def test_empty_messages_mention_the_context(self) -> None:
        msg = guided.empty_message(
            guided.STEP_MATERIAL, course_name="Sistem Basis Data", week=3,
        )
        assert "Sistem Basis Data" in msg
        assert "3" in msg
