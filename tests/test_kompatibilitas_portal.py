"""Permintaan APA ADANYA dari portal kampus harus dilayani, bukan dibalas 422.

`QueryRequest` memakai `extra="forbid"`, dan itu memang disengaja — medan yang
salah ketik lebih baik ditolak daripada diabaikan diam-diam. Tetapi portal
kampus menyebut hal yang sama dengan nama LMS:

    portal mengirim          backend mengenal
    ---------------------    -----------------
    section_ids              weeks
    model_provider + name    model
    user_id                  student_id
    student_name             (dicatat terpisah)

Sebelum ini kelimanya ditolak sekaligus, sehingga `POST /chat/ask` dari portal
dibalas 422 pada SETIAP pesan. Yang rusak bukan sebagian fitur — chat mati
total, dan gejalanya di layar mahasiswa hanya "Layanan AI sedang tidak
tersedia".

Muatan di bawah disalin apa adanya dari api/chatbot.php. Kalau berkas itu
berubah, uji ini harus ikut diperbarui — itulah gunanya.
"""
from __future__ import annotations

import pytest

from src.api.schemas import QueryRequest

# Persis seperti yang disusun api/chatbot.php.
MUATAN_PORTAL = {
    "question": "apa itu GROUP BY",
    "content_id": "class-12",
    "session_id": None,
    "section_ids": [1, 2],
    "model_provider": "openrouter",
    "model_name": "qwen-flash",
    "student_name": "Kevin",
    "user_id": 42,
}


class TestMuatanPortalDiterima:
    def test_tidak_ditolak(self) -> None:
        q = QueryRequest(**MUATAN_PORTAL)
        assert q.question == "apa itu GROUP BY"

    def test_setiap_medan_portal_dikenali(self) -> None:
        """Gagal satu per satu, supaya pesan galatnya menyebut medan mana."""
        for medan in ("section_ids", "model_provider", "model_name",
                      "student_name", "user_id"):
            QueryRequest(question="x", **{medan: MUATAN_PORTAL[medan]})


class TestKanonisasi:
    """Sisa kode hanya boleh mengenal SATU nama untuk satu hal."""

    def test_section_ids_jadi_minggu(self) -> None:
        q = QueryRequest(question="x", section_ids=[3, 1, 3])
        assert q.active_weeks() == [1, 3]

    def test_weeks_menang_atas_section_ids(self) -> None:
        q = QueryRequest(question="x", weeks=[5], section_ids=[1, 2])
        assert q.active_weeks() == [5]

    def test_week_tunggal_masih_dikenali(self) -> None:
        assert QueryRequest(question="x", week=4).active_weeks() == [4]

    def test_tanpa_minggu_kosong_bukan_meledak(self) -> None:
        """Dulu `active_weeks()` menyebut medan yang tidak ada dan melempar."""
        assert QueryRequest(question="x").active_weeks() == []

    def test_model_name_jadi_kunci_registry(self) -> None:
        q = QueryRequest(question="x", model_provider="openrouter", model_name="qwen")
        assert q.kunci_model() == "qwen"

    def test_model_menang_atas_model_name(self) -> None:
        q = QueryRequest(question="x", model="eksplisit", model_name="qwen")
        assert q.kunci_model() == "eksplisit"

    def test_tanpa_model_none(self) -> None:
        assert QueryRequest(question="x").kunci_model() is None

    @pytest.mark.parametrize("nilai, harap", [(42, "42"), ("abc", "abc"), (None, None)])
    def test_user_id_jadi_identitas(self, nilai, harap) -> None:
        assert QueryRequest(question="x", user_id=nilai).identitas_mahasiswa() == harap

    def test_student_id_menang_atas_user_id(self) -> None:
        q = QueryRequest(question="x", student_id="s1", user_id=42)
        assert q.identitas_mahasiswa() == "s1"


class TestSalahKetikTetapDitolak:
    """Longgar terhadap kosakata portal, bukan longgar terhadap apa pun.

    Kalau pemeriksaan ini ikut dilonggarkan, medan yang salah ketik akan
    diabaikan tanpa suara — dan `sectionids` yang luput satu garis bawah
    membuat pembatasan minggu hilang tanpa ada yang tahu.
    """

    @pytest.mark.parametrize("medan", ["sectionids", "section_id_", "modelname", "userid"])
    def test_medan_asing_tetap_422(self, medan: str) -> None:
        with pytest.raises(Exception) as galat:
            QueryRequest(question="x", **{medan: 1})
        assert "extra" in str(galat.value).lower() or "forbidden" in str(galat.value).lower()
