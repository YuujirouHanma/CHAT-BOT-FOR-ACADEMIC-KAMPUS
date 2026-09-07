"""Validasi dosen atas jawaban AI.

Ketepatan pengambilan materi belum terjamin, jadi harus ada jalan bagi dosen
menyatakan mana yang keliru — sekaligus mengumpulkan data berlabel manusia yang
membuat klaim akurasi di makalah dapat diuji.
"""
from __future__ import annotations

import json

import pytest

from src import validation
from src.hitl.logger import CONVERSATION_LOG, log_dir
from src.tenancy import TenantScopeError

A = "kampus-a"
B = "kampus-b"


@pytest.fixture(autouse=True)
def _isolasi(tmp_path, monkeypatch):
    """Arahkan penyimpanan validasi & log ke direktori sementara."""
    monkeypatch.setattr(validation, "VALIDATION_DIR", tmp_path / "validation")
    import src.hitl.logger as hl

    monkeypatch.setattr(hl, "HITL_DIR", tmp_path / "hitl")
    yield


def _tulis_interaksi(tenant: str, iid: str, content_id: str, question: str) -> None:
    """Tiru satu baris log interaksi seperti yang ditulis pipeline."""
    path = log_dir(tenant) / CONVERSATION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "interaction_id": iid,
            "timestamp": "2026-08-20T10:00:00",
            "tenant_id": tenant,
            "content_id": content_id,
            "question": question,
            "answer": "Jawaban AI untuk " + question,
            "sources": [],
        }, ensure_ascii=False) + "\n")


class TestPenjagaan:
    def test_tanpa_tenant_ditolak(self) -> None:
        with pytest.raises(TenantScopeError):
            validation.list_interactions(tenant_id="")

    def test_putusan_tidak_dikenal_ditolak(self) -> None:
        with pytest.raises(ValueError, match="tidak dikenal"):
            validation.record_verdict(
                validation.ValidationRecord("i1", "mungkin"), tenant_id=A,
            )

    def test_menilai_salah_wajib_beralasan(self) -> None:
        # Menandai salah tanpa menjelaskan di mana salahnya tidak menolong
        # mahasiswa maupun analisis datanya nanti.
        for putusan in ("tidak_sesuai", "perlu_perbaikan"):
            with pytest.raises(ValueError, match="Catatan wajib"):
                validation.record_verdict(
                    validation.ValidationRecord("i1", putusan, catatan="  "),
                    tenant_id=A,
                )

    def test_menilai_sesuai_boleh_tanpa_catatan(self) -> None:
        hasil = validation.record_verdict(
            validation.ValidationRecord("i1", "sesuai"), tenant_id=A,
        )
        assert hasil["verdict"] == "sesuai"


class TestAntreanDosen:
    def test_hanya_yang_belum_dinilai(self) -> None:
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "apa itu JOIN")
        _tulis_interaksi(A, "i2", "sbd-minggu-2", "beda for dan while")
        assert len(validation.list_interactions(tenant_id=A)) == 2

        validation.record_verdict(
            validation.ValidationRecord("i1", "sesuai"), tenant_id=A,
        )
        sisa = validation.list_interactions(tenant_id=A)
        assert [r["interaction_id"] for r in sisa] == ["i2"]

    def test_semua_membawa_putusannya(self) -> None:
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "apa itu JOIN")
        validation.record_verdict(
            validation.ValidationRecord(
                "i1", "tidak_sesuai", catatan="Dijawab pakai materi SQL, "
                "padahal yang ditanya perulangan.", dosen_id="d1",
            ),
            tenant_id=A,
        )
        semua = validation.list_interactions(tenant_id=A, only_pending=False)
        assert semua[0]["verdict"] == "tidak_sesuai"
        assert "perulangan" in semua[0]["catatan"]

    def test_disaring_per_mata_kuliah(self) -> None:
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "JOIN")
        _tulis_interaksi(A, "i2", "kka-minggu-1", "apa itu AI")
        hanya_sbd = validation.list_interactions(tenant_id=A, course_id="sbd")
        assert [r["interaction_id"] for r in hanya_sbd] == ["i1"]

    def test_putusan_terbaru_yang_berlaku(self) -> None:
        # Dosen boleh berubah pikiran; riwayatnya tetap tersimpan.
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "JOIN")
        validation.record_verdict(
            validation.ValidationRecord("i1", "sesuai"), tenant_id=A,
        )
        validation.record_verdict(
            validation.ValidationRecord("i1", "perlu_perbaikan",
                                        catatan="Contohnya kurang"),
            tenant_id=A,
        )
        semua = validation.list_interactions(tenant_id=A, only_pending=False)
        assert semua[0]["verdict"] == "perlu_perbaikan"


class TestIsolasiAntarTenant:
    def test_putusan_tenant_lain_tidak_terlihat(self) -> None:
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "milik A")
        _tulis_interaksi(B, "i1", "sbd-minggu-1", "milik B")
        validation.record_verdict(
            validation.ValidationRecord("i1", "sesuai"), tenant_id=A,
        )
        # interaction_id yang sama, tenant berbeda: putusan A tidak boleh
        # membuat jawaban milik B ikut dianggap sudah dinilai.
        antrean_b = validation.list_interactions(tenant_id=B)
        assert [r["interaction_id"] for r in antrean_b] == ["i1"]
        assert validation.latest_verdicts(tenant_id=B) == {}


class TestRingkasan:
    def test_akurasi_none_saat_belum_ada_yang_dinilai(self) -> None:
        # "Belum diukur" tidak boleh terbaca sebagai "akurasinya nol".
        _tulis_interaksi(A, "i1", "sbd-minggu-1", "JOIN")
        s = validation.stats(tenant_id=A)
        assert s["akurasi"] is None
        assert s["belum_dinilai"] == 1

    def test_akurasi_dihitung_dari_yang_dinilai_saja(self) -> None:
        for i in range(4):
            _tulis_interaksi(A, f"i{i}", "sbd-minggu-1", f"tanya {i}")
        validation.record_verdict(
            validation.ValidationRecord("i0", "sesuai"), tenant_id=A)
        validation.record_verdict(
            validation.ValidationRecord("i1", "sesuai"), tenant_id=A)
        validation.record_verdict(
            validation.ValidationRecord("i2", "tidak_sesuai", catatan="salah materi"),
            tenant_id=A)
        s = validation.stats(tenant_id=A)
        assert s["total_jawaban"] == 4
        assert s["sudah_dinilai"] == 3
        assert s["belum_dinilai"] == 1
        assert s["akurasi"] == pytest.approx(2 / 3, abs=0.001)
