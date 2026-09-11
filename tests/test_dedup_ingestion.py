"""Uji deteksi berkas rangkap pada jalur ingestion, dan uji-kering migrasi tenant.

Dua cacat data yang diuji di sini:

(A) Titik index warisan tanpa `tenant_id` tidak terjangkau kueri apa pun.
    Yang diuji di bawah hanyalah janji terpenting skrip migrasinya: mode
    uji-kering benar-benar tidak menulis apa pun.

(B) Berkas yang sama diunggah berulang kali dan setiap salinan menghasilkan
    potongan yang identik, sehingga slot bukti terpakai untuk mengulang
    kalimat yang sama. Yang diuji: salinan ditolak, materi yang berbeda tidak
    ikut tertolak, dan cakupannya berhenti di satu direktori materi.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts import migrate_tenant_id
from src.ingestion.dedup import find_duplicate, hash_file
from src.ingestion.validators import (
    DuplicateFileError,
    FileValidationError,
    validate_file,
    validate_indexable,
)

_ISI = b"Materi minggu 2: percabangan dan perulangan.\n" * 40
_ISI_LAIN = b"Materi minggu 3: struktur data dan kompleksitas.\n" * 40


def _tulis(path: Path, isi: bytes, *, umur_detik: float = 0.0) -> Path:
    """Tulis berkas dengan mtime yang dapat diatur.

    Urutan "siapa lebih dulu ada" ditentukan mtime, jadi uji tidak boleh
    bergantung pada urutan penulisan — pada berkas sistem berresolusi kasar,
    dua penulisan beruntun bisa memiliki mtime yang persis sama.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(isi)
    if umur_detik:
        stempel = path.stat().st_mtime - umur_detik
        os.utime(path, (stempel, stempel))
    return path


class TestSidikBerkas:
    def test_isi_sama_menghasilkan_sidik_sama(self, tmp_path: Path) -> None:
        a = _tulis(tmp_path / "a.pdf", _ISI)
        b = _tulis(tmp_path / "b.pdf", _ISI)
        assert hash_file(a) == hash_file(b)

    def test_isi_beda_menghasilkan_sidik_beda(self, tmp_path: Path) -> None:
        a = _tulis(tmp_path / "a.pdf", _ISI)
        b = _tulis(tmp_path / "b.pdf", _ISI_LAIN)
        assert hash_file(a) != hash_file(b)


class TestDeteksiRangkap:
    """Inti perbaikan (B): salinan tidak boleh ikut masuk index."""

    def test_salinan_ditolak(self, tmp_path: Path) -> None:
        materi = tmp_path / "sbd-minggu-2"
        _tulis(materi / "asli.pdf", _ISI, umur_detik=60)
        salinan = _tulis(materi / "salinan.pdf", _ISI)

        with pytest.raises(DuplicateFileError, match="asli.pdf"):
            validate_indexable(salinan)

    def test_berkas_pertama_tetap_lolos(self, tmp_path: Path) -> None:
        """Yang tertua yang bertahan — kalau tidak, tak satu pun salinan terindeks."""
        materi = tmp_path / "sbd-minggu-2"
        asli = _tulis(materi / "asli.pdf", _ISI, umur_detik=60)
        _tulis(materi / "salinan.pdf", _ISI)

        validate_indexable(asli)

    def test_dari_banyak_salinan_hanya_satu_lolos(self, tmp_path: Path) -> None:
        """Meniru kasus nyata: satu berkas, puluhan salinan di satu direktori."""
        materi = tmp_path / "_uploads"
        berkas = [
            _tulis(materi / f"{i:02d}_lesson.pdf", _ISI, umur_detik=(20 - i) * 10)
            for i in range(20)
        ]

        lolos = []
        for f in berkas:
            try:
                validate_indexable(f)
                lolos.append(f.name)
            except DuplicateFileError:
                pass

        assert lolos == ["00_lesson.pdf"]

    def test_berkas_berbeda_tidak_ditolak(self, tmp_path: Path) -> None:
        materi = tmp_path / "sbd-minggu-2"
        _tulis(materi / "bab1.pdf", _ISI, umur_detik=60)
        lain = _tulis(materi / "bab2.pdf", _ISI_LAIN)

        validate_indexable(lain)

    def test_ukuran_sama_isi_beda_bukan_rangkap(self, tmp_path: Path) -> None:
        """Penyaring ukuran hanyalah pemercepat — keputusannya tetap dari hash."""
        materi = tmp_path / "sbd-minggu-2"
        _tulis(materi / "a.pdf", b"x" * 500, umur_detik=60)
        b = _tulis(materi / "b.pdf", b"y" * 500)

        validate_indexable(b)

    def test_berkas_sama_untuk_minggu_berbeda_tetap_diindeks(
        self, tmp_path: Path
    ) -> None:
        """Keputusan rancangan: cakupan duplikat berhenti di satu direktori materi.

        Slide yang dipakai ulang di minggu lain HARUS tetap terindeks — kueri
        mahasiswa disaring per minggu, jadi menolaknya membuat minggu itu tidak
        punya materi sama sekali.
        """
        _tulis(tmp_path / "sbd-minggu-2" / "slide.pdf", _ISI, umur_detik=60)
        minggu3 = _tulis(tmp_path / "sbd-minggu-3" / "slide.pdf", _ISI)

        validate_indexable(minggu3)

    def test_berkas_sama_milik_tenant_lain_tetap_diindeks(self, tmp_path: Path) -> None:
        """Isolasi tenant sudah bersifat fisik: direktori berbeda, cakupan berbeda."""
        _tulis(tmp_path / "kampus-a" / "sbd-minggu-2" / "slide.pdf", _ISI, umur_detik=60)
        kampus_b = _tulis(tmp_path / "kampus-b" / "sbd-minggu-2" / "slide.pdf", _ISI)

        validate_indexable(kampus_b)

    def test_media_juga_diperiksa(self, tmp_path: Path) -> None:
        """Jalur transkripsi lewat `validate_file`, dan sama mahalnya untuk diulang."""
        materi = tmp_path / "sbd-minggu-3"
        _tulis(materi / "kuliah.mp3", _ISI, umur_detik=60)
        salinan = _tulis(materi / "kuliah-copy.mp3", _ISI)

        with pytest.raises(DuplicateFileError):
            validate_file(salinan)

    def test_berkas_kosong_tetap_galat_kosong(self, tmp_path: Path) -> None:
        """Dua berkas kosong bukan "rangkap" — pesannya harus tetap "empty"."""
        materi = tmp_path / "sbd-minggu-2"
        _tulis(materi / "a.txt", b"", umur_detik=60)
        b = _tulis(materi / "b.txt", b"")

        with pytest.raises(FileValidationError, match="empty"):
            validate_file(b)

    def test_rangkap_adalah_galat_validasi(self, tmp_path: Path) -> None:
        """Rute unggah dan indeks massal hanya menangkap `FileValidationError`.

        Kalau pertalian ini putus, duplikat berubah dari 400 yang rapi menjadi
        500 di rute unggah dan menggagalkan seluruh batch di rute indeks massal.
        """
        assert issubclass(DuplicateFileError, FileValidationError)

    def test_direktori_tak_terbaca_tidak_menggagalkan_unggahan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pemeriksaan duplikat adalah penghematan, bukan syarat kebenaran."""
        berkas = _tulis(tmp_path / "materi" / "a.pdf", _ISI)

        def _meledak(self: Path) -> Any:
            raise OSError("direktori tidak terbaca")

        monkeypatch.setattr(Path, "iterdir", _meledak)
        assert find_duplicate(berkas) is None


class TestUjiKeringMigrasi:
    """Perbaikan (A): mode uji-kering wajib benar-benar tidak menulis apa pun."""

    @staticmethod
    def _klien(titik: list[tuple[Any, dict]]) -> MagicMock:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.scroll.return_value = (
            [MagicMock(id=pid, payload=payload) for pid, payload in titik],
            None,
        )
        return client

    def test_uji_kering_tidak_menulis(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = self._klien([
            (1, {"source_file": "bab1.pdf"}),                  # yatim
            (2, {"source_file": "bab2.pdf", "tenant_id": ""}),  # yatim juga
            (3, {"source_file": "bab3.pdf", "tenant_id": "kampus-a"}),
        ])
        monkeypatch.setattr(migrate_tenant_id, "_buat_klien", lambda: client)
        monkeypatch.setattr(
            migrate_tenant_id.tenant_store, "get", lambda _tid: object()
        )

        assert migrate_tenant_id.main(["kampus-a"]) == 0

        client.set_payload.assert_not_called()
        client.upsert.assert_not_called()
        client.delete.assert_not_called()
        client.overwrite_payload.assert_not_called()

    def test_apply_hanya_mencap_titik_yatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Titik yang sudah bertenant tidak boleh ikut terkirim ke set_payload."""
        client = self._klien([
            (1, {"source_file": "bab1.pdf"}),
            (2, {"source_file": "bab2.pdf", "tenant_id": "kampus-a"}),
            (3, {"source_file": "bab3.pdf", "tenant_id": None}),
        ])
        monkeypatch.setattr(migrate_tenant_id, "_buat_klien", lambda: client)
        monkeypatch.setattr(
            migrate_tenant_id.tenant_store, "get", lambda _tid: object()
        )

        assert migrate_tenant_id.main(["kampus-a", "--apply"]) == 0

        client.set_payload.assert_called_once()
        kwargs = client.set_payload.call_args.kwargs
        assert kwargs["points"] == [1, 3]
        assert kwargs["payload"] == {"tenant_id": "kampus-a"}

    def test_tenant_tak_terdaftar_ditolak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sasaran harus disebut DAN sungguh ada — tidak ada tebakan diam-diam."""
        client = self._klien([(1, {"source_file": "bab1.pdf"})])
        monkeypatch.setattr(migrate_tenant_id, "_buat_klien", lambda: client)
        monkeypatch.setattr(migrate_tenant_id.tenant_store, "get", lambda _tid: None)

        assert migrate_tenant_id.main(["kampus-hantu"]) == 1
        client.scroll.assert_not_called()
        client.set_payload.assert_not_called()

    def test_apply_dan_dry_run_bersamaan_ditolak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = self._klien([(1, {})])
        monkeypatch.setattr(migrate_tenant_id, "_buat_klien", lambda: client)
        monkeypatch.setattr(
            migrate_tenant_id.tenant_store, "get", lambda _tid: object()
        )

        assert migrate_tenant_id.main(["kampus-a", "--apply", "--dry-run"]) == 2
        client.set_payload.assert_not_called()
