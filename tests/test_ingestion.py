"""Tests for the ingestion layer.

Validator tests are deterministic and run against the real filesystem.
Parser tests inject fake Unstructured elements (subclasses of real UImage
and UTable) so isinstance checks work naturally, without monkeypatching
builtins.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from unstructured.documents.elements import (
    Image as UImage,
    NarrativeText,
    Table as UTable,
    Title,
)

from src.ingestion.parser import _build_partition_kwargs, parse_document
from src.ingestion.validators import FileValidationError, validate_file, validate_indexable
from src.schemas import ElementType


def _meta(
    *,
    page_number: int | None = None,
    text_as_html: str | None = None,
    image_base64: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        page_number=page_number,
        text_as_html=text_as_html,
        image_base64=image_base64,
    )


def _fake_title(text: str, **meta_kw: Any) -> Title:
    el = Title(text=text)
    el.metadata = _meta(**meta_kw)
    return el


def _fake_narrative(text: str, **meta_kw: Any) -> NarrativeText:
    el = NarrativeText(text=text)
    el.metadata = _meta(**meta_kw)
    return el


def _fake_table(text: str, **meta_kw: Any) -> UTable:
    el = UTable(text=text)
    el.metadata = _meta(**meta_kw)
    return el


def _fake_image(text: str = "", **meta_kw: Any) -> UImage:
    el = UImage(text=text)
    el.metadata = _meta(**meta_kw)
    return el


def _patch_partition(
    monkeypatch: pytest.MonkeyPatch, elements: list[Any]
) -> None:
    from src.ingestion import parser as parser_mod
    monkeypatch.setattr(parser_mod, "partition", lambda **_: elements)


class TestValidator:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileValidationError, match="not found"):
            validate_file(tmp_path / "missing.pdf")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileValidationError, match="Not a regular file"):
            validate_file(tmp_path)

    def test_disallowed_extension_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "malware.exe"
        bad.write_bytes(b"MZ\x90\x00")
        with pytest.raises(FileValidationError, match="not indexable"):
            validate_indexable(bad)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        with pytest.raises(FileValidationError, match="empty"):
            validate_file(empty)

    def test_oversized_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import settings

        monkeypatch.setattr(settings, "max_upload_size_mb", 1, raising=False)
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * (2 * 1024 * 1024))

        with pytest.raises(FileValidationError, match="exceeds limit"):
            validate_file(big)

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        ok = tmp_path / "doc.txt"
        ok.write_text("hello world")
        validate_file(ok)


class TestParser:
    @staticmethod
    def _real_file(tmp_path: Path) -> Path:
        f = tmp_path / "doc.txt"
        f.write_text("dummy content for validator")
        return f

    def test_buffers_sequential_text_into_single_element(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elements = [
            _fake_title("Chapter 1", page_number=1),
            _fake_narrative("Statistics is important.", page_number=1),
            _fake_narrative("It is used in research.", page_number=1),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))

        assert len(result) == 1
        assert result[0].element_type == ElementType.TEXT
        assert "Chapter 1" in result[0].content
        assert "Statistics" in result[0].content
        assert "research" in result[0].content
        assert result[0].page_number == 1

    def test_page_change_breaks_text_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Teks dari halaman berbeda tidak boleh menyatu jadi satu elemen.

        Sebelum ini `flush_text()` hanya terpanggil saat bertemu tabel atau
        gambar. Slide kuliah kerap tidak punya keduanya, sehingga SATU PDF
        24 halaman menjadi SATU elemen bernomor halaman 1 — sitasi `sources`
        jadi tidak dapat diperiksa mahasiswa, dan potongan hasil chunking
        membentang lintas slide yang tidak berhubungan.
        """
        elements = [
            _fake_title("Percabangan", page_number=1),
            _fake_narrative("if, elif, else.", page_number=1),
            _fake_title("Perulangan", page_number=2),
            _fake_narrative("for dan while.", page_number=2),
            _fake_narrative("Contoh range().", page_number=3),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))

        assert len(result) == 3
        assert [e.page_number for e in result] == [1, 2, 3]
        assert "Percabangan" in result[0].content
        assert "Perulangan" in result[1].content
        assert "range()" in result[2].content
        # Isi halaman berbeda tidak bocor ke elemen tetangganya.
        assert "Perulangan" not in result[0].content

    def test_halaman_tanpa_nomor_tetap_menumpuk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sumber tanpa nomor halaman — transkrip audio — tetap satu elemen.

        Memecah pada `None` akan memecah tiap baris transkrip menjadi elemen
        sendiri, dan potongan sekecil itu kehilangan konteks saat dicari.
        """
        elements = [
            _fake_narrative("Bagian pertama rekaman.", page_number=None),
            _fake_narrative("Bagian kedua rekaman.", page_number=None),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))

        assert len(result) == 1
        assert "pertama" in result[0].content and "kedua" in result[0].content

    def test_table_breaks_text_buffer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elements = [
            _fake_narrative("Before table.", page_number=2),
            _fake_table(
                "A 1 B 2",
                page_number=2,
                text_as_html="<table><tr><td>A</td><td>1</td></tr></table>",
            ),
            _fake_narrative("After table.", page_number=2),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))

        assert [e.element_type for e in result] == [
            ElementType.TEXT,
            ElementType.TABLE,
            ElementType.TEXT,
        ]
        assert "<table>" in result[1].raw_html
        assert result[1].has_visual_payload() is True

    def test_image_with_base64_extracted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elements = [_fake_image(page_number=3, image_base64="aGVsbG8=")]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))

        assert len(result) == 1
        assert result[0].element_type == ElementType.IMAGE
        assert result[0].image_base64 == "aGVsbG8="
        assert result[0].page_number == 3

    def test_image_without_base64_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_partition(monkeypatch, [_fake_image(page_number=1, image_base64=None)])

        result = parse_document(self._real_file(tmp_path))
        assert result == []

    def test_each_element_has_unique_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elements = [
            _fake_narrative("One.", page_number=1),
            _fake_table("t", page_number=1, text_as_html="<table/>"),
            _fake_narrative("Two.", page_number=2),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))
        ids = [e.element_id for e in result]
        assert len(ids) == len(set(ids))

    def test_empty_text_elements_filtered_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elements = [
            _fake_narrative(""),
            _fake_narrative("   "),
            _fake_narrative("Real content.", page_number=1),
        ]
        _patch_partition(monkeypatch, elements)

        result = parse_document(self._real_file(tmp_path))
        assert len(result) == 1
        assert result[0].content == "Real content."

    def test_invalid_file_propagates_validation_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileValidationError):
            parse_document(tmp_path / "nope.txt")

    def test_partition_exception_wrapped_in_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.ingestion import parser as parser_mod

        def boom(**_: Any) -> list[Any]:
            raise ValueError("corrupted PDF")

        monkeypatch.setattr(parser_mod, "partition", boom)

        with pytest.raises(RuntimeError, match="Failed to parse"):
            parse_document(self._real_file(tmp_path))


class TestPartitionKwargs:
    """Regression guard: partition() auto-routes non-PDF formats to their own
    partitioner, which already sets infer_table_structure. Passing it again
    raises 'got multiple values for keyword argument' and breaks docx/pptx/xlsx.
    """

    def test_pdf_uses_fast_strategy_without_infer_table_structure(
        self, tmp_path: Path
    ) -> None:
        kwargs = _build_partition_kwargs(tmp_path / "materi.pdf")
        assert kwargs == {"strategy": "fast"}
        assert "infer_table_structure" not in kwargs

    @pytest.mark.parametrize("ext", ["docx", "pptx", "xlsx", "txt", "md", "csv", "html"])
    def test_non_pdf_never_passes_infer_table_structure(
        self, tmp_path: Path, ext: str
    ) -> None:
        kwargs = _build_partition_kwargs(tmp_path / f"materi.{ext}")
        assert "infer_table_structure" not in kwargs

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        assert _build_partition_kwargs(tmp_path / "SLIDE.PPTX") == {"strategy": "auto"}
        assert _build_partition_kwargs(tmp_path / "REPORT.PDF") == {"strategy": "fast"}


class TestNltkDataCheck:
    """unstructured downloads NLTK data at runtime when missing and often gets
    HTTP 403 in containers — breaking pptx/docx parsing. Startup must surface it."""

    def test_reports_nothing_when_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import unstructured.nlp.tokenize as tok

        from src.ingestion import nltk_data

        monkeypatch.setattr(tok, "check_for_nltk_package", lambda **kw: True)
        assert nltk_data.missing_packages() == []

    def test_reports_each_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import unstructured.nlp.tokenize as tok

        from src.ingestion import nltk_data

        monkeypatch.setattr(tok, "check_for_nltk_package", lambda **kw: False)
        missing = nltk_data.missing_packages()

        assert "tokenizers/punkt_tab" in missing
        assert "taggers/averaged_perceptron_tagger_eng" in missing

    def test_warn_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import unstructured.nlp.tokenize as tok

        from src.ingestion import nltk_data

        def boom(**kw):
            raise RuntimeError("lookup exploded")

        monkeypatch.setattr(tok, "check_for_nltk_package", boom)
        nltk_data.warn_if_missing()  # startup must not crash on a check failure


class TestOcrFallback:
    """A scanned/image PDF parses without error but yields ~no text under `fast`.
    parse_document must retry with OCR, and degrade gracefully when OCR is
    unavailable (e.g. Tesseract not installed) rather than crash the upload.
    """

    @staticmethod
    def _pdf(tmp_path: Path) -> Path:
        f = tmp_path / "scanned.pdf"
        f.write_bytes(b"%PDF-1.4 dummy bytes for validator")
        return f

    @staticmethod
    def _patch_sequence(
        monkeypatch: pytest.MonkeyPatch, returns: list[Any]
    ) -> list[dict[str, Any]]:
        """Make partition() return/raise each item in `returns` on successive
        calls. An item that is an Exception instance is raised. Records kwargs."""
        from src.ingestion import parser as parser_mod

        seq = iter(returns)
        calls: list[dict[str, Any]] = []

        def fake(**kwargs: Any) -> list[Any]:
            calls.append(kwargs)
            item = next(seq)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(parser_mod, "partition", fake)
        return calls

    def test_empty_pdf_retries_with_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._patch_sequence(
            monkeypatch,
            [
                [_fake_narrative("")],  # fast → no text
                [_fake_narrative("Teks hasil OCR dari slide.", page_number=1)],  # ocr_only
            ],
        )

        result = parse_document(self._pdf(tmp_path))

        assert len(calls) == 2
        assert calls[1]["strategy"] == "ocr_only"
        assert len(result) == 1
        assert "OCR" in result[0].content

    def test_ocr_unavailable_degrades_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._patch_sequence(
            monkeypatch,
            [
                [_fake_narrative("")],  # fast → no text
                RuntimeError("tesseract is not installed"),  # ocr attempt fails
            ],
        )

        result = parse_document(self._pdf(tmp_path))  # must NOT raise

        assert len(calls) == 2
        assert result == []

    def test_pdf_with_text_does_not_trigger_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._patch_sequence(
            monkeypatch,
            [[_fake_narrative("Materi lengkap yang teksnya bisa diekstrak.", page_number=1)]],
        )

        result = parse_document(self._pdf(tmp_path))

        assert len(calls) == 1  # no OCR retry
        assert len(result) == 1

    def test_empty_non_pdf_does_not_trigger_ocr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("real content for validator")
        calls = self._patch_sequence(monkeypatch, [[_fake_narrative("")]])

        result = parse_document(txt)

        assert len(calls) == 1  # non-PDF: no OCR retry even when empty
        assert result == []