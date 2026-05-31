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

from src.ingestion.parser import parse_document
from src.ingestion.validators import FileValidationError, validate_file
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
        with pytest.raises(FileValidationError, match="not allowed"):
            validate_file(bad)

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