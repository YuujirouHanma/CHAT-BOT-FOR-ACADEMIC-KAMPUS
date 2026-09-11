"""Document parser using Unstructured.io.

Supports: PDF, DOCX, PPTX, TXT, MD, CSV, XLSX, HTML, RTF, RST, EPUB, TSV.
Unstructured's partition() auto-detects file type.

Strategy:
- PDF: 'fast' (pdfminer, no OCR/Poppler needed — see feedback-parser-strategy).
- Other formats (docx, pptx, xlsx, ...): let partition() auto-route to the
  format-specific partitioner with its own defaults. We deliberately do NOT
  pass infer_table_structure: partition() already forwards it internally, so
  passing it again raises "got multiple values for keyword argument".
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from unstructured.documents.elements import (
    Element,
    Image as UImage,
    Table as UTable,
)
from unstructured.partition.auto import partition

from src.ingestion.validators import validate_indexable
from src.schemas import ElementType, ParsedElement
from src.utils.logger import logger

_PDF_EXTENSIONS = {"pdf"}

# Below this many extracted text chars, a PDF is treated as "no text" — almost
# always a scanned/image-based PDF whose text lives in images. `fast` (pdfminer)
# does no OCR, so we retry with OCR. See feedback-parser-strategy.
_MIN_PDF_TEXT_CHARS = 20


def parse_document(
    file_path: Path,
    content_id: str | None = None,
) -> list[ParsedElement]:
    """Parse a document into a list of ParsedElement.

    Args:
        file_path: Path to the file on disk.
        content_id: Content identifier to stamp on every element.
    """
    validate_indexable(file_path)
    logger.info("Parsing document: {}", file_path.name)

    parsed = _partition_and_assemble(
        file_path, _build_partition_kwargs(file_path), content_id
    )

    # OCR fallback: a scanned/image PDF parses without error but yields ~no text.
    # Retry with strategy="ocr_only" so it becomes searchable instead of empty.
    if _is_pdf(file_path) and _text_char_count(parsed) < _MIN_PDF_TEXT_CHARS:
        logger.warning(
            "PDF '{}' produced only {} text chars with fast parse — likely "
            "scanned/image-based; retrying with OCR (strategy=ocr_only)",
            file_path.name,
            _text_char_count(parsed),
        )
        ocr_parsed = _try_ocr(file_path, content_id)
        if ocr_parsed is not None and _text_char_count(ocr_parsed) > _text_char_count(parsed):
            parsed = ocr_parsed

    counts = _count_by_type(parsed)
    logger.info(
        "Parsed {}: {} text, {} tables, {} images",
        file_path.name,
        counts[ElementType.TEXT],
        counts[ElementType.TABLE],
        counts[ElementType.IMAGE],
    )
    return parsed


def _partition_and_assemble(
    file_path: Path, partition_kwargs: dict[str, Any], content_id: str | None,
) -> list[ParsedElement]:
    """Run partition() with the given kwargs and assemble ParsedElements.

    A partition failure is fatal (wrapped in RuntimeError) — the caller cannot
    recover from a corrupt/unreadable file.
    """
    try:
        elements: list[Element] = partition(filename=str(file_path), **partition_kwargs)
    except Exception as exc:
        logger.exception("Partition failed for {}", file_path.name)
        raise RuntimeError(f"Failed to parse {file_path.name}: {exc}") from exc

    return _assemble_elements(
        elements, source_file=file_path.name, content_id=content_id,
    )


def _try_ocr(file_path: Path, content_id: str | None) -> list[ParsedElement] | None:
    """Re-parse a PDF with OCR. Returns None (not raising) when OCR is
    unavailable — e.g. Tesseract/Poppler not installed — so a scanned PDF
    degrades to an empty parse instead of crashing the upload.
    """
    try:
        elements: list[Element] = partition(
            filename=str(file_path),
            strategy="ocr_only",
            languages=["ind", "eng"],
        )
    except Exception as exc:
        logger.warning(
            "OCR fallback unavailable for {} ({}). To read scanned PDFs, install "
            "Tesseract (tesseract-ocr + tesseract-ocr-ind) and Poppler in the "
            "runtime image.",
            file_path.name,
            exc,
        )
        return None

    return _assemble_elements(
        elements, source_file=file_path.name, content_id=content_id,
    )


def _is_pdf(file_path: Path) -> bool:
    return file_path.suffix.lower().lstrip(".") in _PDF_EXTENSIONS


def _text_char_count(parsed: list[ParsedElement]) -> int:
    return sum(len(e.content) for e in parsed if e.element_type == ElementType.TEXT)


def _build_partition_kwargs(file_path: Path) -> dict[str, Any]:
    ext = file_path.suffix.lower().lstrip(".")
    if ext in _PDF_EXTENSIONS:
        # infer_table_structure omitted — Unstructured passes it internally for PDFs
        # and would raise "multiple values" error if we also pass it.
        return {
            "strategy": "fast",
        }
    # For docx/pptx/xlsx/etc. partition() auto-routes to the format-specific
    # partitioner, which already sets infer_table_structure (default True, so
    # tables still get text_as_html). Passing it here again would raise
    # "got multiple values for keyword argument 'infer_table_structure'".
    return {
        "strategy": "auto",
    }


def _assemble_elements(
    elements: list[Element],
    source_file: str,
    content_id: str | None = None,
) -> list[ParsedElement]:
    parsed: list[ParsedElement] = []
    text_buffer: list[str] = []
    buffer_page: int | None = None

    def flush_text() -> None:
        nonlocal text_buffer, buffer_page
        if not text_buffer:
            return
        parsed.append(
            _make_text_element(text_buffer, source_file, buffer_page, content_id)
        )
        text_buffer = []
        buffer_page = None

    for el in elements:
        page = _page_number_of(el)

        if isinstance(el, UTable):
            flush_text()
            parsed.append(_make_table_element(el, source_file, page, content_id))
            continue

        if isinstance(el, UImage):
            flush_text()
            img = _make_image_element(el, source_file, page, content_id)
            if img is not None:
                parsed.append(img)
            continue

        text = (getattr(el, "text", "") or "").strip()
        if not text:
            continue

        # Ganti halaman = batas elemen. Tanpa ini seluruh dokumen menumpuk
        # menjadi SATU elemen: `flush_text()` hanya terpanggil saat bertemu
        # tabel atau gambar, dan slide kuliah kerap tidak punya keduanya.
        # Akibatnya dua hal sekaligus — nomor halaman seluruh dokumen tercatat
        # sebagai halaman pertama sehingga sitasi `sources` tidak dapat
        # diperiksa mahasiswa, dan potongan hasil chunking membentang lintas
        # slide yang tidak berhubungan sehingga pencarian kehilangan ketepatan.
        if buffer_page is not None and page is not None and page != buffer_page:
            flush_text()

        text_buffer.append(text)
        if buffer_page is None:
            buffer_page = page

    flush_text()
    return parsed


def _page_number_of(element: Element) -> int | None:
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return None
    return getattr(metadata, "page_number", None)


def _make_text_element(
    buffer: list[str], source_file: str, page_number: int | None,
    content_id: str | None,
) -> ParsedElement:
    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.TEXT,
        content="\n\n".join(buffer),
        source_file=source_file,
        page_number=page_number,
        content_id=content_id,
    )


def _make_table_element(
    table: UTable, source_file: str, page_number: int | None,
    content_id: str | None,
) -> ParsedElement:
    raw_html: str | None = None
    metadata = getattr(table, "metadata", None)
    if metadata is not None:
        raw_html = getattr(metadata, "text_as_html", None)

    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.TABLE,
        content=table.text or "",
        raw_html=raw_html,
        source_file=source_file,
        page_number=page_number,
        content_id=content_id,
    )


def _make_image_element(
    image: UImage, source_file: str, page_number: int | None,
    content_id: str | None,
) -> ParsedElement | None:
    metadata = getattr(image, "metadata", None)
    image_base64: str | None = None
    if metadata is not None:
        image_base64 = getattr(metadata, "image_base64", None)

    if not image_base64:
        logger.warning(
            "Image on page {} of {} has no base64 payload, skipping",
            page_number, source_file,
        )
        return None

    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.IMAGE,
        content=image.text or "",
        image_base64=image_base64,
        source_file=source_file,
        page_number=page_number,
        content_id=content_id,
    )


def _count_by_type(elements: list[ParsedElement]) -> dict[ElementType, int]:
    counts = {t: 0 for t in ElementType}
    for el in elements:
        counts[el.element_type] += 1
    return counts
