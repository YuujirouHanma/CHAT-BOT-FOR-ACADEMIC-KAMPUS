"""Document parser using Unstructured.io.

Supports: PDF, DOCX, PPTX, TXT, MD, CSV, XLSX, HTML, RTF, RST, EPUB, TSV.
Unstructured's partition() auto-detects file type.

Strategy:
- PDF: 'hi_res' if available (for table HTML + images), fallback to 'fast'.
- Other formats: 'auto' (Unstructured auto-detects).
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


def parse_document(
    file_path: Path,
    course: str | None = None,
    week: int | None = None,
) -> list[ParsedElement]:
    """Parse a document into a list of ParsedElement.

    Args:
        file_path: Path to the file on disk.
        course: Course identifier to stamp on every element.
        week: Week number to stamp on every element.
    """
    validate_indexable(file_path)
    logger.info("Parsing document: {}", file_path.name)

    partition_kwargs = _build_partition_kwargs(file_path)

    try:
        elements: list[Element] = partition(
            filename=str(file_path), **partition_kwargs
        )
    except Exception as exc:
        logger.exception("Partition failed for {}", file_path.name)
        raise RuntimeError(f"Failed to parse {file_path.name}: {exc}") from exc

    parsed = _assemble_elements(
        elements,
        source_file=file_path.name,
        course=course,
        week=week,
    )

    counts = _count_by_type(parsed)
    logger.info(
        "Parsed {}: {} text, {} tables, {} images",
        file_path.name,
        counts[ElementType.TEXT],
        counts[ElementType.TABLE],
        counts[ElementType.IMAGE],
    )
    return parsed


def _build_partition_kwargs(file_path: Path) -> dict[str, Any]:
    ext = file_path.suffix.lower().lstrip(".")
    if ext in _PDF_EXTENSIONS:
        # infer_table_structure omitted — Unstructured passes it internally for PDFs
        # and would raise "multiple values" error if we also pass it
        return {
            "strategy": "fast",
        }
    return {
        "strategy": "auto",
        "infer_table_structure": True,
    }


def _assemble_elements(
    elements: list[Element],
    source_file: str,
    course: str | None = None,
    week: int | None = None,
) -> list[ParsedElement]:
    parsed: list[ParsedElement] = []
    text_buffer: list[str] = []
    buffer_page: int | None = None

    def flush_text() -> None:
        nonlocal text_buffer, buffer_page
        if not text_buffer:
            return
        parsed.append(
            _make_text_element(text_buffer, source_file, buffer_page, course, week)
        )
        text_buffer = []
        buffer_page = None

    for el in elements:
        page = _page_number_of(el)

        if isinstance(el, UTable):
            flush_text()
            parsed.append(_make_table_element(el, source_file, page, course, week))
            continue

        if isinstance(el, UImage):
            flush_text()
            img = _make_image_element(el, source_file, page, course, week)
            if img is not None:
                parsed.append(img)
            continue

        text = (getattr(el, "text", "") or "").strip()
        if text:
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
    course: str | None, week: int | None,
) -> ParsedElement:
    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.TEXT,
        content="\n\n".join(buffer),
        source_file=source_file,
        page_number=page_number,
        course=course,
        week=week,
    )


def _make_table_element(
    table: UTable, source_file: str, page_number: int | None,
    course: str | None, week: int | None,
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
        course=course,
        week=week,
    )


def _make_image_element(
    image: UImage, source_file: str, page_number: int | None,
    course: str | None, week: int | None,
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
        course=course,
        week=week,
    )


def _count_by_type(elements: list[ParsedElement]) -> dict[ElementType, int]:
    counts = {t: 0 for t in ElementType}
    for el in elements:
        counts[el.element_type] += 1
    return counts