"""Document parser using Unstructured.io.

Converts uploaded documents (PDF, DOCX, PPTX, TXT, MD) into a list of
ParsedElement instances ready for the summarization layer.

Strategy:
- PDF: 'hi_res' to capture image base64 and table HTML structure.
- Other formats: auto-detected by Unstructured (`partition`).

Sequential text elements (Title, NarrativeText, ListItem, etc.) are merged
into a single ParsedElement until interrupted by a Table or Image. This keeps
related prose together so the chunker can split it semantically later.
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

from src.ingestion.validators import validate_file
from src.schemas import ElementType, ParsedElement
from src.utils.logger import logger

_PDF_EXTENSIONS = {"pdf"}


def parse_document(file_path: Path) -> list[ParsedElement]:
    """Parse a document into a list of ParsedElement.

    Args:
        file_path: Absolute path to the file. File must already be on disk;
            for FastAPI uploads, save to a temp file first.

    Returns:
        Ordered list of ParsedElement preserving document reading order.
        Empty list if document has no extractable content.

    Raises:
        FileValidationError: From validate_file().
        RuntimeError: If partition fails (corrupted file, OCR error, etc).
    """
    validate_file(file_path)
    logger.info("Parsing document: {}", file_path.name)


    is_pdf = file_path.suffix.lower() == ".pdf"
    partition_kwargs: dict = (
        {"strategy": "fast"}
        if is_pdf
        else {"strategy": "auto", "infer_table_structure": True}
    )

    try:
        elements: list[Element] = partition(
            filename=str(file_path),
            **partition_kwargs,
        )
    except Exception as exc:
        logger.exception("Partition failed for {}", file_path.name)
        raise RuntimeError(f"Failed to parse {file_path.name}: {exc}") from exc

    parsed = _assemble_elements(elements, source_file=file_path.name)

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
    """Choose partition arguments based on file type."""
    ext = file_path.suffix.lower().lstrip(".")
    if ext in _PDF_EXTENSIONS:
        return {
            "strategy": "hi_res",
            "infer_table_structure": True,
            "extract_images_in_pdf": True,
            "extract_image_block_types": ["Image"],
            "extract_image_block_to_payload": True,
        }
    return {
        "strategy": "auto",
        "infer_table_structure": True,
    }


def _assemble_elements(
    elements: list[Element], source_file: str
) -> list[ParsedElement]:
    """Walk raw Unstructured elements, batching text and emitting structured items.

    Tables and Images are emitted as their own ParsedElement. Everything else
    (Title, NarrativeText, ListItem, ...) is buffered as text and flushed when
    we hit a structural break.
    """
    parsed: list[ParsedElement] = []
    text_buffer: list[str] = []
    buffer_page: int | None = None

    def flush_text() -> None:
        nonlocal text_buffer, buffer_page
        if not text_buffer:
            return
        parsed.append(
            _make_text_element(text_buffer, source_file, buffer_page)
        )
        text_buffer = []
        buffer_page = None

    for el in elements:
        page = _page_number_of(el)

        if isinstance(el, UTable):
            flush_text()
            parsed.append(_make_table_element(el, source_file, page))
            continue

        if isinstance(el, UImage):
            flush_text()
            img = _make_image_element(el, source_file, page)
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
    """Read page number from an element's metadata, if available."""
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return None
    return getattr(metadata, "page_number", None)


def _make_text_element(
    buffer: list[str], source_file: str, page_number: int | None
) -> ParsedElement:
    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.TEXT,
        content="\n\n".join(buffer),
        source_file=source_file,
        page_number=page_number,
    )


def _make_table_element(
    table: UTable, source_file: str, page_number: int | None
) -> ParsedElement:
    raw_html: str | None = None
    metadata = getattr(table, "metadata", None)
    if metadata is not None:
        raw_html = getattr(metadata, "text_as_html", None)

    if raw_html is None:
        logger.warning(
            "Table on page {} of {} has no HTML representation; "
            "falling back to plain text",
            page_number,
            source_file,
        )

    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.TABLE,
        content=table.text or "",
        raw_html=raw_html,
        source_file=source_file,
        page_number=page_number,
    )


def _make_image_element(
    image: UImage, source_file: str, page_number: int | None
) -> ParsedElement | None:
    """Build ParsedElement from an Image. Returns None if no base64 payload."""
    metadata = getattr(image, "metadata", None)
    image_base64: str | None = None
    if metadata is not None:
        image_base64 = getattr(metadata, "image_base64", None)

    if not image_base64:
        logger.warning(
            "Image on page {} of {} has no base64 payload, skipping",
            page_number,
            source_file,
        )
        return None

    return ParsedElement(
        element_id=str(uuid.uuid4()),
        element_type=ElementType.IMAGE,
        content=image.text or "",
        image_base64=image_base64,
        source_file=source_file,
        page_number=page_number,
    )


def _count_by_type(elements: list[ParsedElement]) -> dict[ElementType, int]:
    counts = {t: 0 for t in ElementType}
    for el in elements:
        counts[el.element_type] += 1
    return counts