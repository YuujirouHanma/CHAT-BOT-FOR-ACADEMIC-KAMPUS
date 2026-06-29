"""Chunking layer.

Splits ParsedElement instances into Chunks ready for embedding.

Strategy:
- TEXT elements: split into multiple chunks using LlamaIndex SentenceSplitter,
  respecting sentence boundaries. Configurable chunk_size and chunk_overlap.
- TABLE / IMAGE elements: kept as a single chunk. Their `summary` is the
  embeddable text; raw_html / image_base64 are passed through for retrieval.
  We do NOT split a 3-sentence summary — it's already condensed.

A non-text element without any embeddable text (e.g. image whose
summarization failed and which has no OCR text) is dropped with a warning.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence

from llama_index.core.node_parser import SentenceSplitter

from src.config import settings
from src.schemas import Chunk, ElementType, ParsedElement
from src.utils.logger import logger


class Chunker:
    """Splits ParsedElements into Chunks.

    Construct once per pipeline run; safe to reuse across documents.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

    def chunk(self, elements: Sequence[ParsedElement]) -> list[Chunk]:
        """Convert all elements to chunks. Document order is preserved.

        Args:
            elements: Output of `enrich_elements()`, typically.

        Returns:
            Flat list of chunks across all input elements.
        """
        chunks: list[Chunk] = []
        for el in elements:
            if el.element_type == ElementType.TEXT:
                chunks.extend(self._chunk_text(el))
            else:
                non_text_chunk = self._chunk_non_text(el)
                if non_text_chunk is not None:
                    chunks.append(non_text_chunk)

        logger.info(
            "Produced {} chunks from {} elements",
            len(chunks),
            len(elements),
        )
        return chunks

    def _chunk_text(self, element: ParsedElement) -> list[Chunk]:
        text = (element.content or "").strip()
        if not text:
            return []

        parts = self._splitter.split_text(text)
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=part,
                parent_element_id=element.element_id,
                element_type=ElementType.TEXT,
                source_file=element.source_file,
                page_number=element.page_number,
                chunk_index=idx,
                course=element.course,
                week=element.week,
            )
            for idx, part in enumerate(parts)
            if part.strip()
        ]

    def _chunk_non_text(self, element: ParsedElement) -> Chunk | None:
        text = element.embeddable_text().strip()
        if not text:
            logger.warning(
                "Element {} ({}) has no embeddable text, dropping",
                element.element_id,
                element.element_type.value,
            )
            return None

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=text,
            parent_element_id=element.element_id,
            element_type=element.element_type,
            source_file=element.source_file,
            page_number=element.page_number,
            chunk_index=0,
            course=element.course,
            week=element.week,
            raw_html=element.raw_html,
            image_base64=element.image_base64,
        )