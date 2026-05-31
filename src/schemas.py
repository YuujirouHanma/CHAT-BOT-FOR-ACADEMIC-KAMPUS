"""Shared data schemas for the RAG pipeline.

ParsedElement is the canonical representation of a piece of content
extracted from a document — text, table, or image.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ParsedElement(BaseModel):
    """One piece of content extracted by the parser.

    For TEXT: `content` holds raw text. No `raw_html` or `image_base64`.
    For TABLE: `content` holds a short caption (if any). `raw_html` holds the table.
    For IMAGE: `content` is empty. `image_base64` holds the image data.

    After the summarization step, `summary` is populated for tables and images.
    The `summary` (plus content for text) is what gets embedded.
    """

    element_id: str
    element_type: ElementType
    content: str = ""
    raw_html: str | None = None
    image_base64: str | None = None
    summary: str | None = None

    source_file: str
    page_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def embeddable_text(self) -> str:
        """Return the text that should be sent to the embedding model."""
        if self.element_type == ElementType.TEXT:
            return self.content
        if self.summary:
            return self.summary
        return self.content

    def has_visual_payload(self) -> bool:
        """Whether this element carries non-text data the LLM may need at generation."""
        return self.element_type in (ElementType.TABLE, ElementType.IMAGE)


class Chunk(BaseModel):
    """One unit of text + metadata, ready for embedding and storage.

    For TEXT chunks: produced by splitting a long ParsedElement.content.
    Multiple Chunks can share a parent_element_id (with different chunk_index).

    For TABLE/IMAGE chunks: one-to-one with parent ParsedElement; `text` is the
    summary produced earlier. Visual payloads (raw_html, image_base64) travel
    with the chunk so the LLM can read the original data at generation time.
    """

    chunk_id: str
    text: str

    parent_element_id: str
    element_type: ElementType

    source_file: str
    page_number: int | None = None
    chunk_index: int = 0

    raw_html: str | None = None
    image_base64: str | None = None

    dense_embedding: list[float] | None = Field(default=None, repr=False)
    sparse_embedding: dict[int, float] | None = Field(default=None, repr=False)

    def has_embeddings(self) -> bool:
        return self.dense_embedding is not None