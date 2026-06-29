"""Shared data schemas for the RAG pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class FileCategory(str, Enum):
    """Classification of files in storage."""
    DOCUMENT = "document"   # indexable: pdf, docx, pptx, txt, etc.
    VIDEO = "video"         # mp4, avi, mov, etc.
    AUDIO = "audio"         # mp3, wav, etc.
    IMAGE = "image"         # jpg, png, etc.
    OTHER = "other"         # zip, rar, unknown, etc.


class ParsedElement(BaseModel):
    element_id: str
    element_type: ElementType
    content: str = ""
    raw_html: str | None = None
    image_base64: str | None = None
    summary: str | None = None

    source_file: str
    page_number: int | None = None
    course: str | None = None
    week: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def embeddable_text(self) -> str:
        if self.element_type == ElementType.TEXT:
            return self.content
        if self.summary:
            return self.summary
        return self.content

    def has_visual_payload(self) -> bool:
        return self.element_type in (ElementType.TABLE, ElementType.IMAGE)


class Chunk(BaseModel):
    chunk_id: str
    text: str

    parent_element_id: str
    element_type: ElementType

    source_file: str
    page_number: int | None = None
    chunk_index: int = 0
    course: str | None = None
    week: int | None = None

    raw_html: str | None = None
    image_base64: str | None = None

    dense_embedding: list[float] | None = Field(default=None, repr=False)
    sparse_embedding: dict[int, float] | None = Field(default=None, repr=False)

    def has_embeddings(self) -> bool:
        return self.dense_embedding is not None