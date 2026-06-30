"""API request and response schemas."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Chat ---
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    content_id: str | None = None
    source_filter: str | None = Field(default=None, max_length=255)


class SourceInfo(BaseModel):
    index: int
    source_file: str | None = None
    page_number: int | None = None
    element_type: str | None = None
    content_id: str | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    recommendations: list[str]
    session_id: str
    interaction_id: str | None = None


class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: Literal["membantu", "cukup", "tidak_membantu"]
    issues: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=1000)


# --- Upload / Indexing ---
class IndexResponse(BaseModel):
    source_file: str
    elements_parsed: int
    chunks_created: int
    points_stored: int
    content_id: str | None


# --- Browse ---
class ContentListResponse(BaseModel):
    contents: list[str]


class FileInfo(BaseModel):
    filename: str
    content_id: str
    size_bytes: int
    size_display: str
    category: str          # document, video, audio, image, other
    is_indexable: bool
    indexed: bool


class FileListResponse(BaseModel):
    content_id: str
    files: list[FileInfo]
    total_files: int
    total_indexable: int
    total_indexed: int


# --- Batch ---
class BatchIndexRequest(BaseModel):
    content_id: str


class BatchFileResult(BaseModel):
    filename: str
    content_id: str
    chunks_created: int
    points_stored: int
    status: str


class BatchIndexSummary(BaseModel):
    total_files: int
    indexed: int
    skipped: int
    errors: list[str]
    results: list[BatchFileResult]


# --- Error ---
class ErrorResponse(BaseModel):
    detail: str
