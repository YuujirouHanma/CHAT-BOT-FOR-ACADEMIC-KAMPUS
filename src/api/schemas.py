"""API request and response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


# --- Chat ---
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    course: str | None = None
    week: int | None = None
    source_filter: str | None = Field(default=None, max_length=255)


class SourceInfo(BaseModel):
    index: int
    source_file: str | None = None
    page_number: int | None = None
    element_type: str | None = None
    course: str | None = None
    week: int | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    recommendations: list[str]
    session_id: str


# --- Upload / Indexing ---
class IndexResponse(BaseModel):
    source_file: str
    elements_parsed: int
    chunks_created: int
    points_stored: int
    course: str | None
    week: int | None


# --- Browse ---
class CourseListResponse(BaseModel):
    courses: list[str]


class WeekListResponse(BaseModel):
    course: str
    weeks: list[int]


class FileInfo(BaseModel):
    filename: str
    course: str
    week: int
    size_bytes: int
    size_display: str
    category: str          # document, video, audio, image, other
    is_indexable: bool
    indexed: bool


class FileListResponse(BaseModel):
    course: str
    week: int
    files: list[FileInfo]
    total_files: int
    total_indexable: int
    total_indexed: int


# --- Batch ---
class BatchIndexRequest(BaseModel):
    course: str
    week: int | None = None


class BatchFileResult(BaseModel):
    filename: str
    course: str
    week: int
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