"""Pydantic request/response schemas for the FastAPI layer.

Kept separate from src/schemas.py (domain models) so API contracts
can evolve independently from internal pipeline types.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class IndexResponse(BaseModel):
    source_file: str
    elements_parsed: int
    chunks_created: int
    points_stored: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    source_filter: str | None = None


class SourceInfo(BaseModel):
    index: int
    source_file: str | None = None
    page_number: int | None = None
    element_type: str | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
