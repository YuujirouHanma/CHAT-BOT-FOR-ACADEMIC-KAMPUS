"""Tests for FastAPI endpoints.

The RAGPipeline is replaced via dependency override so tests don't
load real models or contact external services.
"""
from __future__ import annotations

from collections.abc import Generator
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_pipeline
from src.api.main import app
from src.pipeline import IndexResult, QueryResult


class _TestClient(TestClient):
    """TestClient subclass that carries the mock pipeline for assertions."""
    mock_pipeline: AsyncMock


def _make_pipeline_mock() -> AsyncMock:
    pipeline = AsyncMock()
    pipeline.index_document = AsyncMock(
        return_value=IndexResult(
            source_file="test.pdf",
            elements_parsed=10,
            chunks_created=15,
            points_stored=15,
        )
    )
    pipeline.query = AsyncMock(
        return_value=QueryResult(
            answer="Jawaban dari pipeline.",
            sources=[
                {
                    "index": 1,
                    "source_file": "test.pdf",
                    "page_number": 2,
                    "element_type": "text",
                    "rerank_score": 0.92,
                }
            ],
            recommendations=["Pertanyaan lanjutan?"],
        )
    )
    return pipeline


@pytest.fixture
def client() -> Generator[_TestClient, None, None]:
    """TestClient without 'with' so lifespan (which connects to Qdrant) is skipped.
    The pipeline is provided via dependency_overrides instead."""
    mock_pipeline = _make_pipeline_mock()
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[verify_api_key] = lambda: None
    c = _TestClient(app)
    c.mock_pipeline = mock_pipeline
    yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestUploadEndpoint:
    def test_upload_returns_201_with_counts(self, client: TestClient) -> None:
        files = {
            "file": ("lesson.pdf", BytesIO(b"%PDF fake content"), "application/pdf"),
        }
        response = client.post("/documents/upload", files=files)

        assert response.status_code == 201
        body = response.json()
        assert body["source_file"] == "test.pdf"
        assert body["elements_parsed"] == 10
        assert body["chunks_created"] == 15
        assert body["points_stored"] == 15

    def test_upload_without_file_returns_422(self, client: TestClient) -> None:
        response = client.post("/documents/upload")
        assert response.status_code == 422

    def test_upload_propagates_indexing_error(self, client: TestClient) -> None:
        client.mock_pipeline.index_document = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("indexing crashed")
        )
        files = {"file": ("doc.pdf", BytesIO(b"x"), "application/pdf")}
        response = client.post("/documents/upload", files=files)
        assert response.status_code == 500
        assert "indexing crashed" in response.json()["detail"]


class TestChatEndpoint:
    def test_ask_returns_answer_and_sources(self, client: TestClient) -> None:
        response = client.post(
            "/chat/ask",
            json={"question": "Apa itu statistik?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Jawaban dari pipeline."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["source_file"] == "test.pdf"
        assert body["sources"][0]["index"] == 1

    def test_ask_empty_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/chat/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_missing_question_returns_422(self, client: TestClient) -> None:
        response = client.post("/chat/ask", json={})
        assert response.status_code == 422

    def test_ask_with_source_filter(self, client: TestClient) -> None:
        response = client.post(
            "/chat/ask",
            json={"question": "X", "source_filter": "specific.pdf"},
        )
        assert response.status_code == 200
        call_kwargs = client.mock_pipeline.query.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["source_filter"] == "specific.pdf"

    def test_ask_propagates_query_error(self, client: TestClient) -> None:
        client.mock_pipeline.query = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("retrieval failed")
        )
        response = client.post("/chat/ask", json={"question": "q"})
        assert response.status_code == 500
        assert "retrieval failed" in response.json()["detail"]

    def test_question_too_long_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/chat/ask", json={"question": "x" * 3000}
        )
        assert response.status_code == 422