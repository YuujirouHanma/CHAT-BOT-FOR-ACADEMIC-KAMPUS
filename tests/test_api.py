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

from src.api.dependencies import get_pipeline
from src.api.main import app
from src.guided import Choice
from src.pipeline import GuidedTurn, IndexResult, QueryResult
from src.tenancy import TenantContext


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
    # Default: pesan dianggap pertanyaan sungguhan, jadi route lanjut ke query().
    # Konteks yang masuk diteruskan kembali seperti implementasi aslinya —
    # `guided_turn` adalah sumber kebenaran konteks, jadi mock yang mengembalikan
    # None untuk semuanya akan diam-diam menghapus filter dari klien.
    async def _echo_context(_question: str, **kwargs: object) -> GuidedTurn:
        return GuidedTurn(
            step="answer",
            answer_question=True,
            course_id=kwargs.get("course_id"),          # type: ignore[arg-type]
            course_name=kwargs.get("course_name"),      # type: ignore[arg-type]
            weeks=kwargs.get("weeks") or [],            # type: ignore[arg-type]
            content_id=kwargs.get("content_id"),        # type: ignore[arg-type]
            source_file=kwargs.get("source_file"),      # type: ignore[arg-type]
        )

    pipeline.guided_turn = AsyncMock(side_effect=_echo_context)
    return pipeline


@pytest.fixture
def client(override_tenant: TenantContext) -> Generator[_TestClient, None, None]:
    """TestClient without 'with' so lifespan (which connects to Qdrant) is skipped.
    The pipeline is provided via dependency_overrides instead.

    Autentikasi diganti satu tenant tetap lewat `override_tenant`
    (tests/conftest.py). Rute di bawah menuntut hak yang berbeda-beda —
    `chat:ask`, `content:write`, `catalog:read` — jadi SELURUH dependency hak
    akses ditimpa, bukan hanya gerbang `require_tenant`-nya.
    """
    mock_pipeline = _make_pipeline_mock()
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
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

    def test_upload_indexing_error_returns_generic_500(
        self, client: TestClient
    ) -> None:
        """Gagal indexing tetap 500 — tetapi sebabnya tidak ikut keluar.

        Dulu pesan galat asli dipantulkan apa adanya. Pesan pustaka kerap memuat
        path berkas di server dan versi komponen, jadi kini hanya kode galat dan
        `request_id` yang dikirim; rinciannya tinggal di log server.
        """
        client.mock_pipeline.index_document = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("indexing crashed")
        )
        files = {"file": ("doc.pdf", BytesIO(b"x"), "application/pdf")}
        response = client.post("/documents/upload", files=files)
        assert response.status_code == 500
        assert "indexing crashed" not in response.text
        body = response.json()
        assert body["error"] == "galat_internal"
        assert body["detail"] == "Terjadi galat internal"
        assert body["request_id"]


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

    def test_ask_query_error_returns_generic_500(self, client: TestClient) -> None:
        """Kegagalan retrieval dibalas 500 tanpa menyebut sebabnya.

        Rincian galat pencarian bisa memuat potongan kueri beserta datanya —
        peta gratis bagi siapa pun yang sedang menjajaki sistem ini.
        """
        client.mock_pipeline.query = AsyncMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("retrieval failed")
        )
        response = client.post("/chat/ask", json={"question": "q"})
        assert response.status_code == 500
        assert "retrieval failed" not in response.text
        body = response.json()
        assert body["error"] == "galat_internal"
        assert body["detail"] == "Terjadi galat internal"

    def test_question_too_long_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/chat/ask", json={"question": "x" * 3000}
        )
        assert response.status_code == 422

    def test_answer_mode_reported(self, client: TestClient) -> None:
        body = client.post("/chat/ask", json={"question": "q"}).json()
        assert body["mode"] == "answer"
        assert body["step"] == "answer"
        assert body["choices"] == []


class TestGuidedChat:
    """Chatbot menuntun mahasiswa memilih mata kuliah → minggu → materi."""

    def test_guided_returns_choices_without_answering(self, client: TestClient) -> None:
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="course",
                message="Mau belajar mata kuliah apa?",
                choices=[
                    Choice(label="Sistem Basis Data", value="sbd", kind="course"),
                    Choice(label="KKA", value="kka", kind="course"),
                ],
            )
        )
        response = client.post("/chat/ask", json={"question": "halo"})

        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "choices"
        assert body["step"] == "course"
        assert [c["label"] for c in body["choices"]] == ["Sistem Basis Data", "KKA"]
        assert body["choices"][0]["value"] == "sbd"
        assert body["answer"] == "Mau belajar mata kuliah apa?"
        assert body["sources"] == []
        # Yang penting: tidak membakar biaya retrieval/LLM untuk sebuah navigasi.
        client.mock_pipeline.query.assert_not_called()  # type: ignore[attr-defined]

    def test_starter_questions_exposed_as_recommendations(
        self, client: TestClient
    ) -> None:
        """Klien lama yang hanya membaca `recommendations` tetap dapat isinya."""
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="question",
                message="Mau tanya apa?",
                choices=[
                    Choice(label="Apa itu ERD?", value="Apa itu ERD?", kind="question"),
                ],
                course_id="sbd",
                weeks=[3],
                content_id="sbd-minggu-3",
                source_file="bab3.pdf",
            )
        )
        body = client.post("/chat/ask", json={"question": "bab3.pdf"}).json()
        assert body["recommendations"] == ["Apa itu ERD?"]
        assert body["context"]["course_id"] == "sbd"
        assert body["context"]["weeks"] == [3]
        assert body["context"]["source_file"] == "bab3.pdf"

    def test_guided_context_carried_into_query(self, client: TestClient) -> None:
        """Saat turn menyatakan ini pertanyaan, filternya ikut ke retrieval."""
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="answer",
                answer_question=True,
                course_id="sbd",
                weeks=[3],
                content_id="sbd-minggu-3",
                source_file="bab3.pdf",
            )
        )
        response = client.post("/chat/ask", json={"question": "apa itu normalisasi?"})
        assert response.status_code == 200
        kwargs = client.mock_pipeline.query.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["content_id"] == "sbd-minggu-3"
        assert kwargs["source_filter"] == "bab3.pdf"
        assert response.json()["context"]["weeks"] == [3]

    def test_guided_false_skips_guidance(self, client: TestClient) -> None:
        response = client.post(
            "/chat/ask", json={"question": "halo", "guided": False}
        )
        assert response.status_code == 200
        assert response.json()["mode"] == "answer"
        client.mock_pipeline.guided_turn.assert_not_called()  # type: ignore[attr-defined]
        client.mock_pipeline.query.assert_called_once()  # type: ignore[attr-defined]

    def test_explicit_choice_fields_passed_to_guided_turn(
        self, client: TestClient
    ) -> None:
        """Klik tombol mengirim course_id/week eksplisit, bukan mengandalkan teks."""
        client.post(
            "/chat/ask",
            json={"question": "Minggu 3", "course_id": "sbd", "week": 3},
        )
        kwargs = client.mock_pipeline.guided_turn.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["course_id"] == "sbd"
        assert kwargs["weeks"] == [3]

    def test_week_out_of_range_returns_422(self, client: TestClient) -> None:
        response = client.post("/chat/ask", json={"question": "x", "week": 99})
        assert response.status_code == 422

    def test_quiz_offered_once_material_selected(self, client: TestClient) -> None:
        """Kuis ditawarkan sebagai tombol — pengguna baru belum tentu tahu fiturnya."""
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="question", message="Mau tanya apa?",
                choices=[Choice(label="Apa itu ERD?", value="Apa itu ERD?",
                                kind="question")],
                course_id="sbd", weeks=[3],
                content_id="sbd-minggu-3", source_file="bab3.pdf",
            )
        )
        body = client.post("/chat/ask", json={"question": "bab3.pdf"}).json()
        kuis = [c for c in body["choices"] if c["kind"] == "quiz"]
        assert len(kuis) == 1
        assert kuis[0]["value"] == "kuis"


class TestQuizInChat:
    """Kuis dikerjakan di dalam percakapan, satu soal per pesan."""

    QUIZ = [
        {"question": "Apa itu DML?", "options": ["Data Manipulation Language",
                                                 "Data Model Language"],
         "answer_index": 0, "explanation": "DML memanipulasi data."},
        {"question": "Perintah INSERT untuk apa?", "options": ["Menghapus",
                                                              "Menambah data"],
         "answer_index": 1, "explanation": "INSERT menambah baris."},
    ]

    def _siapkan(self, client: TestClient) -> str:
        """Pilih materi lebih dulu, kembalikan session_id."""
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="question", message="Mau tanya apa?",
                course_id="sbd", weeks=[3],
                content_id="sbd-minggu-3", source_file="bab3.pdf",
            )
        )
        client.mock_pipeline.quiz = AsyncMock(return_value=self.QUIZ)  # type: ignore[attr-defined]
        return client.post("/chat/ask", json={"question": "bab3.pdf"}).json()["session_id"]

    def test_quiz_starts_with_first_question_only(self, client: TestClient) -> None:
        sid = self._siapkan(client)
        body = client.post(
            "/chat/ask", json={"question": "kuis", "session_id": sid}
        ).json()

        assert body["mode"] == "choices"
        assert body["step"] == "quiz"
        assert "Soal 1 dari 2" in body["answer"]
        assert "Apa itu DML?" in body["answer"]
        # Soal kedua belum boleh bocor.
        assert "INSERT" not in body["answer"]
        assert [c["label"] for c in body["choices"]] == [
            "A. Data Manipulation Language", "B. Data Model Language",
        ]

    def test_answering_advances_to_next_question(self, client: TestClient) -> None:
        sid = self._siapkan(client)
        client.post("/chat/ask", json={"question": "kuis", "session_id": sid})

        body = client.post("/chat/ask", json={"question": "A", "session_id": sid}).json()
        assert body["step"] == "quiz"
        assert "Soal 2 dari 2" in body["answer"]
        client.mock_pipeline.grade_quiz.assert_not_called()  # type: ignore[attr-defined]

    def test_full_quiz_is_graded_with_explanations(self, client: TestClient) -> None:
        sid = self._siapkan(client)
        client.mock_pipeline.grade_quiz = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "total": 2, "correct": 2, "score": 100.0, "attempt_id": "att_1",
                "results": [
                    {"question": q["question"], "options": q["options"],
                     "your_answer": q["answer_index"],
                     "correct_answer": q["answer_index"],
                     "is_correct": True, "explanation": q["explanation"]}
                    for q in self.QUIZ
                ],
            }
        )
        client.post("/chat/ask", json={"question": "kuis", "session_id": sid})
        client.post("/chat/ask", json={"question": "A", "session_id": sid})
        body = client.post("/chat/ask", json={"question": "B", "session_id": sid}).json()

        assert "Skor kamu: 100" in body["answer"]
        assert "DML memanipulasi data." in body["answer"]
        assert body["step"] == "question"
        kwargs = client.mock_pipeline.grade_quiz.call_args.kwargs  # type: ignore[attr-defined]
        args = client.mock_pipeline.grade_quiz.call_args.args  # type: ignore[attr-defined]
        assert args[2] == [0, 1]        # jawaban terkumpul urut
        assert kwargs["session_id"] == sid

    def test_unrecognised_answer_reasks_without_scoring(self, client: TestClient) -> None:
        """Jawaban tak terbaca tidak boleh diam-diam dihitung salah."""
        sid = self._siapkan(client)
        client.post("/chat/ask", json={"question": "kuis", "session_id": sid})

        body = client.post(
            "/chat/ask", json={"question": "hmm tidak tahu", "session_id": sid}
        ).json()
        assert "belum menangkap pilihanmu" in body["answer"]
        assert "Soal 1 dari 2" in body["answer"]     # masih soal yang sama
        assert body["step"] == "quiz"

    def test_quiz_can_be_stopped(self, client: TestClient) -> None:
        sid = self._siapkan(client)
        client.post("/chat/ask", json={"question": "kuis", "session_id": sid})

        body = client.post(
            "/chat/ask", json={"question": "berhenti", "session_id": sid}
        ).json()
        assert body["step"] == "question"
        assert "hentikan" in body["answer"].lower()

    def test_quiz_not_started_without_material(self, client: TestClient) -> None:
        """Minta kuis sebelum memilih materi → dituntun memilih dulu."""
        client.mock_pipeline.guided_turn = AsyncMock(  # type: ignore[attr-defined]
            return_value=GuidedTurn(
                step="course", message="Mau belajar mata kuliah apa?",
                choices=[Choice(label="SBD", value="sbd", kind="course")],
            )
        )
        body = client.post("/chat/ask", json={"question": "kuis"}).json()
        assert body["step"] == "course"
        client.mock_pipeline.quiz.assert_not_called()  # type: ignore[attr-defined]

    def test_answers_during_quiz_are_not_sent_to_retrieval(
        self, client: TestClient
    ) -> None:
        """Selama kuis, pesan adalah jawaban soal — bukan pertanyaan ke RAG."""
        sid = self._siapkan(client)
        client.post("/chat/ask", json={"question": "kuis", "session_id": sid})
        client.mock_pipeline.query.reset_mock()  # type: ignore[attr-defined]

        client.post("/chat/ask", json={"question": "A", "session_id": sid})
        client.mock_pipeline.query.assert_not_called()  # type: ignore[attr-defined]


class TestContextualFollowups:
    def test_history_passed_to_query_for_contextual_followups(
        self, client: TestClient
    ) -> None:
        """Pertanyaan lanjutan harus mengacu percakapan sebelumnya."""
        first = client.post("/chat/ask", json={"question": "apa itu tabel?"}).json()
        sid = first["session_id"]

        client.post(
            "/chat/ask", json={"question": "apa itu relasi?", "session_id": sid}
        )
        kwargs = client.mock_pipeline.query.call_args.kwargs  # type: ignore[attr-defined]
        history = kwargs["history"]
        assert [t["content"] for t in history] == [
            "apa itu tabel?",
            "Jawaban dari pipeline.",
        ]