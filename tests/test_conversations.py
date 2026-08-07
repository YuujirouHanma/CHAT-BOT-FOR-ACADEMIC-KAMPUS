"""Tests for riwayat percakapan: simpan, buka kembali, hapus.

Yang paling penting diuji di sini bukan sekadar "tersimpan", melainkan bahwa
percakapan tetap dapat dibuka SETELAH session kedaluwarsa di memori — itulah
inti fiturnya. Juga bahwa menghapus benar-benar menghapus, termasuk salinan di
memori, karena tanpa itu percakapan yang dihapus akan muncul lagi.
"""
from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.auth import verify_api_key
from src.api.dependencies import get_pipeline
from src.api.main import app
from src.api.session import Session, SessionStore
from src.pipeline import GuidedTurn, QueryResult
from src.storage import conversation_store


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch) -> Generator[None, None, None]:
    """Simpan ke direktori sementara agar tidak menyentuh data asli."""
    monkeypatch.setattr(conversation_store, "CONVERSATION_DIR", tmp_path / "conv")
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    pipeline = AsyncMock()
    pipeline.query = AsyncMock(
        return_value=QueryResult(
            answer="Jawaban.", sources=[], recommendations=[],
            interaction_id="hitl_1",
        )
    )
    pipeline.guided_turn = AsyncMock(
        return_value=GuidedTurn(step="answer", answer_question=True)
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[verify_api_key] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestStore:
    def test_save_load_roundtrip(self) -> None:
        conversation_store.save({
            "conversation_id": "abc", "title": "Uji", "transcript": [{"role": "user"}],
        })
        assert conversation_store.load("abc")["title"] == "Uji"

    def test_load_missing_returns_none(self) -> None:
        assert conversation_store.load("tidak-ada") is None

    def test_delete_returns_false_when_absent(self) -> None:
        assert conversation_store.delete("tidak-ada") is False

    @pytest.mark.parametrize("jahat", ["../rahasia", "a/b", "..", "", "x" * 100])
    def test_rejects_unsafe_ids(self, jahat: str) -> None:
        """Id dipakai sebagai nama berkas — jangan sampai bisa keluar direktori."""
        assert conversation_store.is_valid_id(jahat) is False
        conversation_store.save({"conversation_id": jahat, "title": "x"})
        assert conversation_store.load(jahat) is None

    def test_list_sorted_newest_first(self) -> None:
        for cid, waktu in [("a", "2026-08-01T10:00:00"), ("b", "2026-08-03T10:00:00")]:
            conversation_store.save({
                "conversation_id": cid, "title": cid, "updated_at": waktu,
            })
        assert [r["conversation_id"]
                for r in conversation_store.list_summaries()] == ["b", "a"]

    def test_list_filters_by_student(self) -> None:
        conversation_store.save({"conversation_id": "a", "student_id": "budi"})
        conversation_store.save({"conversation_id": "b", "student_id": "sari"})
        hasil = conversation_store.list_summaries(student_id="budi")
        assert [r["conversation_id"] for r in hasil] == ["a"]

    def test_corrupt_file_is_skipped_not_fatal(self, tmp_path) -> None:
        conversation_store.CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)
        (conversation_store.CONVERSATION_DIR / "rusak.json").write_text("{bukan json")
        conversation_store.save({"conversation_id": "baik", "title": "Baik"})
        assert [r["conversation_id"]
                for r in conversation_store.list_summaries()] == ["baik"]

    def test_title_from_first_message(self) -> None:
        assert conversation_store.make_title("  apa itu   normalisasi ") == \
            "apa itu normalisasi"
        assert conversation_store.make_title("x" * 100).endswith("…")
        assert conversation_store.make_title("") == "Percakapan baru"


class TestSessionPersistence:
    def test_reopens_after_memory_expiry(self) -> None:
        """Inti fiturnya: percakapan tetap ada setelah session hangus dari RAM."""
        store = SessionStore()
        s = store.create(student_id="budi")
        s.record("user", "apa itu normalisasi?")
        s.record("assistant", "Normalisasi adalah...")
        s.persist()
        sid = s.session_id

        store_baru = SessionStore()          # seolah server baru dinyalakan
        pulih = store_baru.get(sid)
        assert pulih is not None
        assert pulih.student_id == "budi"
        assert pulih.title == "apa itu normalisasi?"
        assert [m["content"] for m in pulih.transcript][0] == "apa itu normalisasi?"

    def test_context_survives_reload(self) -> None:
        store = SessionStore()
        s = store.create()
        s.apply_guided("sbd", "Sistem Basis Data", [3, 4], "sbd-minggu-3", "bab3.pdf")
        s.style = "visual"
        s.topic = "Normalisasi"
        s.persist()

        pulih = SessionStore().get(s.session_id)
        assert pulih is not None
        assert (pulih.course_id, pulih.weeks) == ("sbd", [3, 4])
        assert (pulih.style, pulih.topic) == ("visual", "Normalisasi")

    def test_running_quiz_is_not_restored(self) -> None:
        """Melanjutkan kuis berhari-hari kemudian di tengah soal membingungkan."""
        store = SessionStore()
        s = store.create()
        s.start_quiz([{"question": "q", "options": ["a", "b"], "answer_index": 0}])
        s.persist()
        pulih = SessionStore().get(s.session_id)
        assert pulih is not None
        assert pulih.quiz is None

    def test_greeting_does_not_become_the_title(self) -> None:
        s = Session(session_id="x")
        s.record("user", "halo")
        assert s.title == ""
        s.record("user", "apa itu ERD?")
        assert s.title == "apa itu ERD?"


class TestEndpoints:
    def _buat(self, client: TestClient, pesan: str, student: str = "budi") -> str:
        r = client.post(
            "/chat/ask", json={"question": pesan, "student_id": student}
        ).json()
        return r["session_id"]

    def test_conversation_appears_in_list(self, client: TestClient) -> None:
        sid = self._buat(client, "apa itu normalisasi?")
        body = client.get("/conversations", params={"student_id": "budi"}).json()
        assert body["total"] == 1
        c = body["conversations"][0]
        assert c["conversation_id"] == sid
        assert c["title"] == "apa itu normalisasi?"
        assert c["message_count"] == 2          # pertanyaan + jawaban

    def test_list_only_shows_own_conversations(self, client: TestClient) -> None:
        self._buat(client, "punya budi", student="budi")
        self._buat(client, "punya sari", student="sari")
        body = client.get("/conversations", params={"student_id": "sari"}).json()
        assert [c["title"] for c in body["conversations"]] == ["punya sari"]

    def test_get_returns_full_transcript(self, client: TestClient) -> None:
        sid = self._buat(client, "apa itu ERD?")
        body = client.get(f"/conversations/{sid}").json()
        peran = [m["role"] for m in body["transcript"]]
        assert peran == ["user", "assistant"]
        assert body["transcript"][1]["mode"] == "answer"

    def test_get_missing_returns_404(self, client: TestClient) -> None:
        assert client.get("/conversations/tidak-ada").status_code == 404

    def test_continuing_appends_to_same_conversation(self, client: TestClient) -> None:
        sid = self._buat(client, "pertanyaan pertama")
        client.post("/chat/ask", json={"question": "pertanyaan kedua", "session_id": sid})
        body = client.get(f"/conversations/{sid}").json()
        assert body["message_count"] == 4
        assert body["title"] == "pertanyaan pertama"   # judul tidak berubah

    def test_delete_removes_it(self, client: TestClient) -> None:
        sid = self._buat(client, "mau dihapus")
        assert client.delete(f"/conversations/{sid}").status_code == 204
        assert client.get(f"/conversations/{sid}").status_code == 404
        assert client.get("/conversations").json()["total"] == 0

    def test_delete_missing_returns_404(self, client: TestClient) -> None:
        assert client.delete("/conversations/tidak-ada").status_code == 404

    def test_deleted_conversation_does_not_resurrect(self, client: TestClient) -> None:
        """Salinan di memori harus ikut dibuang, kalau tidak ia tertulis lagi."""
        sid = self._buat(client, "mau dihapus")
        client.delete(f"/conversations/{sid}")
        client.post("/chat/ask", json={"question": "pesan baru", "session_id": sid})
        daftar = client.get("/conversations").json()["conversations"]
        assert sid not in [c["conversation_id"] for c in daftar]
