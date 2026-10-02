"""Guided Learning: template tugas belajar beserta personalisasinya.

Berbeda dari navigasi terpandu (`src/guided.py`) yang menuntun mahasiswa MEMILIH
materi, berkas ini menguji jalur yang menentukan APA yang dilakukan terhadap
materi yang sudah terpilih: pahami konsep, contoh nyata, kuis, dan seterusnya.

Jalur ini sebelumnya tidak punya uji sama sekali, dan justru di situlah dua
kosakata gaya belajar sempat hidup berdampingan tanpa ketahuan.
"""
from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src import learning_styles
from src.api.dependencies import get_pipeline
from src.api.main import app
from src.api.schemas import UserLearningProfile
from src.generation.exercise import FORMAT_LEMBAR_KERJA, Latihan
from src.pipeline import QueryResult


@pytest.fixture
def client(override_tenant) -> Iterator[TestClient]:
    """Klien dengan pipeline tiruan; tenant dan batas laju sudah ditimpa."""
    pipeline = AsyncMock()
    pipeline.guided_query = AsyncMock(
        return_value=QueryResult(
            answer="Penjelasan konsep.",
            sources=[{
                "index": 1, "source_file": "materi.pdf", "page_number": 3,
                "element_type": "text", "rerank_score": 0.81,
            }],
            recommendations=["Lanjutan?"],
        )
    )
    pipeline.generate_exercise = AsyncMock(
        return_value=(
            Latihan(format=FORMAT_LEMBAR_KERJA, judul="Latihan Percabangan",
                    pengantar="Kerjakan berurutan.", tugas=[]),
            [], None,
        )
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    c = TestClient(app)
    c.pipeline = pipeline  # type: ignore[attr-defined]
    yield c
    app.dependency_overrides.pop(get_pipeline, None)


class TestTemplate:
    def test_daftar_template_tidak_memanggil_model(self, client: TestClient) -> None:
        """Daftar template harus seketika: ia hanya membaca tabel, bukan menalar.

        Kalau suatu saat endpoint ini ikut memanggil model, tombol di portal
        kampus akan menunggu puluhan detik hanya untuk memunculkan pilihan.
        """
        r = client.get("/guided/templates")
        assert r.status_code == 200
        daftar = r.json()["templates"]
        assert len(daftar) >= 6
        assert client.pipeline.guided_query.await_count == 0  # type: ignore[attr-defined]

    def test_setiap_template_lengkap(self, client: TestClient) -> None:
        for t in client.get("/guided/templates").json()["templates"]:
            assert t["template_id"] and t["label"] and t["description"]


class TestPersonalisasi:
    """Label yang dibaca mahasiswa harus kalimat, bukan kunci mentah."""

    def test_label_memakai_kata_manusia(self, client: TestClient) -> None:
        r = client.post("/guided/ask", json={
            "template_id": "understand_concept",
            "topic": "jaringan saraf",
            "user_profile": {"student_name": "Budi",
                             "learning_style": "practical",
                             "explanation_depth": "academic"},
        })
        assert r.status_code == 200
        label = r.json()["personalized_label"]
        assert "Budi" in label
        # Bukan "practical" dan "academic" mentah: keduanya kunci internal.
        assert "practical" not in label
        assert "academic" not in label
        assert "penjelasan akademis" in label

    @pytest.mark.parametrize("gaya_lama,gaya_baru", [
        ("reading", "structured"), ("kinesthetic", "practical"),
        ("auditory", "narrative"), ("visual", "visual"),
    ])
    def test_nama_vark_lama_tetap_dilayani(
        self, client: TestClient, gaya_lama: str, gaya_baru: str,
    ) -> None:
        """Profil mahasiswa di portal masih menyimpan nama VARK.

        Menolaknya berarti setiap pengguna lama dibalas 422 sesudah pembaruan.
        """
        r = client.post("/guided/ask", json={
            "template_id": "understand_concept", "topic": "topik",
            "user_profile": {"learning_style": gaya_lama},
        })
        assert r.status_code == 200
        assert learning_styles.resolve(gaya_lama).key == gaya_baru
        assert learning_styles.resolve(gaya_lama).label in r.json()["personalized_label"]


class TestKosakataTunggal:
    """Satu daftar gaya dan satu tabel kedalaman, tanpa kembaran."""

    def test_setiap_nilai_skema_punya_gaya(self) -> None:
        """Nilai yang diterima skema harus benar-benar ada gayanya.

        Skema dan tabel gaya pernah terpisah, sehingga skema menerima nilai yang
        tidak punya instruksi apa pun — jawabannya diam-diam jatuh ke bawaan.
        """
        medan = UserLearningProfile.model_fields["learning_style"]
        for nilai in medan.annotation.__args__:  # type: ignore[union-attr]
            assert learning_styles.get(nilai) is not None, nilai

    def test_setiap_gaya_punya_instruksi_dan_label(self) -> None:
        for s in learning_styles.all_styles():
            assert s.system_suffix.strip(), s.key
            assert s.label.strip(), s.key


class TestLatihanUnduhan:
    def test_menghasilkan_berkas_bukan_jawaban_chat(self, client: TestClient) -> None:
        r = client.post("/guided/exercise", json={
            "topic": "percabangan", "class_name": "Dasar Pemrograman",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["filename"].endswith(".md")
        assert d["mime_type"] == "text/markdown"
        assert d["content"]

    def test_materi_kosong_bukan_galat_server(self, client: TestClient) -> None:
        """Materi belum diunggah adalah keadaan wajar, bukan kerusakan.

        Dibalas 5xx, portal akan menampilkannya sebagai gangguan sistem dan
        mahasiswa mencoba berulang kali padahal yang kurang adalah materinya.
        """
        client.pipeline.generate_exercise = AsyncMock(  # type: ignore[attr-defined]
            return_value=(None, [], "materi_kosong"))
        r = client.post("/guided/exercise", json={"topic": "apa pun"})
        assert r.status_code < 500, r.text
