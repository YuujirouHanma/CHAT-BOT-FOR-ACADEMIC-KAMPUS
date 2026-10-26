"""Tests for the model registry and multi-provider config helpers."""
from __future__ import annotations

import pytest

from src import model_registry
from src.config import Settings


class TestRegistry:
    def test_get_known_and_unknown(self) -> None:
        spec = model_registry.get("qwen3.7-plus")
        assert spec is not None
        assert spec.provider == "openrouter"
        assert model_registry.get("does-not-exist") is None
        assert model_registry.get(None) is None

    def test_all_specs_nonempty_and_unique_keys(self) -> None:
        specs = model_registry.all_specs()
        keys = [s.key for s in specs]
        assert len(keys) == len(set(keys))
        assert "gpt-4o-mini" in keys

    def test_available_filters_by_configured_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only openrouter key set → only openrouter models available.
        # Patch on the class (pydantic instances reject arbitrary attribute set).
        monkeypatch.setattr(
            type(model_registry.settings), "api_key_for",
            lambda self, provider: "sk-x" if provider == "openrouter" else "",
        )
        avail = model_registry.available()
        assert avail, "expected at least one openrouter model"
        assert all(s.provider == "openrouter" for s in avail)


class TestConfigProviderHelpers:
    def test_base_url_for_all_providers(self) -> None:
        s = Settings()
        for provider in ["openai", "groq", "openrouter", "gemini", "ollama", "huggingface"]:
            assert s.base_url_for(provider).startswith("http")

    def test_gemini_uses_openai_compatible_endpoint(self) -> None:
        assert "generativelanguage.googleapis.com" in Settings().base_url_for("gemini")

    def test_api_key_for_ollama_is_placeholder(self) -> None:
        assert Settings().api_key_for("ollama") == "ollama"

    def test_provider_base_urls_is_not_a_settings_field(self) -> None:
        # ClassVar must not become a pydantic field (would try to load from env).
        assert "PROVIDER_BASE_URLS" not in Settings().model_fields


class TestKunciUntuk:
    """Penerjemahan kosakata LMS (penyedia + id model) ke kunci registry.

    DITEMUKAN SAAT DIJALANKAN, BUKAN OLEH UJI
    Portal mengirim `model_name` berisi id milik penyedia, mis.
    "qwen/qwen3.7-plus". Registry memakai kunci pendeknya sendiri,
    "qwen3.7-plus". Tanpa penerjemahan, id penyedia diteruskan apa adanya
    sebagai kunci, `get()` mengembalikan None, dan permintaan dilayani model
    BAWAAN.

    Terukur pada tumpukan yang berjalan 26 Oktober 2026: tiga panggilan dengan
    `model_name=qwen/qwen3.7-plus` seluruhnya tercatat di buku pemakaian sebagai
    qwen3.7-flash. Satu-satunya tanda adalah peringatan "Unknown model" di log
    server; pemanggil menerima HTTP 200 dan tidak tahu apa-apa.

    Untuk fitur yang gunanya MEMBANDINGKAN model dan biayanya, jawaban yang
    benar dari model yang salah lebih buruk daripada galat. 1135 uji tidak
    menangkapnya karena tidak satu pun memeriksa model APA yang akhirnya
    dipanggil.
    """

    def test_id_penyedia_diterjemahkan(self) -> None:
        assert model_registry.kunci_untuk("openrouter", "qwen/qwen3.7-plus") == "qwen3.7-plus"

    def test_kunci_registry_diterima_apa_adanya(self) -> None:
        """Supaya pemanggil yang mengirim kunci di medan `model_name` tidak
        ikut tergelincir."""
        assert model_registry.kunci_untuk("openrouter", "qwen3.7-plus") == "qwen3.7-plus"

    def test_penyedia_yang_salah_tidak_dicocokkan(self) -> None:
        """Menebak lintas penyedia akan membangun klien ke penyedia yang salah,
        lalu gagal dengan galat yang tidak menyebut sebabnya."""
        assert model_registry.kunci_untuk("openai", "qwen/qwen3.7-plus") is None

    def test_tanpa_penyedia_masih_bisa_bila_tunggal(self) -> None:
        assert model_registry.kunci_untuk(None, "qwen/qwen3.7-plus") == "qwen3.7-plus"

    def test_tidak_dikenal_mengembalikan_none(self) -> None:
        assert model_registry.kunci_untuk("openrouter", "tidak-ada-model-ini") is None

    def test_kosong_mengembalikan_none(self) -> None:
        assert model_registry.kunci_untuk("openrouter", "") is None
        assert model_registry.kunci_untuk(None, None) is None

    def test_setiap_id_penyedia_dapat_diterjemahkan_balik(self) -> None:
        """Penjaga terhadap entri baru yang nanti ditambahkan: tiap spec harus
        bisa ditemukan lewat (penyedia, id)-nya sendiri. Kalau tidak, portal
        yang memakai kosakata LMS akan kehilangan model itu diam-diam."""
        for s in model_registry.all_specs():
            assert model_registry.kunci_untuk(s.provider, s.model) == s.key, s.key


class TestKunciModelPadaPermintaan:
    def test_model_name_penyedia_jadi_kunci_registry(self) -> None:
        from src.api.schemas import QueryRequest

        r = QueryRequest(
            question="x", model_provider="openrouter", model_name="qwen/qwen3.7-plus",
        )

        assert r.kunci_model() == "qwen3.7-plus"

    def test_medan_model_menang(self) -> None:
        from src.api.schemas import QueryRequest

        r = QueryRequest(
            question="x", model="qwen3.6-flash",
            model_provider="openrouter", model_name="qwen/qwen3.7-plus",
        )

        assert r.kunci_model() == "qwen3.6-flash"

    def test_yang_tidak_dikenal_diteruskan_apa_adanya(self) -> None:
        """Bukan None: peringatan "Unknown model" di LLMGenerator._resolve harus
        tetap menyebut apa yang sebenarnya diminta pemanggil, bukan 'default'."""
        from src.api.schemas import QueryRequest

        r = QueryRequest(
            question="x", model_provider="openrouter", model_name="model/ngawur",
        )

        assert r.kunci_model() == "model/ngawur"

    def test_setiap_pembawa_kunci_model_memakai_aturan_yang_sama(self) -> None:
        """Tiga kelas mendefinisikan `kunci_model` sendiri — permintaan chat,
        evaluasi, dan livecode. Tiga salinan berarti perbaikan di satu tempat
        tidak sampai ke dua lainnya, dan dua jalur sisanya diam-diam tetap
        memakai model bawaan.

        Kelasnya ditemukan lewat introspeksi, bukan didaftar tangan: daftar
        tangan tidak akan ikut tumbuh ketika kelas keempat ditambahkan nanti,
        dan ujinya tetap hijau.
        """
        import inspect

        from pydantic import BaseModel

        from src.api import schemas

        pembawa = [
            obj for obj in vars(schemas).values()
            if inspect.isclass(obj) and issubclass(obj, BaseModel)
            and "kunci_model" in obj.__dict__
        ]
        assert len(pembawa) >= 3, f"hanya {len(pembawa)} kelas membawa kunci_model"

        for kelas in pembawa:
            # Dibangun lewat construct supaya medan wajib yang berbeda-beda
            # antar kelas tidak perlu diisi: yang diuji hanya penerjemahannya.
            obj = kelas.model_construct(
                model=None,
                model_provider="openrouter",
                model_name="qwen/qwen3.7-plus",
            )

            assert obj.kunci_model() == "qwen3.7-plus", kelas.__name__
