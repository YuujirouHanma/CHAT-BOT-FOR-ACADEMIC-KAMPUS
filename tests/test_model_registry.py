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
