"""Registry of selectable generation models across providers.

The UI lets users switch model per request. Each request may send a `model`
key from this registry; the backend resolves it to (provider, model id) and
builds the right OpenAI-compatible client on the fly.

`available()` returns only models whose provider has an API key configured,
so the UI dropdown never offers a model that would fail.

To add/adjust models, edit `_SPECS` below — that is the single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.config import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str          # stable id the UI/request uses, e.g. "qwen3.7-plus"
    label: str        # human display name
    provider: str     # openrouter | openai | groq | gemini | ...
    model: str        # provider-specific model id
    vision: bool      # supports image input


_SPECS: list[ModelSpec] = [
    ModelSpec("qwen3.7-plus", "Qwen 3.7 Plus", "openrouter", "qwen/qwen3.7-plus", True),
    ModelSpec("qwen3.6-flash", "Qwen 3.6 Flash", "openrouter", "qwen/qwen3.6-flash", True),
    ModelSpec("gpt-4o-mini", "GPT-4o mini", "openai", "gpt-4o-mini", True),
    ModelSpec("gemini-flash", "Gemini 2.5 Flash", "gemini", "gemini-2.5-flash", True),
    ModelSpec("gemini-pro", "Gemini 2.5 Pro", "gemini", "gemini-2.5-pro", True),
    ModelSpec("groq-llama-70b", "Llama 3.3 70B (Groq)", "groq", "llama-3.3-70b-versatile", False),
]

_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def get(key: str | None) -> ModelSpec | None:
    """Resolve a registry key to its spec, or None if unknown/empty."""
    if not key:
        return None
    return _REGISTRY.get(key)


def all_specs() -> list[ModelSpec]:
    return list(_SPECS)


def available() -> list[ModelSpec]:
    """Only models whose provider has an API key configured."""
    return [s for s in _SPECS if settings.api_key_for(s.provider)]
