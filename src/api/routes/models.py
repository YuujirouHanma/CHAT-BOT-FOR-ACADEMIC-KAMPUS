"""List selectable generation models for the UI model-switcher.

Returns only models whose provider has an API key configured, plus the default
(from GENERATION_MODEL). The `key` of each model is what clients send as `model`
in /chat/ask, starter-questions, and quiz.
"""
from __future__ import annotations

from fastapi import APIRouter

from src import model_registry
from src.api.schemas import ModelInfo, ModelListResponse
from src.config import settings

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    specs = model_registry.available()
    # Default key: the registry entry matching the configured model, else "".
    default_key = next(
        (s.key for s in specs if s.model == settings.generation_model), ""
    )
    return ModelListResponse(
        default=default_key,
        models=[
            ModelInfo(key=s.key, label=s.label, provider=s.provider, vision=s.vision)
            for s in specs
        ],
    )
