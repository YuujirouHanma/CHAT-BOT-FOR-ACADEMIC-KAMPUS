"""Helpers for calling OpenRouter APIs from the backend."""
from __future__ import annotations

from src.config import settings

OPENROUTER_API_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }

    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_app_title:
        headers["X-OpenRouter-Title"] = settings.openrouter_app_title

    return headers