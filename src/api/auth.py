"""API key auth for inbound requests (other services calling this API)."""
from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject requests missing a valid X-API-Key header.

    If RAGACADEMIC_API_KEY is unset (empty), auth is disabled — useful for
    local development. Set it in .env to require the header in production.
    """
    expected = settings.ragacademic_api_key.get_secret_value()
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
