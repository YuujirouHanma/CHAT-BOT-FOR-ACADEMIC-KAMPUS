"""Image utility helpers shared across the pipeline."""
from __future__ import annotations

import base64 as _b64


def detect_image_mime(image_base64: str) -> str:
    """Detect image MIME type from base64-encoded magic bytes.

    Returns image/png as a safe default. OpenAI vision auto-detects, but
    sending the correct MIME prevents ambiguity.
    """
    try:
        head = _b64.b64decode(image_base64[:24], validate=False)
    except Exception:
        return "image/png"

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:4] == b"GIF8":
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
