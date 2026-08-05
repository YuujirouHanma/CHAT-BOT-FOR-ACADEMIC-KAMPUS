"""Startup check for the NLTK data that `unstructured` needs.

`unstructured` tokenizes text while parsing (pptx, docx, …). If the NLTK
packages are missing it tries to download them AT RUNTIME from
https://utic-public-cf.s3.amazonaws.com/... which frequently answers
403 Forbidden inside containers — surfacing as a confusing parse failure.

The Dockerfile pre-downloads the data at build time. This check exists so a
misconfigured deployment says so loudly at startup instead of failing later
on a student's upload.
"""
from __future__ import annotations

from src.utils.logger import logger

# (category, package) pairs required by unstructured.nlp.tokenize
_REQUIRED = [
    ("tokenizers", "punkt_tab"),
    ("taggers", "averaged_perceptron_tagger_eng"),
]


def missing_packages() -> list[str]:
    """Return the NLTK packages unstructured needs but can't find."""
    try:
        from unstructured.nlp.tokenize import check_for_nltk_package
    except Exception as exc:  # unstructured not importable — nothing to check
        logger.debug("Skipping NLTK check: {}", exc)
        return []

    missing: list[str] = []
    for category, name in _REQUIRED:
        try:
            found = check_for_nltk_package(package_name=name, package_category=category)
        except Exception:
            found = False
        if not found:
            missing.append(f"{category}/{name}")
    return missing


def warn_if_missing() -> None:
    """Log an actionable warning when the NLTK data isn't present."""
    missing = missing_packages()
    if not missing:
        logger.info("NLTK data OK — document parsing ready")
        return

    logger.warning(
        "NLTK data missing ({}). unstructured will try to download it at "
        "runtime and often gets HTTP 403, making pptx/docx parsing fail. "
        "Fix: run  python -m nltk.downloader -d /usr/share/nltk_data "
        "punkt_tab averaged_perceptron_tagger_eng  (already in the Dockerfile — "
        "rebuild the image if you see this in Docker).",
        ", ".join(missing),
    )
