"""Centralized logger setup using loguru.

Usage:
    from src.utils.logger import logger
    logger.info("message")
"""
import sys
from typing import Any, cast

logger: Any

try:
    from loguru import logger
    from src.config import settings

    logger = cast(Any, logger)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=settings.app_env == "development",
    )
except ImportError:  # pragma: no cover - fallback when loguru not installed
    import logging

    logger = cast(Any, logging.getLogger("ragacademic"))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s")
    )
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

__all__ = ["logger"]