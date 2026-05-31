"""Centralized logger setup using loguru.

Usage:
    from src.utils.logger import logger
    logger.info("message")
"""
import sys

from loguru import logger

from src.config import settings

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

__all__ = ["logger"]