"""Logging setup for the application.

Provides a configured logger instance with sensible defaults for production use.
"""
from __future__ import annotations

import logging
import sys

from content_growth_agent.config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Configure and return a logger.

    Uses environment-configured log level and human-friendly formatting. For a
    production system this can be replaced by structured logging (JSON) and
    external sinks.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
