"""Failure logging shared by the middleware that recover from classification errors."""

from __future__ import annotations

import logging


def log_classification_failure(
    logger: logging.Logger,
    error: Exception,
    fallback: str,
) -> None:
    """Record a classification failure without copying provider content into logs.

    The TypeSafe SDK embeds a truncated response body in `str(error)`, and a
    validation error can echo back part of the classified state. Agent state
    routinely contains user content, so only the error type and the provider's
    request metadata are logged.

    Args:
        logger: Logger belonging to the middleware that failed.
        error: Exception raised while classifying.
        fallback: Description of the behavior being used instead.
    """
    logger.warning(
        "TypeSafe classification failed (%s, status=%s, request_id=%s); %s",
        type(error).__name__,
        getattr(error, "status", None),
        getattr(error, "request_id", None),
        fallback,
    )
