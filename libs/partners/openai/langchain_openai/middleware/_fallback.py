"""Retry decision requests without images when the API rejects them."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_openai.decisions import (
        DecisionRequest,
        DecisionResponse,
        OpenAIDecisions,
    )

_RETRY_STATUS_CODES = frozenset({400, 413})

logger = logging.getLogger(__name__)


def decide_with_text_fallback(
    decisions: OpenAIDecisions, requests: Sequence[DecisionRequest]
) -> DecisionResponse:
    """Try each request in order, falling back when one with images is rejected.

    Args:
        decisions: Decision model used to answer the requests.
        requests: Requests to try, ending with one that contains no images.

    Returns:
        The response to the first request the API accepts.
    """
    for request in requests[:-1]:
        try:
            return decisions.invoke(request)
        except openai.APIStatusError as e:
            _raise_unless_retryable(e)
    return decisions.invoke(requests[-1])


async def adecide_with_text_fallback(
    decisions: OpenAIDecisions, requests: Sequence[DecisionRequest]
) -> DecisionResponse:
    """Asynchronously try each request, falling back when one with images is rejected.

    Args:
        decisions: Decision model used to answer the requests.
        requests: Requests to try, ending with one that contains no images.

    Returns:
        The response to the first request the API accepts.
    """
    for request in requests[:-1]:
        try:
            return await decisions.ainvoke(request)
        except openai.APIStatusError as e:
            _raise_unless_retryable(e)
    return await decisions.ainvoke(requests[-1])


def _raise_unless_retryable(error: openai.APIStatusError) -> None:
    if error.status_code not in _RETRY_STATUS_CODES:
        raise error
    logger.warning(
        "Decision request with images was rejected (HTTP %s); retrying without images.",
        error.status_code,
    )
