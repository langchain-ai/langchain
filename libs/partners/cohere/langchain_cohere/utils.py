from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import cohere
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    # support v4 and v5
    retry_conditions = (
        retry_if_exception_type(cohere.error.CohereError)
        if hasattr(cohere, "error")
        else retry_if_exception_type(Exception)
    )

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=retry_conditions,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def render_messages(messages: Sequence[BaseMessage]) -> str:
    """Renders one or more BaseMessage implementations into prompt content."""
    return "".join(
        [
            f"<|START_OF_TURN_TOKEN|>{render_role(message)}{message.content}<|END_OF_TURN_TOKEN|>"
            for message in messages
        ]
    )


def render_role(message: BaseMessage) -> str:
    """Renders the role of a message into prompt content."""
    if isinstance(message, AIMessage):
        return "<|CHATBOT_TOKEN|>"
    elif isinstance(message, SystemMessage):
        return "<|SYSTEM_TOKEN|>"
    else:
        return "<|USER_TOKEN|>"
