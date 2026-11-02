"""Middleware for continuing paused Anthropic server-side turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        ModelCallResult,
        ModelRequest,
        ModelResponse,
    )
except ImportError as e:
    msg = (
        "AnthropicPauseTurnMiddleware requires 'langchain' to be installed. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


class PauseTurnLimitExceededError(RuntimeError):
    """Raised when an Anthropic turn remains paused past its continuation limit."""


class AnthropicPauseTurnMiddleware(AgentMiddleware):
    """Continue Anthropic server-side turns that return `pause_turn`.

    Add this middleware to an agent to let Anthropic finish server-side tool
    loops, such as web search, that can span multiple API responses. Without
    this middleware, agent behavior is unchanged.

    Requires both `langchain` and `langchain-anthropic` packages.
    """

    def __init__(self, max_continuations: int = 5) -> None:
        """Initialize the middleware.

        Args:
            max_continuations: Maximum number of follow-up requests after the
                initial paused response.

        Raises:
            ValueError: If `max_continuations` is negative.
        """
        if max_continuations < 0:
            msg = "max_continuations must be greater than or equal to 0"
            raise ValueError(msg)
        self.max_continuations = max_continuations

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Continue a paused turn using synchronous model calls.

        Args:
            request: The original model request.
            handler: The synchronous model call handler.

        Returns:
            The response, including intermediate paused assistant messages.

        Raises:
            PauseTurnLimitExceededError: If the turn remains paused after the
                configured number of continuations.
        """
        messages = list(request.messages)
        collected: list[BaseMessage] = []

        for continuation in range(self.max_continuations + 1):
            response = handler(request.override(messages=messages))
            if not isinstance(response, ModelResponse):
                return response
            collected.extend(response.result)
            paused = _get_paused_message(response)
            if paused is None:
                return ModelResponse(
                    result=collected,
                    structured_response=response.structured_response,
                )
            if continuation == self.max_continuations:
                msg = (
                    "Anthropic turn remained paused after "
                    f"{self.max_continuations} continuations"
                )
                raise PauseTurnLimitExceededError(msg)
            messages.extend(cast(list[AnyMessage], response.result))

        msg = "unreachable"
        raise AssertionError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Continue a paused turn using asynchronous model calls.

        Args:
            request: The original model request.
            handler: The asynchronous model call handler.

        Returns:
            The response, including intermediate paused assistant messages.

        Raises:
            PauseTurnLimitExceededError: If the turn remains paused after the
                configured number of continuations.
        """
        messages = list(request.messages)
        collected: list[BaseMessage] = []

        for continuation in range(self.max_continuations + 1):
            response = await handler(request.override(messages=messages))
            if not isinstance(response, ModelResponse):
                return response
            collected.extend(response.result)
            paused = _get_paused_message(response)
            if paused is None:
                return ModelResponse(
                    result=collected,
                    structured_response=response.structured_response,
                )
            if continuation == self.max_continuations:
                msg = (
                    "Anthropic turn remained paused after "
                    f"{self.max_continuations} continuations"
                )
                raise PauseTurnLimitExceededError(msg)
            messages.extend(cast(list[AnyMessage], response.result))

        msg = "unreachable"
        raise AssertionError(msg)


def _get_paused_message(response: ModelResponse) -> AIMessage | None:
    """Return the paused assistant message when the response indicates one."""
    for message in reversed(response.result):
        if isinstance(message, AIMessage):
            if message.response_metadata.get("stop_reason") == "pause_turn":
                return message
            return None
    return None


__all__ = ["AnthropicPauseTurnMiddleware", "PauseTurnLimitExceededError"]
