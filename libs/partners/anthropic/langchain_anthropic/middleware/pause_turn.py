"""Anthropic `pause_turn` continuation middleware.

Requires:
    - `langchain`: For agent middleware framework
    - `langchain-anthropic`: For `ChatAnthropic` model (already a dependency)
"""

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
        "This middleware is designed for use with LangChain agents. "
        "Install it with: pip install langchain"
    )
    raise ImportError(msg) from e


class PauseTurnLimitExceededError(Exception):
    """Raised when a turn is still paused after the allowed continuations."""

    def __init__(self, max_continuations: int) -> None:
        """Initialize the exception with the bound that was reached.

        Args:
            max_continuations: The number of continuations the middleware sent.
        """
        self.max_continuations = max_continuations
        msg = (
            "The model turn was still paused (stop_reason 'pause_turn') after "
            f"{max_continuations} continuations. Raise `max_continuations` on "
            "`AnthropicPauseTurnMiddleware` to let longer server-tool turns finish."
        )
        super().__init__(msg)


class AnthropicPauseTurnMiddleware(AgentMiddleware):
    """Continue model turns that Anthropic pauses with `stop_reason: "pause_turn"`.

    Anthropic runs server tools such as web search and web fetch in a loop of its
    own and can pause a long turn before it finishes. The response then ends with
    `stop_reason: "pause_turn"`, sometimes with a server tool call that has not
    run yet, and carries no client tool calls, so the agent loop would end the run
    on the unfinished turn.

    This middleware sends a paused response back unchanged, with the same tools
    and no added user message, until the turn ends. Every response, paused or
    final, is kept in the conversation in order. Each continuation is a separate
    request, and a turn still paused after `max_continuations` of them raises
    `PauseTurnLimitExceededError`.

    Responses from models other than Anthropic's never pause, so they pass through
    unchanged.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_anthropic import ChatAnthropic
        from langchain_anthropic.middleware import AnthropicPauseTurnMiddleware

        agent = create_agent(
            model=ChatAnthropic(model="claude-sonnet-5-5"),
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            middleware=[AnthropicPauseTurnMiddleware()],
        )
        ```

    Learn more about paused turns
    [here](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools).
    """

    def __init__(self, max_continuations: int = 5) -> None:
        """Initialize the middleware.

        Args:
            max_continuations: The most continuation requests to send for one
                paused turn.

        Raises:
            ValueError: If `max_continuations` is less than one.
        """
        if max_continuations < 1:
            msg = f"max_continuations must be at least 1, got {max_continuations}"
            raise ValueError(msg)
        self.max_continuations = max_continuations

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Call the model and continue the turn while it is paused.

        Args:
            request: The model request.
            handler: The handler to execute the model request.

        Returns:
            The responses of the turn, the paused ones first.

        Raises:
            PauseTurnLimitExceededError: If the turn is still paused after
                `max_continuations` continuations.
        """
        response = handler(request)
        messages = list(response.result)
        for _ in range(self.max_continuations):
            if not _is_paused(messages):
                break
            response = handler(_continuation(request, messages))
            messages.extend(response.result)
        return self._finish(messages, response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Call the model and continue the turn while it is paused (async version).

        Args:
            request: The model request.
            handler: The async handler to execute the model request.

        Returns:
            The responses of the turn, the paused ones first.

        Raises:
            PauseTurnLimitExceededError: If the turn is still paused after
                `max_continuations` continuations.
        """
        response = await handler(request)
        messages = list(response.result)
        for _ in range(self.max_continuations):
            if not _is_paused(messages):
                break
            response = await handler(_continuation(request, messages))
            messages.extend(response.result)
        return self._finish(messages, response)

    def _finish(
        self, messages: list[BaseMessage], response: ModelResponse
    ) -> ModelResponse:
        if _is_paused(messages):
            raise PauseTurnLimitExceededError(self.max_continuations)
        return ModelResponse(
            result=messages, structured_response=response.structured_response
        )


def _is_paused(messages: list[BaseMessage]) -> bool:
    """Return whether the last message is an AI message Anthropic paused."""
    if not messages or not isinstance(messages[-1], AIMessage):
        return False
    return messages[-1].response_metadata.get("stop_reason") == "pause_turn"


def _continuation(request: ModelRequest, turn: list[BaseMessage]) -> ModelRequest:
    """Build the request that sends the turn so far back unchanged."""
    messages = cast("list[AnyMessage]", [*request.messages, *turn])
    return request.override(messages=messages)
