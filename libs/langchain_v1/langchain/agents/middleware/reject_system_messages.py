"""Middleware to reject user-injected system messages from conversation history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import AnyMessage, SystemMessage
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

OnViolation = Literal["filter", "error"]
"""How to handle system messages found in conversation history.

- `'filter'`: Silently remove system messages from the conversation (default).
- `'error'`: Raise a `SystemMessageViolationError`.
"""


class SystemMessageViolationError(ValueError):
    """Raised when system messages are found in conversation history and on_violation='error'."""


class RejectSystemMessagesMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Remove or reject system messages injected into conversation history.

    Agents typically hard-code a system message via `request.system_message`.
    End users can inject additional `SystemMessage` instances into the conversation
    history (`request.messages`), which may cause the model to ignore the agent's
    intended system prompt.

    This middleware guards against that by filtering out (or raising on) any
    `SystemMessage` found in `request.messages`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import RejectSystemMessagesMiddleware

        agent = create_agent(
            "openai:gpt-4.1",
            middleware=[RejectSystemMessagesMiddleware()],
        )
        ```

    Example:
        ```python
        from langchain.agents.middleware import RejectSystemMessagesMiddleware

        agent = create_agent(
            "openai:gpt-4.1",
            middleware=[RejectSystemMessagesMiddleware(on_violation="error")],
        )
        ```
    """

    def __init__(self, *, on_violation: OnViolation = "filter") -> None:
        """Initialize the middleware.

        Args:
            on_violation: How to handle system messages found in conversation history.
        """
        self.on_violation = on_violation
        self.tools = []

    def _filter_messages(self, messages: list[AnyMessage]) -> tuple[list[AnyMessage], int]:
        """Return messages with SystemMessages removed and the count removed."""
        filtered: list[AnyMessage] = [m for m in messages if not isinstance(m, SystemMessage)]
        return filtered, len(messages) - len(filtered)

    def _check_violation(self, removed_count: int) -> None:
        """Raise or log depending on on_violation setting."""
        if self.on_violation == "error":
            msg = (
                f"Found {removed_count} system message(s) in conversation history. "
                "System messages should not be injected by end users."
            )
            raise SystemMessageViolationError(msg)
        logger.warning("Filtered %d system message(s) from conversation history.", removed_count)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelCallResult[ResponseT]:
        """Filter or reject system messages from conversation history before model call."""
        filtered, removed_count = self._filter_messages(request.messages)
        if removed_count == 0:
            return handler(request)
        self._check_violation(removed_count)
        return handler(request.override(messages=filtered))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelCallResult[ResponseT]:
        """Async version of wrap_model_call."""
        filtered, removed_count = self._filter_messages(request.messages)
        if removed_count == 0:
            return await handler(request)
        self._check_violation(removed_count)
        return await handler(request.override(messages=filtered))
