"""Middleware to deduplicate parallel tool calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest


def _canonicalize_args(args: Any) -> str:
    """Canonicalize arguments by sorting keys in any nested dictionaries.

    Args:
        args: The arguments to canonicalize.

    Returns:
        A JSON string representation of the arguments with sorted keys.
    """
    try:
        return json.dumps(args, sort_keys=True)
    except Exception:
        return str(args)


class DeduplicateToolCallsMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that deduplicates identical parallel tool calls within a single AIMessage.

    When using tool-calling models, the model may emit duplicate parallel tool calls
    (same tool name and equivalent arguments, sometimes with reordered JSON keys).
    This middleware filters out duplicate tool calls, preserving only the first
    occurrence of each unique tool call in the `AIMessage`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import DeduplicateToolCallsMiddleware

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[DeduplicateToolCallsMiddleware()],
        )
        ```
    """

    def _deduplicate_message(self, message: BaseMessage) -> BaseMessage:
        """Deduplicate tool calls in an AIMessage.

        Args:
            message: The message to deduplicate.

        Returns:
            The message with duplicate tool calls removed.
        """
        if not isinstance(message, AIMessage):
            return message

        has_tool_calls = bool(message.tool_calls)
        has_invalid_tool_calls = bool(getattr(message, "invalid_tool_calls", None))

        if not has_tool_calls and not has_invalid_tool_calls:
            return message

        seen_keys = set()
        update_dict: dict[str, Any] = {}

        if has_tool_calls:
            deduplicated_tool_calls = []
            for tool_call in message.tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
                key = (name, _canonicalize_args(args))
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduplicated_tool_calls.append(tool_call)
            update_dict["tool_calls"] = deduplicated_tool_calls

        if has_invalid_tool_calls:
            deduplicated_invalid_tool_calls = []
            for invalid_call in message.invalid_tool_calls:
                name = (
                    invalid_call.get("name")
                    if isinstance(invalid_call, dict)
                    else getattr(invalid_call, "name", None)
                )
                args = (
                    invalid_call.get("args")
                    if isinstance(invalid_call, dict)
                    else getattr(invalid_call, "args", None)
                )
                key = (name, _canonicalize_args(args))
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduplicated_invalid_tool_calls.append(invalid_call)
            update_dict["invalid_tool_calls"] = deduplicated_invalid_tool_calls

        if hasattr(message, "model_copy"):
            return message.model_copy(update=update_dict)
        return message.copy(update=update_dict)

    def _deduplicate_response(self, response: ModelResponse[ResponseT]) -> ModelResponse[ResponseT]:
        """Deduplicate tool calls in the model response result messages.

        Args:
            response: The original model response.

        Returns:
            A model response with deduplicated messages.
        """
        deduplicated_messages = [self._deduplicate_message(msg) for msg in response.result]
        return ModelResponse(
            result=deduplicated_messages,
            structured_response=response.structured_response,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Intercept model call and deduplicate parallel tool calls in the response.

        Args:
            request: The model call request.
            handler: The handler to execute the model call.

        Returns:
            The model response with deduplicated tool calls.
        """
        response = handler(request)
        return self._deduplicate_response(response)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Intercept model call and deduplicate parallel tool calls in the response asynchronously.

        Args:
            request: The model call request.
            handler: The handler to execute the model call.

        Returns:
            The model response with deduplicated tool calls.
        """
        response = await handler(request)
        return self._deduplicate_response(response)
