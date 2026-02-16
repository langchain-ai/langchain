"""Utils file included for backwards compat imports."""

import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_messages_from_invalid_tool_calls
from langgraph.prebuilt import InjectedState, InjectedStore, ToolRuntime
from langgraph.prebuilt.tool_node import (
    ToolCallRequest,
    ToolCallWithContext,
    ToolCallWrapper,
)
from langgraph.prebuilt.tool_node import (
    ToolNode as _ToolNode,  # noqa: F401
)

logger = logging.getLogger(__name__)

__all__ = [
    "InjectedState",
    "InjectedStore",
    "ToolCallRequest",
    "ToolCallWithContext",
    "ToolCallWrapper",
    "ToolRuntime",
    "convert_invalid_tool_calls",
]


def convert_invalid_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    """Convert invalid tool calls from an ``AIMessage`` to error ``ToolMessage`` objects.

    If the ``AIMessage`` has no ``invalid_tool_calls``, returns an empty list.

    This is used by the agent loop to surface JSON parsing errors back to the
    LLM so it can self-correct, rather than silently terminating.

    Args:
        ai_message: The AI message potentially containing invalid tool calls.

    Returns:
        List of error ``ToolMessage`` objects, one per invalid tool call.
    """
    if not ai_message.invalid_tool_calls:
        return []

    logger.debug(
        "Converting %d invalid tool call(s) to error messages",
        len(ai_message.invalid_tool_calls),
    )
    return tool_messages_from_invalid_tool_calls(ai_message.invalid_tool_calls)
