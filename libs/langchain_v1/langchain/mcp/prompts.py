"""Prompts adapter for converting MCP prompts to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

This module provides functionality to convert MCP prompt messages into LangChain
message objects, handling both user and assistant message types.
"""

from typing import Any, assert_never

from langchain_core.messages import AIMessage, HumanMessage
from mcp import ClientSession
from mcp.types import PromptMessage


def convert_mcp_prompt_message_to_langchain_message(
    message: PromptMessage,
) -> HumanMessage | AIMessage:
    """Convert an MCP prompt message to a LangChain message.

    Args:
        message: MCP prompt message to convert

    Returns:
        A LangChain message

    """
    if message.content.type == "text":
        if message.role == "user":
            return HumanMessage(content=message.content.text)
        if message.role == "assistant":
            return AIMessage(content=message.content.text)
        assert_never(message.role)

    msg = f"Unsupported prompt message content type: {message.content.type}"
    raise ValueError(msg)


async def load_mcp_prompt(
    session: ClientSession,
    name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> list[HumanMessage | AIMessage]:
    """Load MCP prompt and convert to LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages).

    Args:
        session: The MCP client session.
        name: Name of the prompt to load.
        arguments: Optional arguments to pass to the prompt.

    Returns:
        A list of LangChain [messages](https://docs.langchain.com/oss/python/langchain/messages)
            converted from the MCP prompt.
    """
    response = await session.get_prompt(name, arguments)
    return [
        convert_mcp_prompt_message_to_langchain_message(message) for message in response.messages
    ]
