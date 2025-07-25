"""Compatibility module for handling v1 message format conversions."""

from __future__ import annotations

from typing import Any, Literal, Optional, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    is_data_content_block,
)
from langchain_core.messages.content_blocks import (
    NonStandardContentBlock,
    ReasoningContentBlock,
    TextContentBlock,
)
from typing_extensions import TypedDict


class ToolCallReferenceBlock(TypedDict):
    """Reference to a tool call (metadata only).

    This is used in v1 content blocks to reference tool calls
    without duplicating the full tool call data.
    """

    type: Literal["tool_call"]
    id: Optional[str]


def _convert_unknown_content_block_to_non_standard(
    block: dict,
) -> NonStandardContentBlock:
    """Convert unknown content block to NonStandardContentBlock format.

    This enables forward compatibility by preserving unknown content block types
    instead of raising errors.

    Args:
        block: Unknown content block dictionary.

    Returns:
        NonStandardContentBlock containing the original block data.
    """
    return NonStandardContentBlock(type="non_standard", value=block)


def _convert_from_v1_message(message: AIMessage) -> AIMessage:
    """Convert a v1 message to Ollama-compatible request format.

    Returns AIMessage with v0-style content and reasoning in ``additional_kwargs``.

    If input is already v0 format, returns unchanged.

    Args:
        message: The message to convert, potentially in v1 format.

    Returns:
        AIMessage in v0 format suitable for Ollama API.
    """
    if not isinstance(message.content, list):
        # Already v0 format or string content (determined by content type)
        return message

    # Extract components from v1 content blocks
    text_content = ""
    reasoning_content = None

    for block in message.content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text_content += block.get("text", "")
            elif block_type == "reasoning":
                # Extract reasoning for additional_kwargs
                reasoning_content = block.get("reasoning", "")
            elif block_type == "tool_call":
                # Skip - handled via tool_calls property
                continue
            elif is_data_content_block(block):
                # Keep data blocks as-is (images already supported)
                continue
            else:
                # Convert unknown content blocks to NonStandardContentBlock
                # TODO what to do from here?
                _convert_unknown_content_block_to_non_standard(block)
                continue

    # Update message with extracted content
    updates: dict[str, Any] = {"content": text_content if text_content else ""}
    if reasoning_content:
        additional_kwargs = dict(message.additional_kwargs)
        additional_kwargs["reasoning_content"] = reasoning_content
        updates["additional_kwargs"] = additional_kwargs

    return message.model_copy(update=updates)


def _convert_to_v1_message(message: AIMessage) -> AIMessage:
    """Convert an Ollama message to v1 format.

    Args:
        message: AIMessage in v0 format from Ollama.

    Returns:
        AIMessage in v1 format with content blocks.
    """
    new_content: list[Any] = []

    # Handle reasoning content first (from additional_kwargs)
    additional_kwargs = dict(message.additional_kwargs)
    if reasoning_content := additional_kwargs.pop("reasoning_content", None):
        reasoning_block = ReasoningContentBlock(
            type="reasoning", reasoning=reasoning_content
        )
        new_content.append(reasoning_block)

    # Convert text content to content blocks
    if isinstance(message.content, str) and message.content:
        text_block = TextContentBlock(type="text", text=message.content)
        new_content.append(text_block)

    # Convert tool calls to content blocks
    for tool_call in message.tool_calls:
        if tool_call_id := tool_call.get("id"):
            # Create a tool call reference block (metadata only)
            tool_call_block = ToolCallReferenceBlock(type="tool_call", id=tool_call_id)
            new_content.append(dict(tool_call_block))

    # Handle any non-standard content blocks that might be stored in additional_kwargs
    if non_standard_blocks := additional_kwargs.pop(
        "non_standard_content_blocks", None
    ):
        new_content.extend(non_standard_blocks)

    return message.model_copy(
        update={"content": new_content, "additional_kwargs": additional_kwargs}
    )


def _convert_to_v1_chunk(chunk: AIMessageChunk) -> AIMessageChunk:
    """Convert an Ollama chunk to v1 format.

    Args:
        chunk: AIMessageChunk in v0 format from Ollama.

    Returns:
        AIMessageChunk in v1 format with content blocks.
    """
    result = _convert_to_v1_message(cast(AIMessage, chunk))
    return cast(AIMessageChunk, result)


def _convert_from_v03_ai_message(message: AIMessage) -> AIMessage:
    """Convert a LangChain v0.3 AIMessage to v1 format.

    This handles compatibility for users migrating stored/cached AIMessage objects
    from LangChain v0.3.

    Args:
        message: AIMessage potentially in v0.3 format.

    Returns:
        AIMessage in v1 format.
    """
    # For now, treat v0.3 messages the same as v0 messages
    return _convert_to_v1_message(message)
