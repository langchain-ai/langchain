"""Derivations of standard content blocks from Ollama content."""

from typing import cast
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Ollama v0 content."""
    content_blocks: list[types.ContentBlock] = []

    # First, handle reasoning content from additional_kwargs. There will be only one
    if "reasoning_content" in message.additional_kwargs:
        reasoning_content = message.additional_kwargs["reasoning_content"]
        # `reasoning_content` is only ever a str
        if reasoning_content:
            content_blocks.append(
                cast(
                    "types.ReasoningContentBlock",
                    {
                        "type": "reasoning",
                        "reasoning": reasoning_content,
                    },
                )
            )

    # Handle main content
    if message.content and isinstance(message.content, str):
        content_blocks.append(
            cast("types.TextContentBlock", {"type": "text", "text": message.content})
        )
    elif isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    block = {"type": "text", "text": item["text"]}

                    if item.get("id"):
                        block["id"] = item["id"]

                    content_blocks.append(
                        cast(
                            "types.TextContentBlock",
                            block,
                        )
                    )
                else:
                    # Keep other content types as-is (multimodal, etc.)
                    # These should be handled during `_normalize_messages`
                    content_blocks.append(cast("types.ContentBlock", item))
            elif isinstance(item, str):
                content_blocks.append(
                    cast("types.TextContentBlock", {"type": "text", "text": item})
                )

    # Handle tool calls
    content_blocks.extend(
        [
            cast(
                "types.ToolCall",
                {
                    "type": "tool_call",
                    "id": tool_call.get("id", str(uuid4())),
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                },
            )
            for tool_call in message.tool_calls
        ]
    )

    return content_blocks


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Ollama content."""
    content_blocks: list[types.ContentBlock] = []

    if "reasoning_content" in message.additional_kwargs:
        reasoning_content = message.additional_kwargs["reasoning_content"]
        if reasoning_content:
            content_blocks.append(
                cast(
                    "types.ReasoningContentBlock",
                    {
                        "type": "reasoning",
                        "reasoning": reasoning_content,
                    },
                )
            )

    # Handle main content
    if isinstance(message.content, str) and message.content:
        content_blocks.append(
            cast("types.TextContentBlock", {"type": "text", "text": message.content})
        )
    elif isinstance(message.content, list):
        for item in message.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    block = {"type": "text", "text": item["text"]}

                    if item.get("id"):
                        block["id"] = item["id"]

                    content_blocks.append(
                        cast(
                            "types.TextContentBlock",
                            block,
                        )
                    )
                else:
                    content_blocks.append(cast("types.ContentBlock", item))
            elif isinstance(item, str):
                content_blocks.append(
                    cast("types.TextContentBlock", {"type": "text", "text": item})
                )

    # Handle tool calls
    content_blocks.extend(
        [
            cast(
                "types.ToolCall",
                {
                    "type": "tool_call",
                    "id": tool_call.get("id", str(uuid4())),
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                },
            )
            for tool_call in message.tool_calls
        ]
    )

    # Handle tool call chunks
    for tool_call_chunk in message.tool_call_chunks:
        tc: types.ToolCallChunk = {
            "type": "tool_call_chunk",
            "id": tool_call_chunk.get("id"),
            "name": tool_call_chunk.get("name"),
            "args": tool_call_chunk.get("args"),
        }
        if (idx := tool_call_chunk.get("index")) is not None:
            tc["index"] = idx
        content_blocks.append(tc)

    return content_blocks


def _register_ollama_translator() -> None:
    """Register the Ollama translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator("ollama", translate_content, translate_content_chunk)


_register_ollama_translator()
