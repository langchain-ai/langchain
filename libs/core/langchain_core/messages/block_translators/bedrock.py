"""Derivations of standard content blocks from Bedrock content."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.anthropic import (
    _convert_to_v1_from_anthropic,
)


def _convert_to_v1_from_bedrock(message: AIMessage) -> list[types.ContentBlock]:
    """Convert bedrock message content to v1 format."""
    out = _convert_to_v1_from_anthropic(message)

    content_tool_call_ids = {
        block.get("id")
        for block in out
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    for tool_call in message.tool_calls:
        if (id_ := tool_call.get("id")) and id_ not in content_tool_call_ids:
            tool_call_block: types.ToolCall = {
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            if "index" in tool_call:
                tool_call_block["index"] = tool_call["index"]  # type: ignore[typeddict-item]
            if "extras" in tool_call:
                tool_call_block["extras"] = tool_call["extras"]  # type: ignore[typeddict-item]
            out.append(tool_call_block)
    return out


def _convert_to_v1_from_bedrock_chunk(
    message: AIMessageChunk,
) -> list[types.ContentBlock]:
    """Convert bedrock message chunk content to v1 format."""
    if (
        message.content == ""
        and not message.additional_kwargs
        and not message.tool_calls
    ):
        # Bedrock outputs multiple chunks containing response metadata
        return []

    out = _convert_to_v1_from_anthropic(message)

    if (
        message.tool_call_chunks
        and not message.content
        and message.chunk_position != "last"  # keep tool_calls if aggregated
    ):
        for tool_call_chunk in message.tool_call_chunks:
            tc: types.ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tool_call_chunk.get("id"),
                "name": tool_call_chunk.get("name"),
                "args": tool_call_chunk.get("args"),
            }
            if (idx := tool_call_chunk.get("index")) is not None:
                tc["index"] = idx
            out.append(tc)
    return out


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Bedrock content."""
    if "claude" not in message.response_metadata.get("model_name", "").lower():
        raise NotImplementedError  # fall back to best-effort parsing
    return _convert_to_v1_from_bedrock(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Bedrock content."""
    # TODO: add model_name to all Bedrock chunks and update core merging logic
    # to not append during aggregation. Then raise NotImplementedError here if
    # not an Anthropic model to fall back to best-effort parsing.
    return _convert_to_v1_from_bedrock_chunk(message)


def _register_bedrock_translator() -> None:
    """Register the bedrock translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("bedrock", translate_content, translate_content_chunk)


_register_bedrock_translator()
