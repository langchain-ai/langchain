"""Derivations of standard content blocks from Bedrock content."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.anthropic import (
    _convert_to_v1_from_anthropic,
    _convert_to_v1_from_anthropic_input,
)


def _convert_to_v1_from_bedrock_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Attempt to unpack non-standard blocks."""
    return _convert_to_v1_from_anthropic_input(content)


def _convert_to_v1_from_bedrock(message: AIMessage) -> list[types.ContentBlock]:
    """Convert bedrock message content to v1 format."""
    return _convert_to_v1_from_anthropic(message)


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
    return _convert_to_v1_from_bedrock(message)


def _register_bedrock_translator() -> None:
    """Register the bedrock translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("bedrock", translate_content, translate_content_chunk)


_register_bedrock_translator()
