"""Derivations of standard content blocks from Amazon (Bedrock) content."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Bedrock content."""
    raise NotImplementedError


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a chunk with Bedrock content."""
    raise NotImplementedError


def _register_bedrock_translator() -> None:
    """Register the Bedrock translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator(
        "amazon_bedrock_chat", translate_content, translate_content_chunk
    )


_register_bedrock_translator()
