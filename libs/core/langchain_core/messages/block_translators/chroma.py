"""Derivations of standard content blocks from Chroma content."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Chroma content."""
    raise NotImplementedError


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Chroma content."""
    raise NotImplementedError


def _register_chroma_translator() -> None:
    """Register the Chroma translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator("chroma", translate_content, translate_content_chunk)


_register_chroma_translator()
