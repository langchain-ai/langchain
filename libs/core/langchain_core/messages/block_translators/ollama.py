"""Derivations of standard content blocks from Ollama content."""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Ollama content."""
    raise NotImplementedError


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Ollama content."""
    raise NotImplementedError


def _register_ollama_translator() -> None:
    """Register the Ollama translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator("ollama", translate_content, translate_content_chunk)


_register_ollama_translator()
