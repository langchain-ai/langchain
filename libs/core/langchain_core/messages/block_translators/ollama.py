"""Derivations of standard content blocks from Ollama content."""

import warnings

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types

WARNED = False


def translate_content(message: AIMessage) -> list[types.ContentBlock]:  # noqa: ARG001
    """Derive standard content blocks from a message with Ollama content."""
    global WARNED  # noqa: PLW0603
    if not WARNED:
        warning_message = (
            "Content block standardization is not yet fully supported for Ollama."
        )
        warnings.warn(warning_message, stacklevel=2)
        WARNED = True
    raise NotImplementedError


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:  # noqa: ARG001
    """Derive standard content blocks from a message chunk with Ollama content."""
    global WARNED  # noqa: PLW0603
    if not WARNED:
        warning_message = (
            "Content block standardization is not yet fully supported for Ollama."
        )
        warnings.warn(warning_message, stacklevel=2)
        WARNED = True
    raise NotImplementedError


def _register_ollama_translator() -> None:
    """Register the Ollama translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("ollama", translate_content, translate_content_chunk)


_register_ollama_translator()
