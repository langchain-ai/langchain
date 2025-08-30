"""Derivations of standard content blocks from Amazon (Bedrock) content."""

import warnings

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types

WARNED = False


def translate_content(message: AIMessage) -> list[types.ContentBlock]:  # noqa: ARG001
    """Derive standard content blocks from a message with Bedrock content."""
    global WARNED  # noqa: PLW0603
    if not WARNED:
        warning_message = (
            "Content block standardization is not yet fully supported for Bedrock."
        )
        warnings.warn(warning_message, stacklevel=2)
        WARNED = True
    raise NotImplementedError


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:  # noqa: ARG001
    """Derive standard content blocks from a chunk with Bedrock content."""
    global WARNED  # noqa: PLW0603
    if not WARNED:
        warning_message = (
            "Content block standardization is not yet fully supported for Bedrock."
        )
        warnings.warn(warning_message, stacklevel=2)
        WARNED = True
    raise NotImplementedError


def _register_bedrock_translator() -> None:
    """Register the Bedrock translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator("bedrock", translate_content, translate_content_chunk)


_register_bedrock_translator()
