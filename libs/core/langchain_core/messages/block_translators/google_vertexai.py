"""Derivations of standard content blocks from Google (VertexAI) content."""

from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types

WARNED = False


def _convert_to_v1_from_vertexai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert Google (VertexAI) format blocks to v1 format.

    Called when message isn't an `AIMessage` or `model_provider` isn't set on
    `response_metadata`.

    During the `.content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a ``'non_standard'`` block with the original block stored in the ``value``
    field. This function attempts to unpack those blocks and convert any blocks that
    might be GenAI format to v1 ContentBlocks.

    If conversion fails, the block is left as a ``'non_standard'`` block.

    Args:
        content: List of content blocks to process.

    Returns:
        Updated list with VertexAI blocks converted to v1 format.
    """

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        blocks: list[dict[str, Any]] = [
            cast("dict[str, Any]", block)
            if block.get("type") != "non_standard"
            else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
            for block in content
        ]
        for block in blocks:
            num_keys = len(block)

            if num_keys == 1 and (text := block.get("text")):
                # This is probably a TextContentBlock
                yield {"type": "text", "text": text}

            elif (
                num_keys == 1
                and (document := block.get("document"))
                and isinstance(document, dict)
                and "format" in document
            ):
                # Probably a document of some kind - TODO
                yield {"type": "non_standard", "value": block}

            elif (
                num_keys == 1
                and (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                # Probably an image of some kind - TODO
                yield {"type": "non_standard", "value": block}

            elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                # We see a standard block type, so we just cast it, even if
                # we don't fully understand it. This may be dangerous, but
                # it's better than losing information.
                yield cast("types.ContentBlock", block)

            else:
                # We don't understand this block at all.
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _convert_to_v1_from_vertexai(message: AIMessage) -> list[types.ContentBlock]:
    """Convert Google (VertexAI) message content to v1 format.

    Calling `.content_blocks` on an `AIMessage` where `response_metadata.model_provider`
    is set to `'google_vertexai'` will invoke this function to parse the content into
    standard content blocks for returning.

    Args:
        message: The AIMessage or AIMessageChunk to convert.

    Returns:
        List of standard content blocks derived from the message content.
    """
    return message  # type: ignore[return-value]


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Google (VertexAI) content."""
    return _convert_to_v1_from_vertexai(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a chunk with Google (VertexAI) content."""
    return _convert_to_v1_from_vertexai(message)


def _register_google_vertexai_translator() -> None:
    """Register the Google (VertexAI) translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_vertexai", translate_content, translate_content_chunk)


_register_google_vertexai_translator()
