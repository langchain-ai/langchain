"""Derivations of standard content blocks from Amazon (Bedrock Converse) content."""

import base64
from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _populate_extras(
    standard_block: types.ContentBlock, block: dict[str, Any], known_fields: set[str]
) -> types.ContentBlock:
    """Mutate a block, populating extras."""
    if standard_block.get("type") == "non_standard":
        return standard_block

    for key, value in block.items():
        if key not in known_fields:
            if "extras" not in standard_block:
                # Below type-ignores are because mypy thinks a non-standard block can
                # get here, although we exclude them above.
                standard_block["extras"] = {}  # type: ignore[typeddict-unknown-key]
            standard_block["extras"][key] = value  # type: ignore[typeddict-item]

    return standard_block


def _convert_to_v1_from_converse_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert Bedrock Converse format blocks to v1 format.

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any blocks that
    might be Converse format to v1 ContentBlocks.

    If conversion fails, the block is left as a `'non_standard'` block.

    Args:
        content: List of content blocks to process.

    Returns:
        Updated list with Converse blocks converted to v1 format.
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
                yield {"type": "text", "text": text}

            elif (
                num_keys == 1
                and (document := block.get("document"))
                and isinstance(document, dict)
                and "format" in document
            ):
                if document.get("format") == "pdf":
                    if "bytes" in document.get("source", {}):
                        file_block: types.FileContentBlock = {
                            "type": "file",
                            "base64": _bytes_to_b64_str(document["source"]["bytes"]),
                            "mime_type": "application/pdf",
                        }
                        _populate_extras(file_block, document, {"format", "source"})
                        yield file_block

                    else:
                        yield {"type": "non_standard", "value": block}

                elif document["format"] == "txt":
                    if "text" in document.get("source", {}):
                        plain_text_block: types.PlainTextContentBlock = {
                            "type": "text-plain",
                            "text": document["source"]["text"],
                            "mime_type": "text/plain",
                        }
                        _populate_extras(
                            plain_text_block, document, {"format", "source"}
                        )
                        yield plain_text_block
                    else:
                        yield {"type": "non_standard", "value": block}

                else:
                    yield {"type": "non_standard", "value": block}

            elif (
                num_keys == 1
                and (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                if "bytes" in image.get("source", {}):
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": _bytes_to_b64_str(image["source"]["bytes"]),
                        "mime_type": f"image/{image['format']}",
                    }
                    _populate_extras(image_block, image, {"format", "source"})
                    yield image_block

                else:
                    yield {"type": "non_standard", "value": block}

            elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)

            else:
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _convert_citation_to_v1(citation: dict[str, Any]) -> types.Annotation:
    standard_citation: types.Citation = {"type": "citation"}
    if "title" in citation:
        standard_citation["title"] = citation["title"]
    if (
        (source_content := citation.get("source_content"))
        and isinstance(source_content, list)
        and all(isinstance(item, dict) for item in source_content)
    ):
        standard_citation["cited_text"] = "".join(
            item.get("text", "") for item in source_content
        )

    known_fields = {"type", "source_content", "title", "index", "extras"}

    for key, value in citation.items():
        if key not in known_fields:
            if "extras" not in standard_citation:
                standard_citation["extras"] = {}
            standard_citation["extras"][key] = value

    return standard_citation


def _convert_to_v1_from_converse(message: AIMessage) -> list[types.ContentBlock]:
    """Convert Bedrock Converse message content to v1 format."""
    if (
        message.content == ""
        and not message.additional_kwargs
        and not message.tool_calls
    ):
        # Converse outputs multiple chunks containing response metadata
        return []

    if isinstance(message.content, str):
        message.content = [{"type": "text", "text": message.content}]

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        for block in message.content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                if citations := block.get("citations"):
                    text_block: types.TextContentBlock = {
                        "type": "text",
                        "text": block.get("text", ""),
                        "annotations": [_convert_citation_to_v1(a) for a in citations],
                    }
                else:
                    text_block = {"type": "text", "text": block["text"]}
                if "index" in block:
                    text_block["index"] = block["index"]
                yield text_block

            elif block_type == "reasoning_content":
                reasoning_block: types.ReasoningContentBlock = {"type": "reasoning"}
                if reasoning_content := block.get("reasoning_content"):
                    if reasoning := reasoning_content.get("text"):
                        reasoning_block["reasoning"] = reasoning
                    if signature := reasoning_content.get("signature"):
                        if "extras" not in reasoning_block:
                            reasoning_block["extras"] = {}
                        reasoning_block["extras"]["signature"] = signature

                if "index" in block:
                    reasoning_block["index"] = block["index"]

                known_fields = {"type", "reasoning_content", "index", "extras"}
                for key in block:
                    if key not in known_fields:
                        if "extras" not in reasoning_block:
                            reasoning_block["extras"] = {}
                        reasoning_block["extras"][key] = block[key]
                yield reasoning_block

            elif block_type == "tool_use":
                if (
                    isinstance(message, AIMessageChunk)
                    and len(message.tool_call_chunks) == 1
                    and message.chunk_position != "last"
                ):
                    # Isolated chunk
                    chunk = message.tool_call_chunks[0]
                    tool_call_chunk = types.ToolCallChunk(
                        name=chunk.get("name"),
                        id=chunk.get("id"),
                        args=chunk.get("args"),
                        type="tool_call_chunk",
                    )
                    index = chunk.get("index")
                    if index is not None:
                        tool_call_chunk["index"] = index
                    yield tool_call_chunk
                else:
                    tool_call_block: types.ToolCall | None = None
                    # Non-streaming or gathered chunk
                    if len(message.tool_calls) == 1:
                        tool_call_block = {
                            "type": "tool_call",
                            "name": message.tool_calls[0]["name"],
                            "args": message.tool_calls[0]["args"],
                            "id": message.tool_calls[0].get("id"),
                        }
                    elif call_id := block.get("id"):
                        for tc in message.tool_calls:
                            if tc.get("id") == call_id:
                                tool_call_block = {
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "id": tc.get("id"),
                                }
                                break
                    if not tool_call_block:
                        tool_call_block = {
                            "type": "tool_call",
                            "name": block.get("name", ""),
                            "args": block.get("input", {}),
                            "id": block.get("id", ""),
                        }
                    if "index" in block:
                        tool_call_block["index"] = block["index"]
                    yield tool_call_block

            elif (
                block_type == "input_json_delta"
                and isinstance(message, AIMessageChunk)
                and len(message.tool_call_chunks) == 1
            ):
                chunk = message.tool_call_chunks[0]
                tool_call_chunk = types.ToolCallChunk(
                    name=chunk.get("name"),
                    id=chunk.get("id"),
                    args=chunk.get("args"),
                    type="tool_call_chunk",
                )
                index = chunk.get("index")
                if index is not None:
                    tool_call_chunk["index"] = index
                yield tool_call_chunk

            else:
                new_block: types.NonStandardContentBlock = {
                    "type": "non_standard",
                    "value": block,
                }
                if "index" in new_block["value"]:
                    new_block["index"] = new_block["value"].pop("index")
                yield new_block

    return list(_iter_blocks())


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Bedrock Converse content.

    Args:
        message: The message to translate.

    Returns:
        The derived content blocks.
    """
    return _convert_to_v1_from_converse(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a chunk with Bedrock Converse content.

    Args:
        message: The message chunk to translate.

    Returns:
        The derived content blocks.
    """
    return _convert_to_v1_from_converse(message)


def _register_bedrock_converse_translator() -> None:
    """Register the Bedrock Converse translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("bedrock_converse", translate_content, translate_content_chunk)


_register_bedrock_converse_translator()
