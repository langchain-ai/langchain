"""Derivations of standard content blocks from Google (GenAI) content."""

import base64
import re
from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.content import Citation, create_citation


def _bytes_to_b64_str(bytes_: bytes) -> str:
    """Convert bytes to base64 encoded string."""
    return base64.b64encode(bytes_).decode("utf-8")


def translate_grounding_metadata_to_citations(
    grounding_metadata: dict[str, Any],
) -> list[Citation]:
    """Translate Google AI grounding metadata to LangChain Citations.

    Args:
        grounding_metadata: Google AI grounding metadata containing web search
            queries, grounding chunks, and grounding supports.

    Returns:
        List of Citation content blocks derived from the grounding metadata.

    Example:
        >>> metadata = {
        ...     "webSearchQueries": ["UEFA Euro 2024 winner"],
        ...     "groundingChunks": [
        ...         {
        ...             "web": {
        ...                 "uri": "https://uefa.com/euro2024",
        ...                 "title": "UEFA Euro 2024 Results",
        ...             }
        ...         }
        ...     ],
        ...     "groundingSupports": [
        ...         {
        ...             "segment": {
        ...                 "startIndex": 0,
        ...                 "endIndex": 47,
        ...                 "text": "Spain won the UEFA Euro 2024 championship",
        ...             },
        ...             "groundingChunkIndices": [0],
        ...         }
        ...     ],
        ... }
        >>> citations = translate_grounding_metadata_to_citations(metadata)
        >>> len(citations)
        1
        >>> citations[0]["url"]
        'https://uefa.com/euro2024'
    """
    if not grounding_metadata:
        return []

    grounding_chunks = grounding_metadata.get("groundingChunks", [])
    grounding_supports = grounding_metadata.get("groundingSupports", [])
    web_search_queries = grounding_metadata.get("webSearchQueries", [])

    citations: list[Citation] = []

    for support in grounding_supports:
        segment = support.get("segment", {})
        chunk_indices = support.get("groundingChunkIndices", [])

        start_index = segment.get("startIndex")
        end_index = segment.get("endIndex")
        cited_text = segment.get("text")

        # Create a citation for each referenced chunk
        for chunk_index in chunk_indices:
            if chunk_index < len(grounding_chunks):
                chunk = grounding_chunks[chunk_index]
                web_info = chunk.get("web", {})

                citation = create_citation(
                    url=web_info.get("uri"),
                    title=web_info.get("title"),
                    start_index=start_index,
                    end_index=end_index,
                    cited_text=cited_text,
                    extras={
                        "google_ai_metadata": {
                            "web_search_queries": web_search_queries,
                            "grounding_chunk_index": chunk_index,
                            "confidence_scores": support.get("confidenceScores", []),
                        }
                    },
                )
                citations.append(citation)

    return citations


def _convert_to_v1_from_genai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Helper function for generic structural transformations of content blocks.

    Handles format normalization and post-processing cleanup of content blocks that have
    already been extracted from messages. It focuses on:

    - Processing standard content types that could appear across providers
    - Post-processing cleanup of non-standard blocks
    - Generic structural transformations of already-extracted blocks
    - Format normalization (e.g. base64 conversion, MIME type handling)

    Non-standard blocks are unpacked and analyzed to determine if they represent
    known content types like text, documents, or images that can be converted to
    standard v1 format.

    Args:
        content: List of content blocks that may contain non-standard blocks to unpack.

    Returns:
        List of content blocks with non-standard blocks converted to standard types
        where possible, or preserved as non-standard if conversion is not feasible.

    Note:
        This is a helper function for post-processing content blocks. For converting
        full Google GenAI messages with provider-specific semantics, use
        ``_convert_to_v1_from_genai`` instead.
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
            block_type = block.get("type")

            if num_keys == 1 and (text := block.get("text")):
                # This is probably a TextContentBlock
                yield {"type": "text", "text": text}

            elif (
                num_keys == 1
                and (document := block.get("document"))
                and isinstance(document, dict)
                and "format" in document
            ):
                # Handle document format conversion
                doc_format = document.get("format")
                source = document.get("source", {})

                if doc_format == "pdf" and "bytes" in source:
                    # PDF document with byte data
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": "application/pdf",
                    }
                    # Preserve extra fields
                    extras = {
                        key: value
                        for key, value in document.items()
                        if key not in {"format", "source"}
                    }
                    if extras:
                        file_block["extras"] = extras
                    yield file_block

                elif doc_format == "txt" and "text" in source:
                    # Text document
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": source["text"],
                        "mime_type": "text/plain",
                    }
                    # Preserve extra fields
                    extras = {
                        key: value
                        for key, value in document.items()
                        if key not in {"format", "source"}
                    }
                    if extras:
                        plain_text_block["extras"] = extras
                    yield plain_text_block

                else:
                    # Unknown document format
                    yield {"type": "non_standard", "value": block}

            elif (
                num_keys == 1
                and (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                # Handle image format conversion
                img_format = image.get("format")
                source = image.get("source", {})

                if "bytes" in source:
                    # Image with byte data
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": f"image/{img_format}",
                    }
                    # Preserve extra fields
                    extras = {}
                    for key, value in image.items():
                        if key not in {"format", "source"}:
                            extras[key] = value
                    if extras:
                        image_block["extras"] = extras
                    yield image_block

                else:
                    # Image without byte data
                    yield {"type": "non_standard", "value": block}

            elif block_type == "file_data" and "file_uri" in block:
                # Handle FileData URI-based content
                uri_file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": block["file_uri"],
                }
                if mime_type := block.get("mime_type"):
                    uri_file_block["mime_type"] = mime_type
                yield uri_file_block

            elif block_type == "function_call" and "name" in block:
                # Handle function calls
                tool_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "name": block["name"],
                    "args": block.get("args", {}),
                    "id": block.get("id", ""),
                }
                yield tool_call_block

            elif block_type == "function_response" and "name" in block:
                # Handle function responses - use NonStandardContentBlock for now
                # TODO: Add proper ToolResult type to standard types
                yield {
                    "type": "non_standard",
                    "value": {
                        "type": "tool_result",
                        "name": block["name"],
                        "result": block.get("response", {}),
                        "id": block.get("id", ""),
                    },
                }

            elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                # We see a standard block type, so we just cast it, even if
                # we don't fully understand it. This may be dangerous, but
                # it's better than losing information.
                yield cast("types.ContentBlock", block)

            else:
                # We don't understand this block at all.
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _convert_to_v1_from_genai(message: AIMessage) -> list[types.ContentBlock]:
    """Main function for converting GoogleGenAI messages with Google-specific semantics.

    This function handles the complete conversion of Google GenAI messages to standard
    v1 content blocks. It focuses on:

    - Provider-specific message structure and semantic content types
    - Processing content that requires message context (e.g. tool calls, streaming)
    - Handling Google-specific block types that don't exist in other providers
    - Managing metadata integration (citations, grounding, generation_info)

    Converts messages from Google's generativelanguage_v1beta API format to LangChain's
    standardized v1 format, handling various Google GenAI specific content types
    including text, thinking (reasoning), executable code, code execution results, and
    image content. Also processes grounding metadata to create citation annotations for
    text blocks.

    Args:
        message: The LangChain message to convert that has Google GenAI content. Can
            contain string content, list content with various block types, or mixed
            content structures.

    Returns:
        List of standardized v1 content blocks. Supported block types include text,
        reasoning, image, and non-standard blocks for unrecognized content. Text
        blocks may include citation annotations derived from grounding metadata.

    Note:
        This is the main entry point for converting complete Google GenAI messages.
        For post-processing already-extracted content blocks, use
        ``_convert_to_v1_from_genai_input`` instead.
    """
    if isinstance(message.content, str):
        # String content -> TextContentBlock
        return [{"type": "text", "text": message.content}]

    # TODO: handle dictionary content that is not a list? e.g. a text-dict style

    if not isinstance(message.content, list):
        # Unexpected content type, return as is
        return [{"type": "text", "text": str(message.content)}]

    standard_blocks: list[types.ContentBlock] = []

    for item in message.content:
        if isinstance(item, str):
            standard_blocks.append({"type": "text", "text": item})  # TextContentBlock

        elif isinstance(item, dict):
            item_type = item.get("type")

            if item_type == "text":
                # Ensure `text` key exists and is a string
                text = item.get("text", "")
                if isinstance(text, str):
                    standard_blocks.append({"type": "text", "text": text})
                else:  # Fallback
                    standard_blocks.append({"type": "non_standard", "value": item})

            elif item_type == "thinking":
                # DEPRECATED: Legacy handling for custom 'thinking' v0 LangChain type
                # This maintains backwards compatibility with old LangChain installs
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("thinking", ""),
                }
                # Signature was never available for 'thinking' blocks
                standard_blocks.append(reasoning_block)

            elif item_type == "thought":
                thought_reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("text", ""),
                }
                # Add thought signature if available; required to pass block back in
                if "thought_signature" in item:
                    thought_reasoning_block["extras"] = {
                        "signature": item["thought_signature"]
                    }
                standard_blocks.append(thought_reasoning_block)

            elif item_type == "executable_code":
                # Convert to non-standard block for code execution
                # TODO: migrate to std server tool block
                standard_blocks.append(
                    {
                        "type": "non_standard",
                        "value": {
                            "type": "executable_code",
                            "executable_code": item.get("executable_code", ""),
                            "language": item.get("language", ""),
                        },
                    }
                )

            elif item_type == "code_execution_result":
                # Convert to non-standard block for execution result
                # TODO: migrate to std server tool block
                standard_blocks.append(
                    {
                        "type": "non_standard",
                        "value": {
                            "type": "code_execution_result",
                            "code_execution_result": item.get(
                                "code_execution_result", ""
                            ),
                            "outcome": item.get("outcome", ""),
                        },
                    }
                )

            elif item_type == "image_url":
                # Convert image_url to standard image block
                image_url = item.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    # Extract base64 data
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                        mime_type, base64_data = match.groups()
                        standard_blocks.append(
                            {
                                "type": "image",
                                "base64": base64_data,
                                "mime_type": mime_type,
                            }
                        )
                    else:
                        standard_blocks.append({"type": "non_standard", "value": item})
                else:
                    # TODO: URL-based image, keep as non-standard for now?
                    standard_blocks.append({"type": "non_standard", "value": item})

            elif item_type == "function_call":
                # Handle Google GenAI function calls
                tool_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "args": item.get("args", {}),
                    "id": item.get("id", ""),
                }
                standard_blocks.append(tool_call_block)

            elif item_type == "function_response":
                # Handle Google GenAI function responses - use non-standard for now
                # TODO: Add proper ToolResult type to standard types
                standard_blocks.append(
                    {
                        "type": "non_standard",
                        "value": {
                            "type": "tool_result",
                            "name": item.get("name", ""),
                            "result": item.get("response", {}),
                            "id": item.get("id", ""),
                        },
                    }
                )

            elif item_type == "file_data":
                # Handle FileData URI-based content
                file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": item.get("file_uri", ""),
                }
                if mime_type := item.get("mime_type"):
                    file_block["mime_type"] = mime_type
                standard_blocks.append(file_block)

            else:
                # Unknown type, preserve as non-standard
                standard_blocks.append({"type": "non_standard", "value": item})
        else:
            # Non-dict, non-string content
            standard_blocks.append({"type": "non_standard", "value": item})

    # Handle grounding metadata from generation_info if present
    generation_info = getattr(message, "generation_info", {})
    grounding_metadata = generation_info.get("grounding_metadata")

    if grounding_metadata:
        citations = translate_grounding_metadata_to_citations(grounding_metadata)

        # Add citations to text blocks
        for block in standard_blocks:
            if block["type"] == "text" and citations:
                block["annotations"] = list(citations)
                break

    return standard_blocks


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Google (GenAI) content."""
    return _convert_to_v1_from_genai(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a chunk with Google (GenAI) content."""
    return _convert_to_v1_from_genai(message)


def _register_google_genai_translator() -> None:
    """Register the Google (GenAI) translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_genai", translate_content, translate_content_chunk)


_register_google_genai_translator()
