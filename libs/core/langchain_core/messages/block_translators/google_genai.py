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
        ...     "web_search_queries": ["UEFA Euro 2024 winner"],
        ...     "grounding_chunks": [
        ...         {
        ...             "web": {
        ...                 "uri": "https://uefa.com/euro2024",
        ...                 "title": "UEFA Euro 2024 Results",
        ...             }
        ...         }
        ...     ],
        ...     "grounding_supports": [
        ...         {
        ...             "segment": {
        ...                 "start_index": 0,
        ...                 "end_index": 47,
        ...                 "text": "Spain won the UEFA Euro 2024 championship",
        ...             },
        ...             "grounding_chunk_indices": [0],
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

    grounding_chunks = grounding_metadata.get("grounding_chunks", [])
    grounding_supports = grounding_metadata.get("grounding_supports", [])
    web_search_queries = grounding_metadata.get("web_search_queries", [])

    citations: list[Citation] = []

    for support in grounding_supports:
        segment = support.get("segment", {})
        chunk_indices = support.get("grounding_chunk_indices", [])

        start_index = segment.get("start_index")
        end_index = segment.get("end_index")
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
                            "confidence_scores": support.get("confidence_scores", []),
                        }
                    },
                )
                citations.append(citation)

    return citations


def _convert_to_v1_from_genai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert Google GenAI format blocks to v1 format.

    Called when message isn't an `AIMessage` or `model_provider` isn't set on
    `response_metadata`.

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any blocks that
    might be GenAI format to v1 ContentBlocks.

    If conversion fails, the block is left as a `'non_standard'` block.

    Args:
        content: List of content blocks to process.

    Returns:
        Updated list with GenAI blocks converted to v1 format.
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

            elif block_type == "executable_code":
                server_tool_call_input: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": block.get("executable_code", ""),
                        "language": block.get("language", "python"),
                    },
                    "id": block.get("id", ""),
                }
                yield server_tool_call_input

            elif block_type == "code_execution_result":
                outcome = block.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result_input: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": block.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": block.get("code_execution_result", ""),
                }
                if outcome is not None:
                    server_tool_result_input["extras"] = {"outcome": outcome}
                yield server_tool_result_input

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
    """Convert Google GenAI message content to v1 format.

    Calling `.content_blocks` on an `AIMessage` where `response_metadata.model_provider`
    is set to `'google_genai'` will invoke this function to parse the content into
    standard content blocks for returning.

    Args:
        message: The `AIMessage` or `AIMessageChunk` to convert.

    Returns:
        List of standard content blocks derived from the message content.
    """
    if isinstance(message.content, str):
        # String content -> TextContentBlock (only add if non-empty in case of audio)
        string_blocks: list[types.ContentBlock] = []
        if message.content:
            string_blocks.append({"type": "text", "text": message.content})

        # Add any missing tool calls from message.tool_calls field
        content_tool_call_ids = {
            block.get("id")
            for block in string_blocks
            if isinstance(block, dict) and block.get("type") == "tool_call"
        }
        for tool_call in message.tool_calls:
            id_ = tool_call.get("id")
            if id_ and id_ not in content_tool_call_ids:
                string_tool_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "id": id_,
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                }
                string_blocks.append(string_tool_call_block)

        # Handle audio from additional_kwargs if present (for empty content cases)
        audio_data = message.additional_kwargs.get("audio")
        if audio_data and isinstance(audio_data, bytes):
            audio_block: types.AudioContentBlock = {
                "type": "audio",
                "base64": _bytes_to_b64_str(audio_data),
                "mime_type": "audio/wav",  # Default to WAV for Google GenAI
            }
            string_blocks.append(audio_block)

        grounding_metadata = message.response_metadata.get("grounding_metadata")
        if grounding_metadata:
            citations = translate_grounding_metadata_to_citations(grounding_metadata)

            for block in string_blocks:
                if block["type"] == "text" and citations:
                    # Add citations to the first text block only
                    block["annotations"] = cast("list[types.Annotation]", citations)
                    break

        return string_blocks

    if not isinstance(message.content, list):
        # Unexpected content type, attempt to represent as text
        return [{"type": "text", "text": str(message.content)}]

    converted_blocks: list[types.ContentBlock] = []

    for item in message.content:
        if isinstance(item, str):
            # Conversation history strings

            # Citations are handled below after all blocks are converted
            converted_blocks.append({"type": "text", "text": item})  # TextContentBlock

        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "image_url":
                # Convert image_url to standard image block (base64)
                # (since the original implementation returned as url-base64 CC style)
                image_url = item.get("image_url", {})
                url = image_url.get("url", "")
                if url:
                    # Extract base64 data
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                        # Data URI provided
                        mime_type, base64_data = match.groups()
                        converted_blocks.append(
                            {
                                "type": "image",
                                "base64": base64_data,
                                "mime_type": mime_type,
                            }
                        )
                    else:
                        # Assume it's raw base64 without data URI
                        try:
                            # Validate base64 and decode for mime type detection
                            decoded_bytes = base64.b64decode(url, validate=True)

                            image_url_b64_block = {
                                "type": "image",
                                "base64": url,
                            }

                            try:
                                import filetype  # type: ignore[import-not-found] # noqa: PLC0415

                                # Guess mime type based on file bytes
                                mime_type = None
                                kind = filetype.guess(decoded_bytes)
                                if kind:
                                    mime_type = kind.mime
                                if mime_type:
                                    image_url_b64_block["mime_type"] = mime_type
                            except ImportError:
                                # filetype library not available, skip type detection
                                pass

                            converted_blocks.append(
                                cast("types.ImageContentBlock", image_url_b64_block)
                            )
                        except Exception:
                            # Not valid base64, treat as non-standard
                            converted_blocks.append(
                                {"type": "non_standard", "value": item}
                            )
                else:
                    # This likely won't be reached according to previous implementations
                    converted_blocks.append({"type": "non_standard", "value": item})
                    msg = "Image URL not a data URI; appending as non-standard block."
                    raise ValueError(msg)
            elif item_type == "function_call":
                # Handle Google GenAI function calls
                function_call_block: types.ToolCall = {
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "args": item.get("args", {}),
                    "id": item.get("id", ""),
                }
                converted_blocks.append(function_call_block)
            elif item_type == "file_data":
                # Handle FileData URI-based content
                file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": item.get("file_uri", ""),
                }
                if mime_type := item.get("mime_type"):
                    file_block["mime_type"] = mime_type
                converted_blocks.append(file_block)
            elif item_type == "thinking":
                # Handling for the 'thinking' type we package thoughts as
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("thinking", ""),
                }
                if signature := item.get("signature"):
                    reasoning_block["extras"] = {"signature": signature}

                converted_blocks.append(reasoning_block)
            elif item_type == "executable_code":
                # Convert to standard server tool call block at the moment
                server_tool_call_block: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": item.get("executable_code", ""),
                        "language": item.get("language", "python"),  # Default to python
                    },
                    "id": item.get("id", ""),
                }
                converted_blocks.append(server_tool_call_block)
            elif item_type == "code_execution_result":
                # Map outcome to status: OUTCOME_OK (1) → success, else → error
                outcome = item.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result_block: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": item.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": item.get("code_execution_result", ""),
                }
                server_tool_result_block["extras"] = {"block_type": item_type}
                # Preserve original outcome in extras
                if outcome is not None:
                    server_tool_result_block["extras"]["outcome"] = outcome
                converted_blocks.append(server_tool_result_block)
            else:
                # Unknown type, preserve as non-standard
                converted_blocks.append({"type": "non_standard", "value": item})
        else:
            # Non-dict, non-string content
            converted_blocks.append({"type": "non_standard", "value": item})

    grounding_metadata = message.response_metadata.get("grounding_metadata")
    if grounding_metadata:
        citations = translate_grounding_metadata_to_citations(grounding_metadata)

        for block in converted_blocks:
            if block["type"] == "text" and citations:
                # Add citations to text blocks (only the first text block)
                block["annotations"] = cast("list[types.Annotation]", citations)
                break

    # Audio is stored on the message.additional_kwargs
    audio_data = message.additional_kwargs.get("audio")
    if audio_data and isinstance(audio_data, bytes):
        audio_block_kwargs: types.AudioContentBlock = {
            "type": "audio",
            "base64": _bytes_to_b64_str(audio_data),
            "mime_type": "audio/wav",  # Default to WAV for Google GenAI
        }
        converted_blocks.append(audio_block_kwargs)

    # Add any missing tool calls from message.tool_calls field
    content_tool_call_ids = {
        block.get("id")
        for block in converted_blocks
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    for tool_call in message.tool_calls:
        id_ = tool_call.get("id")
        if id_ and id_ not in content_tool_call_ids:
            missing_tool_call_block: types.ToolCall = {
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            converted_blocks.append(missing_tool_call_block)

    return converted_blocks


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
