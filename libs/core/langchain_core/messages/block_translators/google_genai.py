"""Derivations of standard content blocks from Google (GenAI) content."""

import base64
import re
from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.content import Citation, create_citation

try:
    import filetype  # type: ignore[import-not-found]

    _HAS_FILETYPE = True
except ImportError:
    _HAS_FILETYPE = False


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

                # Handle web and maps grounding
                web_info = chunk.get("web") or {}
                maps_info = chunk.get("maps") or {}

                # Extract citation info depending on source
                url = maps_info.get("uri") or web_info.get("uri")
                title = maps_info.get("title") or web_info.get("title")

                # Note: confidence_scores is a legacy field from Gemini 2.0 and earlier
                # that indicated confidence (0.0-1.0) for each grounding chunk.
                #
                # In Gemini 2.5+, this field is always None/empty and should be ignored.
                extras_metadata = {
                    "web_search_queries": web_search_queries,
                    "grounding_chunk_index": chunk_index,
                    "confidence_scores": support.get("confidence_scores") or [],
                }

                # Add maps-specific metadata if present
                if maps_info.get("placeId"):
                    extras_metadata["place_id"] = maps_info["placeId"]

                citation = create_citation(
                    url=url,
                    title=title,
                    start_index=start_index,
                    end_index=end_index,
                    cited_text=cited_text,
                    google_ai_metadata=extras_metadata,
                )
                citations.append(citation)

    return citations


def _convert_to_v1_from_genai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert Google GenAI format blocks to v1 format.

    Called when message isn't an `AIMessage` or `model_provider` isn't set on
    `response_metadata`.

    Args:
        content: List of content blocks to process.

    Returns:
        Updated list with GenAI blocks converted to v1 format.
    """

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        blocks: list[dict[str, Any]] = [
            cast("dict[str, Any]", block)
            if block.get("type") != "non_standard"
            else block["value"]  # type: ignore[typeddict-item]
            for block in content
        ]
        for block in blocks:
            block_type = block.get("type")

            # FIX: Remove strict num_keys == 1 check to accommodate metadata/signatures
            if (text := block.get("text")) is not None:
                yield {"type": "text", "text": text}

            elif (
                (document := block.get("document"))
                and isinstance(document, dict)
                and "format" in document
            ):
                doc_format = document.get("format")
                source = document.get("source", {})

                if doc_format == "pdf" and "bytes" in source:
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": "application/pdf",
                    }
                    extras = {
                        k: v for k, v in document.items() if k not in {"format", "source"}
                    }
                    if extras:
                        file_block["extras"] = extras
                    yield file_block

                elif doc_format == "txt" and "text" in source:
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": source["text"],
                        "mime_type": "text/plain",
                    }
                    extras = {
                        k: v for k, v in document.items() if k not in {"format", "source"}
                    }
                    if extras:
                        plain_text_block["extras"] = extras
                    yield plain_text_block
                else:
                    yield {"type": "non_standard", "value": block}

            elif (
                (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                img_format = image.get("format")
                source = image.get("source", {})

                if "bytes" in source:
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": source["bytes"]
                        if isinstance(source["bytes"], str)
                        else _bytes_to_b64_str(source["bytes"]),
                        "mime_type": f"image/{img_format}",
                    }
                    extras = {k: v for k, v in image.items() if k not in {"format", "source"}}
                    if extras:
                        image_block["extras"] = extras
                    yield image_block
                else:
                    yield {"type": "non_standard", "value": block}

            elif block_type == "file_data" and "file_uri" in block:
                uri_file_block: types.FileContentBlock = {
                    "type": "file",
                    "url": block["file_uri"],
                }
                if mime_type := block.get("mime_type"):
                    uri_file_block["mime_type"] = mime_type
                yield uri_file_block

            elif block_type == "function_call" and "name" in block:
                yield {
                    "type": "tool_call",
                    "name": block["name"],
                    "args": block.get("args", {}),
                    "id": block.get("id", ""),
                }

            elif block_type == "executable_code":
                yield {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": block.get("executable_code", ""),
                        "language": block.get("language", "python"),
                    },
                    "id": block.get("id", ""),
                }

            elif block_type == "code_execution_result":
                outcome = block.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": block.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": block.get("code_execution_result", ""),
                }
                if outcome is not None:
                    server_tool_result["extras"] = {"outcome": outcome}
                yield server_tool_result

            elif block.get("type") in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)

            else:
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _convert_to_v1_from_genai(message: AIMessage) -> list[types.ContentBlock]:
    """Convert Google GenAI message content to v1 format."""
    if isinstance(message.content, str):
        string_blocks: list[types.ContentBlock] = []
        if message.content:
            string_blocks.append({"type": "text", "text": message.content})

        content_tool_call_ids = {
            block.get("id")
            for block in string_blocks
            if isinstance(block, dict) and block.get("type") == "tool_call"
        }
        for tool_call in message.tool_calls:
            id_ = tool_call.get("id")
            if id_ and id_ not in content_tool_call_ids:
                string_blocks.append({
                    "type": "tool_call",
                    "id": id_,
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                })

        audio_data = message.additional_kwargs.get("audio")
        if audio_data and isinstance(audio_data, bytes):
            string_blocks.append({
                "type": "audio",
                "base64": _bytes_to_b64_str(audio_data),
                "mime_type": "audio/wav",
            })

        grounding_metadata = message.response_metadata.get("grounding_metadata")
        if grounding_metadata:
            citations = translate_grounding_metadata_to_citations(grounding_metadata)
            for block in string_blocks:
                if block["type"] == "text" and citations:
                    block["annotations"] = cast("list[types.Annotation]", citations)
                    break

        return string_blocks

    if not isinstance(message.content, list):
        return [{"type": "text", "text": str(message.content)}]

    converted_blocks: list[types.ContentBlock] = []

    for item in message.content:
        if isinstance(item, str):
            converted_blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "")
                if url:
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                        mime_type, base64_data = match.groups()
                        converted_blocks.append({
                            "type": "image",
                            "base64": base64_data,
                            "mime_type": mime_type,
                        })
                    else:
                        try:
                            decoded_bytes = base64.b64decode(url, validate=True)
                            image_url_b64_block = {"type": "image", "base64": url}
                            if _HAS_FILETYPE:
                                kind = filetype.guess(decoded_bytes)
                                if kind:
                                    image_url_b64_block["mime_type"] = kind.mime
                            converted_blocks.append(cast("types.ImageContentBlock", image_url_b64_block))
                        except Exception:
                            converted_blocks.append({"type": "non_standard", "value": item})
                else:
                    converted_blocks.append({"type": "non_standard", "value": item})

            elif item_type == "function_call":
                converted_blocks.append({
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "args": item.get("args", {}),
                    "id": item.get("id", ""),
                })
            elif item_type == "file_data":
                file_block: types.FileContentBlock = {"type": "file", "url": item.get("file_uri", "")}
                if mime_type := item.get("mime_type"):
                    file_block["mime_type"] = mime_type
                converted_blocks.append(file_block)
            elif item_type == "thinking":
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("thinking", ""),
                }
                if signature := item.get("signature"):
                    reasoning_block["extras"] = {"signature": signature}
                converted_blocks.append(reasoning_block)
            elif item_type == "executable_code":
                converted_blocks.append({
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "args": {
                        "code": item.get("executable_code", ""),
                        "language": item.get("language", "python"),
                    },
                    "id": item.get("id", ""),
                })
            elif item_type == "code_execution_result":
                outcome = item.get("outcome", 1)
                status = "success" if outcome == 1 else "error"
                server_tool_result: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": item.get("tool_call_id", ""),
                    "status": status,  # type: ignore[typeddict-item]
                    "output": item.get("code_execution_result", ""),
                }
                server_tool_result["extras"] = {"block_type": item_type}
                if outcome is not None:
                    server_tool_result["extras"]["outcome"] = outcome
                converted_blocks.append(server_tool_result)
            elif item_type == "text" or "text" in item:
                # Ensure blocks with 'text' and metadata are treated as text
                converted_blocks.append({
                    "type": "text",
                    "text": item.get("text", ""),
                    **{k: v for k, v in item.items() if k not in {"type", "text"}}
                })
            else:
                converted_blocks.append({"type": "non_standard", "value": item})
        else:
            converted_blocks.append({"type": "non_standard", "value": item})

    grounding_metadata = message.response_metadata.get("grounding_metadata")
    if grounding_metadata:
        citations = translate_grounding_metadata_to_citations(grounding_metadata)
        for block in converted_blocks:
            if block["type"] == "text" and citations:
                block["annotations"] = cast("list[types.Annotation]", citations)
                break

    audio_data = message.additional_kwargs.get("audio")
    if audio_data and isinstance(audio_data, bytes):
        converted_blocks.append({
            "type": "audio",
            "base64": _bytes_to_b64_str(audio_data),
            "mime_type": "audio/wav",
        })

    content_tool_call_ids = {
        block.get("id")
        for block in converted_blocks
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    for tool_call in message.tool_calls:
        id_ = tool_call.get("id")
        if id_ and id_ not in content_tool_call_ids:
            converted_blocks.append({
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            })

    return converted_blocks


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Google (GenAI) content."""
    return _convert_to_v1_from_genai(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a chunk with Google (GenAI) content."""
    return _convert_to_v1_from_genai(message)


def _register_google_genai_translator() -> None:
    """Register the Google (GenAI) translator with the central registry."""
    from langchain_core.messages.block_translators import register_translator
    register_translator("google_genai", translate_content, translate_content_chunk)


_register_google_genai_translator()
