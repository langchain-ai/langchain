"""Derivations of standard content blocks from Google (GenAI) content."""

import re
from collections.abc import Iterable
from typing import Any, Optional, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.content import Citation, create_citation


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


def translate_citations_to_grounding_metadata(
    citations: list[Citation], web_search_queries: Optional[list[str]] = None
) -> dict[str, Any]:
    """Translate LangChain Citations to Google AI grounding metadata format.

    Args:
        citations: List of Citation content blocks.
        web_search_queries: Optional list of search queries that generated
            the grounding data.

    Returns:
        Google AI grounding metadata dictionary.

    Example:
        >>> citations = [
        ...     create_citation(
        ...         url="https://uefa.com/euro2024",
        ...         title="UEFA Euro 2024 Results",
        ...         start_index=0,
        ...         end_index=47,
        ...         cited_text="Spain won the UEFA Euro 2024 championship",
        ...     )
        ... ]
        >>> metadata = translate_citations_to_grounding_metadata(citations)
        >>> len(metadata["groundingChunks"])
        1
        >>> metadata["groundingChunks"][0]["web"]["uri"]
        'https://uefa.com/euro2024'
    """
    if not citations:
        return {}

    # Group citations by text segment (start_index, end_index, cited_text)
    segment_to_citations: dict[
        tuple[Optional[int], Optional[int], Optional[str]], list[Citation]
    ] = {}

    for citation in citations:
        key = (
            citation.get("start_index"),
            citation.get("end_index"),
            citation.get("cited_text"),
        )
        if key not in segment_to_citations:
            segment_to_citations[key] = []
        segment_to_citations[key].append(citation)

    # Build grounding chunks from unique URLs
    url_to_chunk_index: dict[str, int] = {}
    grounding_chunks: list[dict[str, Any]] = []

    for citation in citations:
        url = citation.get("url")
        if url and url not in url_to_chunk_index:
            url_to_chunk_index[url] = len(grounding_chunks)
            grounding_chunks.append(
                {"web": {"uri": url, "title": citation.get("title", "")}}
            )

    # Build grounding supports
    grounding_supports: list[dict[str, Any]] = []

    for (
        start_index,
        end_index,
        cited_text,
    ), citations_group in segment_to_citations.items():
        if start_index is not None and end_index is not None and cited_text:
            chunk_indices = []
            confidence_scores = []

            for citation in citations_group:
                url = citation.get("url")
                if url and url in url_to_chunk_index:
                    chunk_indices.append(url_to_chunk_index[url])

                    # Extract confidence scores from extras if available
                    extras = citation.get("extras", {})
                    google_metadata = extras.get("google_ai_metadata", {})
                    scores = google_metadata.get("confidence_scores", [])
                    confidence_scores.extend(scores)

            support = {
                "segment": {
                    "startIndex": start_index,
                    "endIndex": end_index,
                    "text": cited_text,
                },
                "groundingChunkIndices": chunk_indices,
            }

            if confidence_scores:
                support["confidenceScores"] = confidence_scores

            grounding_supports.append(support)

    # Extract search queries from extras if not provided
    if web_search_queries is None:
        web_search_queries = []
        for citation in citations:
            extras = citation.get("extras", {})
            google_metadata = extras.get("google_ai_metadata", {})
            queries = google_metadata.get("web_search_queries", [])
            web_search_queries.extend(queries)
        # Remove duplicates while preserving order
        web_search_queries = list(dict.fromkeys(web_search_queries))

    return {
        "webSearchQueries": web_search_queries,
        "groundingChunks": grounding_chunks,
        "groundingSupports": grounding_supports,
    }


def _convert_to_v1_from_genai_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Attempt to unpack non-standard blocks."""

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
                # Probably a document of some kind
                pass

            elif (
                num_keys == 1
                and (image := block.get("image"))
                and isinstance(image, dict)
                and "format" in image
            ):
                # Probably an image of some kind
                pass

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
    """Convert Google (GenAI) input message (generativelanguage_v1beta) to v1 format.

    Args:
        message: The input message in Google (GenAI generativelanguage_v1beta) format.

    Returns:
        List of content blocks in v1 format.
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
                # Convert thinking to reasoning block
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": item.get("thinking", ""),
                }
                # Add signature if available in extras
                if "signature" in item.get("extras", {}):
                    reasoning_block["extras"] = {
                        "signature": item["extras"]["signature"]
                    }
                standard_blocks.append(reasoning_block)

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
                    # URL-based image, keep as non-standard for now? TODO
                    standard_blocks.append({"type": "non_standard", "value": item})
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
                text_block = cast("types.TextContentBlock", block)
                text_block["annotations"] = citations
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
