"""Derivations of standard content blocks from OpenAI content."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.language_models._utils import (
    _parse_data_uri,
    is_openai_data_block,
)
from langchain_core.messages import content as types

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.messages import AIMessage, AIMessageChunk


def convert_to_openai_image_block(block: dict[str, Any]) -> dict:
    """Convert `ImageContentBlock` to format expected by OpenAI Chat Completions."""
    if "url" in block:
        return {
            "type": "image_url",
            "image_url": {
                "url": block["url"],
            },
        }
    if "base64" in block or block.get("source_type") == "base64":
        if "mime_type" not in block:
            error_message = "mime_type key is required for base64 data."
            raise ValueError(error_message)
        mime_type = block["mime_type"]
        base64_data = block["data"] if "data" in block else block["base64"]
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_data}",
            },
        }
    error_message = "Unsupported source type. Only 'url' and 'base64' are supported."
    raise ValueError(error_message)


def convert_to_openai_data_block(
    block: dict, api: Literal["chat/completions", "responses"] = "chat/completions"
) -> dict:
    """Format standard data content block to format expected by OpenAI.

    "Standard data content block" can include old-style LangChain v0 blocks
    (URLContentBlock, Base64ContentBlock, IDContentBlock) or new ones.
    """
    if block["type"] == "image":
        chat_completions_block = convert_to_openai_image_block(block)
        if api == "responses":
            formatted_block = {
                "type": "input_image",
                "image_url": chat_completions_block["image_url"]["url"],
            }
            if chat_completions_block["image_url"].get("detail"):
                formatted_block["detail"] = chat_completions_block["image_url"][
                    "detail"
                ]
        else:
            formatted_block = chat_completions_block

    elif block["type"] == "file":
        if block.get("source_type") == "base64" or "base64" in block:
            # Handle v0 format (Base64CB): {"source_type": "base64", "data": "...", ...}
            # Handle v1 format (IDCB): {"base64": "...", ...}
            base64_data = block["data"] if "source_type" in block else block["base64"]
            file = {"file_data": f"data:{block['mime_type']};base64,{base64_data}"}
            if filename := block.get("filename"):
                file["filename"] = filename
            elif (extras := block.get("extras")) and ("filename" in extras):
                file["filename"] = extras["filename"]
            elif (extras := block.get("metadata")) and ("filename" in extras):
                # Backward compat
                file["filename"] = extras["filename"]
            else:
                # Can't infer filename
                warnings.warn(
                    "OpenAI may require a filename for file uploads. Specify a filename"
                    " in the content block, e.g.: {'type': 'file', 'mime_type': "
                    "'...', 'base64': '...', 'filename': 'my-file.pdf'}",
                    stacklevel=1,
                )
            formatted_block = {"type": "file", "file": file}
            if api == "responses":
                formatted_block = {"type": "input_file", **formatted_block["file"]}
        elif block.get("source_type") == "id" or "file_id" in block:
            # Handle v0 format (IDContentBlock): {"source_type": "id", "id": "...", ...}
            # Handle v1 format (IDCB): {"file_id": "...", ...}
            file_id = block["id"] if "source_type" in block else block["file_id"]
            formatted_block = {"type": "file", "file": {"file_id": file_id}}
            if api == "responses":
                formatted_block = {"type": "input_file", **formatted_block["file"]}
        elif "url" in block:  # Intentionally do not check for source_type="url"
            if api == "chat/completions":
                error_msg = "OpenAI Chat Completions does not support file URLs."
                raise ValueError(error_msg)
            # Only supported by Responses API; return in that format
            formatted_block = {"type": "input_file", "file_url": block["url"]}
        else:
            error_msg = "Keys base64, url, or file_id required for file blocks."
            raise ValueError(error_msg)

    elif block["type"] == "audio":
        if "base64" in block or block.get("source_type") == "base64":
            # Handle v0 format: {"source_type": "base64", "data": "...", ...}
            # Handle v1 format: {"base64": "...", ...}
            base64_data = block["data"] if "source_type" in block else block["base64"]
            audio_format = block["mime_type"].split("/")[-1]
            formatted_block = {
                "type": "input_audio",
                "input_audio": {"data": base64_data, "format": audio_format},
            }
        else:
            error_msg = "Key base64 is required for audio blocks."
            raise ValueError(error_msg)
    else:
        error_msg = f"Block of type {block['type']} is not supported."
        raise ValueError(error_msg)

    return formatted_block


# v1 / Chat Completions
def _convert_to_v1_from_chat_completions(
    message: AIMessage,
) -> list[types.ContentBlock]:
    """Mutate a Chat Completions message to v1 format."""
    content_blocks: list[types.ContentBlock] = []
    if isinstance(message.content, str):
        if message.content:
            content_blocks = [{"type": "text", "text": message.content}]
        else:
            content_blocks = []

    for tool_call in message.tool_calls:
        content_blocks.append(
            {
                "type": "tool_call",
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call.get("id"),
            }
        )

    return content_blocks


def _convert_to_v1_from_chat_completions_input(
    content: list[types.ContentBlock],
) -> list[types.ContentBlock]:
    """Convert OpenAI Chat Completions format blocks to v1 format.

    During the `content_blocks` parsing process, we wrap blocks not recognized as a v1
    block as a `'non_standard'` block with the original block stored in the `value`
    field. This function attempts to unpack those blocks and convert any blocks that
    might be OpenAI format to v1 ContentBlocks.

    If conversion fails, the block is left as a `'non_standard'` block.

    Args:
        content: List of content blocks to process.

    Returns:
        Updated list with OpenAI blocks converted to v1 format.
    """
    from langchain_core.messages import content as types  # noqa: PLC0415

    converted_blocks = []
    unpacked_blocks: list[dict[str, Any]] = [
        cast("dict[str, Any]", block)
        if block.get("type") != "non_standard"
        else block["value"]  # type: ignore[typeddict-item]  # this is only non-standard blocks
        for block in content
    ]
    for block in unpacked_blocks:
        if block.get("type") in {
            "image_url",
            "input_audio",
            "file",
        } and is_openai_data_block(block):
            converted_block = _convert_openai_format_to_data_block(block)
            # If conversion succeeded, use it; otherwise keep as non_standard
            if (
                isinstance(converted_block, dict)
                and converted_block.get("type") in types.KNOWN_BLOCK_TYPES
            ):
                converted_blocks.append(cast("types.ContentBlock", converted_block))
            else:
                converted_blocks.append({"type": "non_standard", "value": block})
        elif block.get("type") in types.KNOWN_BLOCK_TYPES:
            converted_blocks.append(cast("types.ContentBlock", block))
        else:
            converted_blocks.append({"type": "non_standard", "value": block})

    return converted_blocks


def _convert_to_v1_from_chat_completions_chunk(
    chunk: AIMessageChunk,
) -> list[types.ContentBlock]:
    """Mutate a Chat Completions chunk to v1 format."""
    content_blocks: list[types.ContentBlock] = []
    if isinstance(chunk.content, str):
        if chunk.content:
            content_blocks = [{"type": "text", "text": chunk.content}]
        else:
            content_blocks = []

    if chunk.chunk_position == "last":
        for tool_call in chunk.tool_calls:
            content_blocks.append(
                {
                    "type": "tool_call",
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call.get("id"),
                }
            )

    else:
        for tool_call_chunk in chunk.tool_call_chunks:
            tc: types.ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tool_call_chunk.get("id"),
                "name": tool_call_chunk.get("name"),
                "args": tool_call_chunk.get("args"),
            }
            if (idx := tool_call_chunk.get("index")) is not None:
                tc["index"] = idx
            content_blocks.append(tc)

    return content_blocks


def _convert_from_v1_to_chat_completions(message: AIMessage) -> AIMessage:
    """Convert a v1 message to the Chat Completions format."""
    if isinstance(message.content, list):
        new_content: list = []
        for block in message.content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    # Strip annotations
                    new_content.append({"type": "text", "text": block["text"]})
                elif block_type in ("reasoning", "tool_call"):
                    pass
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        return message.model_copy(update={"content": new_content})

    return message


# Responses
_FUNCTION_CALL_IDS_MAP_KEY = "__openai_function_call_ids__"


def _convert_from_v03_ai_message(message: AIMessage) -> AIMessage:
    """Convert v0 AIMessage into `output_version="responses/v1"` format."""
    from langchain_core.messages import AIMessageChunk  # noqa: PLC0415

    # Only update ChatOpenAI v0.3 AIMessages
    is_chatopenai_v03 = (
        isinstance(message.content, list)
        and all(isinstance(b, dict) for b in message.content)
    ) and (
        any(
            item in message.additional_kwargs
            for item in [
                "reasoning",
                "tool_outputs",
                "refusal",
                _FUNCTION_CALL_IDS_MAP_KEY,
            ]
        )
        or (
            isinstance(message.id, str)
            and message.id.startswith("msg_")
            and (response_id := message.response_metadata.get("id"))
            and isinstance(response_id, str)
            and response_id.startswith("resp_")
        )
    )
    if not is_chatopenai_v03:
        return message

    content_order = [
        "reasoning",
        "code_interpreter_call",
        "mcp_call",
        "image_generation_call",
        "text",
        "refusal",
        "function_call",
        "computer_call",
        "mcp_list_tools",
        "mcp_approval_request",
        # N. B. "web_search_call" and "file_search_call" were not passed back in
        # in v0.3
    ]

    # Build a bucket for every known block type
    buckets: dict[str, list] = {key: [] for key in content_order}
    unknown_blocks = []

    # Reasoning
    if reasoning := message.additional_kwargs.get("reasoning"):
        if isinstance(message, AIMessageChunk) and message.chunk_position != "last":
            buckets["reasoning"].append({**reasoning, "type": "reasoning"})
        else:
            buckets["reasoning"].append(reasoning)

    # Refusal
    if refusal := message.additional_kwargs.get("refusal"):
        buckets["refusal"].append({"type": "refusal", "refusal": refusal})

    # Text
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            block_copy = block.copy()
            if isinstance(message.id, str) and message.id.startswith("msg_"):
                block_copy["id"] = message.id
            buckets["text"].append(block_copy)
        else:
            unknown_blocks.append(block)

    # Function calls
    function_call_ids = message.additional_kwargs.get(_FUNCTION_CALL_IDS_MAP_KEY)
    if (
        isinstance(message, AIMessageChunk)
        and len(message.tool_call_chunks) == 1
        and message.chunk_position != "last"
    ):
        # Isolated chunk
        tool_call_chunk = message.tool_call_chunks[0]
        function_call = {
            "type": "function_call",
            "name": tool_call_chunk.get("name"),
            "arguments": tool_call_chunk.get("args"),
            "call_id": tool_call_chunk.get("id"),
        }
        if function_call_ids is not None and (
            _id := function_call_ids.get(tool_call_chunk.get("id"))
        ):
            function_call["id"] = _id
        buckets["function_call"].append(function_call)
    else:
        for tool_call in message.tool_calls:
            function_call = {
                "type": "function_call",
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
                "call_id": tool_call["id"],
            }
            if function_call_ids is not None and (
                _id := function_call_ids.get(tool_call["id"])
            ):
                function_call["id"] = _id
            buckets["function_call"].append(function_call)

    # Tool outputs
    tool_outputs = message.additional_kwargs.get("tool_outputs", [])
    for block in tool_outputs:
        if isinstance(block, dict) and (key := block.get("type")) and key in buckets:
            buckets[key].append(block)
        else:
            unknown_blocks.append(block)

    # Re-assemble the content list in the canonical order
    new_content = []
    for key in content_order:
        new_content.extend(buckets[key])
    new_content.extend(unknown_blocks)

    new_additional_kwargs = dict(message.additional_kwargs)
    new_additional_kwargs.pop("reasoning", None)
    new_additional_kwargs.pop("refusal", None)
    new_additional_kwargs.pop("tool_outputs", None)

    if "id" in message.response_metadata:
        new_id = message.response_metadata["id"]
    else:
        new_id = message.id

    return message.model_copy(
        update={
            "content": new_content,
            "additional_kwargs": new_additional_kwargs,
            "id": new_id,
        },
        deep=False,
    )


def _convert_openai_format_to_data_block(
    block: dict,
) -> types.ContentBlock | dict[Any, Any]:
    """Convert OpenAI image/audio/file content block to respective v1 multimodal block.

    We expect that the incoming block is verified to be in OpenAI Chat Completions
    format.

    If parsing fails, passes block through unchanged.

    Mappings (Chat Completions to LangChain v1):
    - Image -> `ImageContentBlock`
    - Audio -> `AudioContentBlock`
    - File -> `FileContentBlock`

    """

    # Extract extra keys to put them in `extras`
    def _extract_extras(block_dict: dict, known_keys: set[str]) -> dict[str, Any]:
        """Extract unknown keys from block to preserve as extras."""
        return {k: v for k, v in block_dict.items() if k not in known_keys}

    # base64-style image block
    if (block["type"] == "image_url") and (
        parsed := _parse_data_uri(block["image_url"]["url"])
    ):
        known_keys = {"type", "image_url"}
        extras = _extract_extras(block, known_keys)

        # Also extract extras from nested image_url dict
        image_url_known_keys = {"url"}
        image_url_extras = _extract_extras(block["image_url"], image_url_known_keys)

        # Merge extras
        all_extras = {**extras}
        for key, value in image_url_extras.items():
            if key == "detail":  # Don't rename
                all_extras["detail"] = value
            else:
                all_extras[f"image_url_{key}"] = value

        return types.create_image_block(
            # Even though this is labeled as `url`, it can be base64-encoded
            base64=parsed["data"],
            mime_type=parsed["mime_type"],
            **all_extras,
        )

    # url-style image block
    if (block["type"] == "image_url") and isinstance(
        block["image_url"].get("url"), str
    ):
        known_keys = {"type", "image_url"}
        extras = _extract_extras(block, known_keys)

        image_url_known_keys = {"url"}
        image_url_extras = _extract_extras(block["image_url"], image_url_known_keys)

        all_extras = {**extras}
        for key, value in image_url_extras.items():
            if key == "detail":  # Don't rename
                all_extras["detail"] = value
            else:
                all_extras[f"image_url_{key}"] = value

        return types.create_image_block(
            url=block["image_url"]["url"],
            **all_extras,
        )

    # base64-style audio block
    # audio is only represented via raw data, no url or ID option
    if block["type"] == "input_audio":
        known_keys = {"type", "input_audio"}
        extras = _extract_extras(block, known_keys)

        # Also extract extras from nested audio dict
        audio_known_keys = {"data", "format"}
        audio_extras = _extract_extras(block["input_audio"], audio_known_keys)

        all_extras = {**extras}
        for key, value in audio_extras.items():
            all_extras[f"audio_{key}"] = value

        return types.create_audio_block(
            base64=block["input_audio"]["data"],
            mime_type=f"audio/{block['input_audio']['format']}",
            **all_extras,
        )

    # id-style file block
    if block.get("type") == "file" and "file_id" in block.get("file", {}):
        known_keys = {"type", "file"}
        extras = _extract_extras(block, known_keys)

        file_known_keys = {"file_id"}
        file_extras = _extract_extras(block["file"], file_known_keys)

        all_extras = {**extras}
        for key, value in file_extras.items():
            all_extras[f"file_{key}"] = value

        return types.create_file_block(
            file_id=block["file"]["file_id"],
            **all_extras,
        )

    # base64-style file block
    if (block["type"] == "file") and (
        parsed := _parse_data_uri(block["file"]["file_data"])
    ):
        known_keys = {"type", "file"}
        extras = _extract_extras(block, known_keys)

        file_known_keys = {"file_data", "filename"}
        file_extras = _extract_extras(block["file"], file_known_keys)

        all_extras = {**extras}
        for key, value in file_extras.items():
            all_extras[f"file_{key}"] = value

        filename = block["file"].get("filename")
        return types.create_file_block(
            base64=parsed["data"],
            mime_type="application/pdf",
            filename=filename,
            **all_extras,
        )

    # Escape hatch
    return block


# v1 / Responses
def _convert_annotation_to_v1(annotation: dict[str, Any]) -> types.Annotation:
    annotation_type = annotation.get("type")

    if annotation_type == "url_citation":
        known_fields = {
            "type",
            "url",
            "title",
            "cited_text",
            "start_index",
            "end_index",
        }
        url_citation = cast("types.Citation", {})
        for field in ("end_index", "start_index", "title"):
            if field in annotation:
                url_citation[field] = annotation[field]
        url_citation["type"] = "citation"
        url_citation["url"] = annotation["url"]
        for field, value in annotation.items():
            if field not in known_fields:
                if "extras" not in url_citation:
                    url_citation["extras"] = {}
                url_citation["extras"][field] = value
        return url_citation

    if annotation_type == "file_citation":
        known_fields = {
            "type",
            "title",
            "cited_text",
            "start_index",
            "end_index",
            "filename",
        }
        document_citation: types.Citation = {"type": "citation"}
        if "filename" in annotation:
            document_citation["title"] = annotation["filename"]
        for field, value in annotation.items():
            if field not in known_fields:
                if "extras" not in document_citation:
                    document_citation["extras"] = {}
                document_citation["extras"][field] = value

        return document_citation

    # TODO: standardise container_file_citation?
    non_standard_annotation: types.NonStandardAnnotation = {
        "type": "non_standard_annotation",
        "value": annotation,
    }
    return non_standard_annotation


def _explode_reasoning(block: dict[str, Any]) -> Iterable[types.ReasoningContentBlock]:
    if "summary" not in block:
        yield cast("types.ReasoningContentBlock", block)
        return

    known_fields = {"type", "reasoning", "id", "index"}
    unknown_fields = [
        field for field in block if field != "summary" and field not in known_fields
    ]
    if unknown_fields:
        block["extras"] = {}
    for field in unknown_fields:
        block["extras"][field] = block.pop(field)

    if not block["summary"]:
        # [{'id': 'rs_...', 'summary': [], 'type': 'reasoning', 'index': 0}]
        block = {k: v for k, v in block.items() if k != "summary"}
        if "index" in block:
            meaningful_idx = f"{block['index']}_0"
            block["index"] = f"lc_rs_{meaningful_idx.encode().hex()}"
        yield cast("types.ReasoningContentBlock", block)
        return

    # Common part for every exploded line, except 'summary'
    common = {k: v for k, v in block.items() if k in known_fields}

    # Optional keys that must appear only in the first exploded item
    first_only = block.pop("extras", None)

    for idx, part in enumerate(block["summary"]):
        new_block = dict(common)
        new_block["reasoning"] = part.get("text", "")
        if idx == 0 and first_only:
            new_block.update(first_only)
        if "index" in new_block:
            summary_index = part.get("index", 0)
            meaningful_idx = f"{new_block['index']}_{summary_index}"
            new_block["index"] = f"lc_rs_{meaningful_idx.encode().hex()}"

        yield cast("types.ReasoningContentBlock", new_block)


def _convert_to_v1_from_responses(message: AIMessage) -> list[types.ContentBlock]:
    """Convert a Responses message to v1 format."""

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        for raw_block in message.content:
            if not isinstance(raw_block, dict):
                continue
            block = raw_block.copy()
            block_type = block.get("type")

            if block_type == "text":
                if "text" not in block:
                    block["text"] = ""
                if "annotations" in block:
                    block["annotations"] = [
                        _convert_annotation_to_v1(a) for a in block["annotations"]
                    ]
                if "index" in block:
                    block["index"] = f"lc_txt_{block['index']}"
                yield cast("types.TextContentBlock", block)

            elif block_type == "reasoning":
                yield from _explode_reasoning(block)

            elif block_type == "image_generation_call" and (
                result := block.get("result")
            ):
                new_block = {"type": "image", "base64": result}
                if output_format := block.get("output_format"):
                    new_block["mime_type"] = f"image/{output_format}"
                if "id" in block:
                    new_block["id"] = block["id"]
                if "index" in block:
                    new_block["index"] = f"lc_img_{block['index']}"
                for extra_key in (
                    "status",
                    "background",
                    "output_format",
                    "quality",
                    "revised_prompt",
                    "size",
                ):
                    if extra_key in block:
                        if "extras" not in new_block:
                            new_block["extras"] = {}
                        new_block["extras"][extra_key] = block[extra_key]
                yield cast("types.ImageContentBlock", new_block)

            elif block_type == "function_call":
                tool_call_block: (
                    types.ToolCall | types.InvalidToolCall | types.ToolCallChunk | None
                ) = None
                call_id = block.get("call_id", "")

                from langchain_core.messages import AIMessageChunk  # noqa: PLC0415

                if (
                    isinstance(message, AIMessageChunk)
                    and len(message.tool_call_chunks) == 1
                    and message.chunk_position != "last"
                ):
                    tool_call_block = message.tool_call_chunks[0].copy()  # type: ignore[assignment]
                elif call_id:
                    for tool_call in message.tool_calls or []:
                        if tool_call.get("id") == call_id:
                            tool_call_block = {
                                "type": "tool_call",
                                "name": tool_call["name"],
                                "args": tool_call["args"],
                                "id": tool_call.get("id"),
                            }
                            break
                    else:
                        for invalid_tool_call in message.invalid_tool_calls or []:
                            if invalid_tool_call.get("id") == call_id:
                                tool_call_block = invalid_tool_call.copy()
                                break
                else:
                    pass
                if tool_call_block:
                    if "id" in block:
                        if "extras" not in tool_call_block:
                            tool_call_block["extras"] = {}
                        tool_call_block["extras"]["item_id"] = block["id"]
                    if "index" in block:
                        tool_call_block["index"] = f"lc_tc_{block['index']}"
                    yield tool_call_block

            elif block_type == "web_search_call":
                web_search_call = {
                    "type": "server_tool_call",
                    "name": "web_search",
                    "args": {},
                    "id": block["id"],
                }
                if "index" in block:
                    web_search_call["index"] = f"lc_wsc_{block['index']}"

                sources: dict[str, Any] | None = None
                if "action" in block and isinstance(block["action"], dict):
                    if "sources" in block["action"]:
                        sources = block["action"]["sources"]
                    web_search_call["args"] = {
                        k: v for k, v in block["action"].items() if k != "sources"
                    }
                for key in block:
                    if key not in ("type", "id", "action", "status", "index"):
                        web_search_call[key] = block[key]

                yield cast("types.ServerToolCall", web_search_call)

                # If .content already has web_search_result, don't add
                if not any(
                    isinstance(other_block, dict)
                    and other_block.get("type") == "web_search_result"
                    and other_block.get("id") == block["id"]
                    for other_block in message.content
                ):
                    web_search_result = {
                        "type": "server_tool_result",
                        "tool_call_id": block["id"],
                    }
                    if sources:
                        web_search_result["output"] = {"sources": sources}

                    status = block.get("status")
                    if status == "failed":
                        web_search_result["status"] = "error"
                    elif status == "completed":
                        web_search_result["status"] = "success"
                    elif status:
                        web_search_result["extras"] = {"status": status}
                    else:
                        pass
                    if "index" in block and isinstance(block["index"], int):
                        web_search_result["index"] = f"lc_wsr_{block['index'] + 1}"
                    yield cast("types.ServerToolResult", web_search_result)

            elif block_type == "file_search_call":
                file_search_call = {
                    "type": "server_tool_call",
                    "name": "file_search",
                    "id": block["id"],
                    "args": {"queries": block.get("queries", [])},
                }
                if "index" in block:
                    file_search_call["index"] = f"lc_fsc_{block['index']}"

                for key in block:
                    if key not in (
                        "type",
                        "id",
                        "queries",
                        "results",
                        "status",
                        "index",
                    ):
                        file_search_call[key] = block[key]

                yield cast("types.ServerToolCall", file_search_call)

                file_search_result = {
                    "type": "server_tool_result",
                    "tool_call_id": block["id"],
                }
                if file_search_output := block.get("results"):
                    file_search_result["output"] = file_search_output

                status = block.get("status")
                if status == "failed":
                    file_search_result["status"] = "error"
                elif status == "completed":
                    file_search_result["status"] = "success"
                elif status:
                    file_search_result["extras"] = {"status": status}
                else:
                    pass
                if "index" in block and isinstance(block["index"], int):
                    file_search_result["index"] = f"lc_fsr_{block['index'] + 1}"
                yield cast("types.ServerToolResult", file_search_result)

            elif block_type == "code_interpreter_call":
                code_interpreter_call = {
                    "type": "server_tool_call",
                    "name": "code_interpreter",
                    "id": block["id"],
                }
                if "code" in block:
                    code_interpreter_call["args"] = {"code": block["code"]}
                if "index" in block:
                    code_interpreter_call["index"] = f"lc_cic_{block['index']}"
                known_fields = {
                    "type",
                    "id",
                    "outputs",
                    "status",
                    "code",
                    "extras",
                    "index",
                }
                for key in block:
                    if key not in known_fields:
                        if "extras" not in code_interpreter_call:
                            code_interpreter_call["extras"] = {}
                        code_interpreter_call["extras"][key] = block[key]

                code_interpreter_result = {
                    "type": "server_tool_result",
                    "tool_call_id": block["id"],
                }
                if "outputs" in block:
                    code_interpreter_result["output"] = block["outputs"]

                status = block.get("status")
                if status == "failed":
                    code_interpreter_result["status"] = "error"
                elif status == "completed":
                    code_interpreter_result["status"] = "success"
                elif status:
                    code_interpreter_result["extras"] = {"status": status}
                else:
                    pass
                if "index" in block and isinstance(block["index"], int):
                    code_interpreter_result["index"] = f"lc_cir_{block['index'] + 1}"

                yield cast("types.ServerToolCall", code_interpreter_call)
                yield cast("types.ServerToolResult", code_interpreter_result)

            elif block_type == "mcp_call":
                mcp_call = {
                    "type": "server_tool_call",
                    "name": "remote_mcp",
                    "id": block["id"],
                }
                if (arguments := block.get("arguments")) and isinstance(arguments, str):
                    try:
                        mcp_call["args"] = json.loads(block["arguments"])
                    except json.JSONDecodeError:
                        mcp_call["extras"] = {"arguments": arguments}
                if "name" in block:
                    if "extras" not in mcp_call:
                        mcp_call["extras"] = {}
                    mcp_call["extras"]["tool_name"] = block["name"]
                if "server_label" in block:
                    if "extras" not in mcp_call:
                        mcp_call["extras"] = {}
                    mcp_call["extras"]["server_label"] = block["server_label"]
                if "index" in block:
                    mcp_call["index"] = f"lc_mcp_{block['index']}"
                known_fields = {
                    "type",
                    "id",
                    "arguments",
                    "name",
                    "server_label",
                    "output",
                    "error",
                    "extras",
                    "index",
                }
                for key in block:
                    if key not in known_fields:
                        if "extras" not in mcp_call:
                            mcp_call["extras"] = {}
                        mcp_call["extras"][key] = block[key]

                yield cast("types.ServerToolCall", mcp_call)

                mcp_result = {
                    "type": "server_tool_result",
                    "tool_call_id": block["id"],
                }
                if mcp_output := block.get("output"):
                    mcp_result["output"] = mcp_output

                error = block.get("error")
                if error:
                    if "extras" not in mcp_result:
                        mcp_result["extras"] = {}
                    mcp_result["extras"]["error"] = error
                    mcp_result["status"] = "error"
                else:
                    mcp_result["status"] = "success"

                if "index" in block and isinstance(block["index"], int):
                    mcp_result["index"] = f"lc_mcpr_{block['index'] + 1}"
                yield cast("types.ServerToolResult", mcp_result)

            elif block_type == "mcp_list_tools":
                mcp_list_tools_call = {
                    "type": "server_tool_call",
                    "name": "mcp_list_tools",
                    "args": {},
                    "id": block["id"],
                }
                if "server_label" in block:
                    mcp_list_tools_call["extras"] = {}
                    mcp_list_tools_call["extras"]["server_label"] = block[
                        "server_label"
                    ]
                if "index" in block:
                    mcp_list_tools_call["index"] = f"lc_mlt_{block['index']}"
                known_fields = {
                    "type",
                    "id",
                    "name",
                    "server_label",
                    "tools",
                    "error",
                    "extras",
                    "index",
                }
                for key in block:
                    if key not in known_fields:
                        if "extras" not in mcp_list_tools_call:
                            mcp_list_tools_call["extras"] = {}
                        mcp_list_tools_call["extras"][key] = block[key]

                yield cast("types.ServerToolCall", mcp_list_tools_call)

                mcp_list_tools_result = {
                    "type": "server_tool_result",
                    "tool_call_id": block["id"],
                }
                if mcp_output := block.get("tools"):
                    mcp_list_tools_result["output"] = mcp_output

                error = block.get("error")
                if error:
                    if "extras" not in mcp_list_tools_result:
                        mcp_list_tools_result["extras"] = {}
                    mcp_list_tools_result["extras"]["error"] = error
                    mcp_list_tools_result["status"] = "error"
                else:
                    mcp_list_tools_result["status"] = "success"

                if "index" in block and isinstance(block["index"], int):
                    mcp_list_tools_result["index"] = f"lc_mltr_{block['index'] + 1}"
                yield cast("types.ServerToolResult", mcp_list_tools_result)

            elif block_type in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)
            else:
                new_block = {"type": "non_standard", "value": block}
                if "index" in new_block["value"]:
                    new_block["index"] = f"lc_ns_{new_block['value'].pop('index')}"
                yield cast("types.NonStandardContentBlock", new_block)

    return list(_iter_blocks())


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with OpenAI content."""
    if isinstance(message.content, str):
        return _convert_to_v1_from_chat_completions(message)
    message = _convert_from_v03_ai_message(message)
    return _convert_to_v1_from_responses(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with OpenAI content."""
    if isinstance(message.content, str):
        return _convert_to_v1_from_chat_completions_chunk(message)
    message = _convert_from_v03_ai_message(message)  # type: ignore[assignment]
    return _convert_to_v1_from_responses(message)


def _register_openai_translator() -> None:
    """Register the OpenAI translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("openai", translate_content, translate_content_chunk)


_register_openai_translator()
