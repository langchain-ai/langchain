"""Derivations of standard content blocks from OpenAI content."""

from collections.abc import Iterable
from typing import Any, Optional, Union, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


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
        content_blocks.append(tool_call)

    return content_blocks


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
                tool_call_block: Optional[
                    Union[types.ToolCall, types.InvalidToolCall, types.ToolCallChunk]
                ] = None
                call_id = block.get("call_id", "")
                if (
                    isinstance(message, AIMessageChunk)
                    and len(message.tool_call_chunks) == 1
                ):
                    tool_call_block = message.tool_call_chunks[0].copy()  # type: ignore[assignment]
                elif call_id:
                    for tool_call in message.tool_calls or []:
                        if tool_call.get("id") == call_id:
                            tool_call_block = tool_call.copy()
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
                web_search_call = {"type": "web_search_call", "id": block["id"]}
                if "index" in block:
                    web_search_call["index"] = f"lc_wsc_{block['index']}"
                if (
                    "action" in block
                    and isinstance(block["action"], dict)
                    and block["action"].get("type") == "search"
                    and "query" in block["action"]
                ):
                    web_search_call["query"] = block["action"]["query"]
                for key in block:
                    if key not in ("type", "id", "index"):
                        web_search_call[key] = block[key]

                yield cast("types.WebSearchCall", web_search_call)

                # If .content already has web_search_result, don't add
                if not any(
                    isinstance(other_block, dict)
                    and other_block.get("type") == "web_search_result"
                    and other_block.get("id") == block["id"]
                    for other_block in message.content
                ):
                    web_search_result = {"type": "web_search_result", "id": block["id"]}
                    if "index" in block and isinstance(block["index"], int):
                        web_search_result["index"] = f"lc_wsr_{block['index'] + 1}"
                    yield cast("types.WebSearchResult", web_search_result)

            elif block_type == "code_interpreter_call":
                code_interpreter_call = {
                    "type": "code_interpreter_call",
                    "id": block["id"],
                }
                if "code" in block:
                    code_interpreter_call["code"] = block["code"]
                if "index" in block:
                    code_interpreter_call["index"] = f"lc_cic_{block['index']}"
                known_fields = {"type", "id", "language", "code", "extras", "index"}
                for key in block:
                    if key not in known_fields:
                        if "extras" not in code_interpreter_call:
                            code_interpreter_call["extras"] = {}
                        code_interpreter_call["extras"][key] = block[key]

                code_interpreter_result = {
                    "type": "code_interpreter_result",
                    "id": block["id"],
                }
                if "outputs" in block:
                    code_interpreter_result["outputs"] = block["outputs"]
                    for output in block["outputs"]:
                        if (
                            isinstance(output, dict)
                            and (output_type := output.get("type"))
                            and output_type == "logs"
                        ):
                            if "output" not in code_interpreter_result:
                                code_interpreter_result["output"] = []
                            code_interpreter_result["output"].append(
                                {
                                    "type": "code_interpreter_output",
                                    "stdout": output.get("logs", ""),
                                }
                            )

                if "status" in block:
                    code_interpreter_result["status"] = block["status"]
                if "index" in block and isinstance(block["index"], int):
                    code_interpreter_result["index"] = f"lc_cir_{block['index'] + 1}"

                yield cast("types.CodeInterpreterCall", code_interpreter_call)
                yield cast("types.CodeInterpreterResult", code_interpreter_result)

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
    return _convert_to_v1_from_responses(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with OpenAI content."""
    if isinstance(message.content, str):
        return _convert_to_v1_from_chat_completions_chunk(message)
    return _convert_to_v1_from_responses(message)
