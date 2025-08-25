"""Derivations of standard content blocks from Anthropic content."""

import json
from collections.abc import Iterable
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types


def _populate_extras(
    standard_block: types.ContentBlock, block: dict[str, Any], known_fields: set[str]
) -> types.ContentBlock:
    """Mutate a block, populating extras."""
    if standard_block.get("type") == "non_standard":
        return standard_block

    for key, value in block.items():
        if key not in known_fields:
            if "extras" not in block:
                # Below type-ignores are because mypy thinks a non-standard block can
                # get here, although we exclude them above.
                standard_block["extras"] = {}  # type: ignore[typeddict-unknown-key]
            standard_block["extras"][key] = value  # type: ignore[typeddict-item]

    return standard_block


def _convert_to_v1_from_anthropic_input(
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
            block_type = block.get("type")

            if (
                block_type == "document"
                and "source" in block
                and "type" in block["source"]
            ):
                if block["source"]["type"] == "base64":
                    file_block: types.FileContentBlock = {
                        "type": "file",
                        "base64": block["source"]["data"],
                        "mime_type": block["source"]["media_type"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "url":
                    file_block = {
                        "type": "file",
                        "url": block["source"]["url"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "file":
                    file_block = {
                        "type": "file",
                        "id": block["source"]["file_id"],
                    }
                    _populate_extras(file_block, block, {"type", "source"})
                    yield file_block

                elif block["source"]["type"] == "text":
                    plain_text_block: types.PlainTextContentBlock = {
                        "type": "text-plain",
                        "text": block["source"]["data"],
                        "mime_type": block.get("media_type", "text/plain"),
                    }
                    _populate_extras(plain_text_block, block, {"type", "source"})
                    yield plain_text_block

                else:
                    yield {"type": "non_standard", "value": block}

            elif (
                block_type == "image"
                and "source" in block
                and "type" in block["source"]
            ):
                if block["source"]["type"] == "base64":
                    image_block: types.ImageContentBlock = {
                        "type": "image",
                        "base64": block["source"]["data"],
                        "mime_type": block["source"]["media_type"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif block["source"]["type"] == "url":
                    image_block = {
                        "type": "image",
                        "url": block["source"]["url"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                elif block["source"]["type"] == "file":
                    image_block = {
                        "type": "image",
                        "id": block["source"]["file_id"],
                    }
                    _populate_extras(image_block, block, {"type", "source"})
                    yield image_block

                else:
                    yield {"type": "non_standard", "value": block}

            elif block_type in types.KNOWN_BLOCK_TYPES:
                yield cast("types.ContentBlock", block)

            else:
                yield {"type": "non_standard", "value": block}

    return list(_iter_blocks())


def _convert_citation_to_v1(citation: dict[str, Any]) -> types.Annotation:
    citation_type = citation.get("type")

    if citation_type == "web_search_result_location":
        url_citation: types.Citation = {
            "type": "citation",
            "cited_text": citation["cited_text"],
            "url": citation["url"],
        }
        if title := citation.get("title"):
            url_citation["title"] = title
        known_fields = {"type", "cited_text", "url", "title", "index", "extras"}
        for key, value in citation.items():
            if key not in known_fields:
                if "extras" not in url_citation:
                    url_citation["extras"] = {}
                url_citation["extras"][key] = value

        return url_citation

    if citation_type in (
        "char_location",
        "content_block_location",
        "page_location",
        "search_result_location",
    ):
        document_citation: types.Citation = {
            "type": "citation",
            "cited_text": citation["cited_text"],
        }
        if "document_title" in citation:
            document_citation["title"] = citation["document_title"]
        elif title := citation.get("title"):
            document_citation["title"] = title
        else:
            pass
        known_fields = {
            "type",
            "cited_text",
            "document_title",
            "title",
            "index",
            "extras",
        }
        for key, value in citation.items():
            if key not in known_fields:
                if "extras" not in document_citation:
                    document_citation["extras"] = {}
                document_citation["extras"][key] = value

        return document_citation

    return {
        "type": "non_standard_annotation",
        "value": citation,
    }


def _convert_to_v1_from_anthropic(message: AIMessage) -> list[types.ContentBlock]:
    """Convert Anthropic message content to v1 format."""
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

            elif block_type == "thinking":
                reasoning_block: types.ReasoningContentBlock = {
                    "type": "reasoning",
                    "reasoning": block.get("thinking", ""),
                }
                if "index" in block:
                    reasoning_block["index"] = block["index"]
                known_fields = {"type", "thinking", "index", "extras"}
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
                ):
                    tool_call_chunk: types.ToolCallChunk = (
                        message.tool_call_chunks[0].copy()  # type: ignore[assignment]
                    )
                    if "type" not in tool_call_chunk:
                        tool_call_chunk["type"] = "tool_call_chunk"
                    yield tool_call_chunk
                elif (
                    not isinstance(message, AIMessageChunk)
                    and len(message.tool_calls) == 1
                ):
                    tool_call_block: types.ToolCall = {
                        "type": "tool_call",
                        "name": message.tool_calls[0]["name"],
                        "args": message.tool_calls[0]["args"],
                        "id": message.tool_calls[0].get("id"),
                    }
                    if "index" in block:
                        tool_call_block["index"] = block["index"]
                    yield tool_call_block
                else:
                    tool_call_block = {
                        "type": "tool_call",
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                        "id": block.get("id", ""),
                    }
                    yield tool_call_block

            elif (
                block_type == "input_json_delta"
                and isinstance(message, AIMessageChunk)
                and len(message.tool_call_chunks) == 1
            ):
                tool_call_chunk = (
                    message.tool_call_chunks[0].copy()  # type: ignore[assignment]
                )
                if "type" not in tool_call_chunk:
                    tool_call_chunk["type"] = "tool_call_chunk"
                yield tool_call_chunk

            elif block_type == "server_tool_use":
                if block.get("name") == "web_search":
                    web_search_call: types.WebSearchCall = {"type": "web_search_call"}

                    if query := block.get("input", {}).get("query"):
                        web_search_call["query"] = query

                    elif block.get("input") == {} and "partial_json" in block:
                        try:
                            input_ = json.loads(block["partial_json"])
                            if isinstance(input_, dict) and "query" in input_:
                                web_search_call["query"] = input_["query"]
                        except json.JSONDecodeError:
                            pass

                    if "id" in block:
                        web_search_call["id"] = block["id"]
                    if "index" in block:
                        web_search_call["index"] = block["index"]
                    known_fields = {"type", "name", "input", "id", "index"}
                    for key, value in block.items():
                        if key not in known_fields:
                            if "extras" not in web_search_call:
                                web_search_call["extras"] = {}
                            web_search_call["extras"][key] = value
                    yield web_search_call

                elif block.get("name") == "code_execution":
                    code_interpreter_call: types.CodeInterpreterCall = {
                        "type": "code_interpreter_call"
                    }

                    if code := block.get("input", {}).get("code"):
                        code_interpreter_call["code"] = code

                    elif block.get("input") == {} and "partial_json" in block:
                        try:
                            input_ = json.loads(block["partial_json"])
                            if isinstance(input_, dict) and "code" in input_:
                                code_interpreter_call["code"] = input_["code"]
                        except json.JSONDecodeError:
                            pass

                    if "id" in block:
                        code_interpreter_call["id"] = block["id"]
                    if "index" in block:
                        code_interpreter_call["index"] = block["index"]
                    known_fields = {"type", "name", "input", "id", "index"}
                    for key, value in block.items():
                        if key not in known_fields:
                            if "extras" not in code_interpreter_call:
                                code_interpreter_call["extras"] = {}
                            code_interpreter_call["extras"][key] = value
                    yield code_interpreter_call

                else:
                    new_block: types.NonStandardContentBlock = {
                        "type": "non_standard",
                        "value": block,
                    }
                    if "index" in new_block["value"]:
                        new_block["index"] = new_block["value"].pop("index")
                    yield new_block

            elif block_type == "web_search_tool_result":
                web_search_result: types.WebSearchResult = {"type": "web_search_result"}
                if "tool_use_id" in block:
                    web_search_result["id"] = block["tool_use_id"]
                if "index" in block:
                    web_search_result["index"] = block["index"]

                if web_search_result_content := block.get("content", []):
                    if "extras" not in web_search_result:
                        web_search_result["extras"] = {}
                    urls = []
                    extra_content = []
                    for result_content in web_search_result_content:
                        if isinstance(result_content, dict):
                            if "url" in result_content:
                                urls.append(result_content["url"])
                            extra_content.append(result_content)
                    web_search_result["extras"]["content"] = extra_content
                    if urls:
                        web_search_result["urls"] = urls
                yield web_search_result

            elif block_type == "code_execution_tool_result":
                code_interpreter_result: types.CodeInterpreterResult = {
                    "type": "code_interpreter_result",
                    "output": [],
                }
                if "tool_use_id" in block:
                    code_interpreter_result["id"] = block["tool_use_id"]
                if "index" in block:
                    code_interpreter_result["index"] = block["index"]

                code_interpreter_output: types.CodeInterpreterOutput = {
                    "type": "code_interpreter_output"
                }

                code_execution_content = block.get("content", {})
                if code_execution_content.get("type") == "code_execution_result":
                    if "return_code" in code_execution_content:
                        code_interpreter_output["return_code"] = code_execution_content[
                            "return_code"
                        ]
                    if "stdout" in code_execution_content:
                        code_interpreter_output["stdout"] = code_execution_content[
                            "stdout"
                        ]
                    if stderr := code_execution_content.get("stderr"):
                        code_interpreter_output["stderr"] = stderr
                    if (
                        output := code_interpreter_output.get("content")
                    ) and isinstance(output, list):
                        if "extras" not in code_interpreter_result:
                            code_interpreter_result["extras"] = {}
                        code_interpreter_result["extras"]["content"] = output
                        for output_block in output:
                            if "file_id" in output_block:
                                if "file_ids" not in code_interpreter_output:
                                    code_interpreter_output["file_ids"] = []
                                code_interpreter_output["file_ids"].append(
                                    output_block["file_id"]
                                )
                    code_interpreter_result["output"].append(code_interpreter_output)

                elif (
                    code_execution_content.get("type")
                    == "code_execution_tool_result_error"
                ):
                    if "extras" not in code_interpreter_result:
                        code_interpreter_result["extras"] = {}
                    code_interpreter_result["extras"]["error_code"] = (
                        code_execution_content.get("error_code")
                    )

                yield code_interpreter_result

            else:
                new_block = {"type": "non_standard", "value": block}
                if "index" in new_block["value"]:
                    new_block["index"] = new_block["value"].pop("index")
                yield new_block

    return list(_iter_blocks())


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with OpenAI content."""
    return _convert_to_v1_from_anthropic(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with OpenAI content."""
    return _convert_to_v1_from_anthropic(message)


def _register_anthropic_translator() -> None:
    """Register the Anthropic translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import register_translator

    register_translator("anthropic", translate_content, translate_content_chunk)


_register_anthropic_translator()
