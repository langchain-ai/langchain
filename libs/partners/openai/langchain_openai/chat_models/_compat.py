"""
This module converts between AIMessage output formats, which are governed by the
``output_version`` attribute on ChatOpenAI. Supported values are ``"v0"`` and
``"responses/v1"``.

``"v0"`` corresponds to the format as of ChatOpenAI v0.3. For the Responses API, it
stores reasoning and tool outputs in AIMessage.additional_kwargs:

.. code-block:: python

    AIMessage(
        content=[
            {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
        ],
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            },
            "tool_outputs": [
                {
                    "type": "web_search_call",
                    "id": "websearch_123",
                    "status": "completed",
                }
            ],
            "refusal": "I cannot assist with that.",
        },
        response_metadata={"id": "resp_123"},
        id="msg_123",
    )

``"responses/v1"`` is only applicable to the Responses API. It retains information
about response item sequencing and accommodates multiple reasoning items by
representing these items in the content sequence:

.. code-block:: python

    AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
                "id": "rs_123",
            },
            {
                "type": "text",
                "text": "Hello, world!",
                "annotations": [{"type": "foo"}],
                "id": "msg_123",
            },
            {"type": "refusal", "refusal": "I cannot assist with that."},
            {"type": "web_search_call", "id": "websearch_123", "status": "completed"},
        ],
        response_metadata={"id": "resp_123"},
        id="resp_123",
    )

There are other, small improvements as well-- e.g., we store message IDs on text
content blocks, rather than on the AIMessage.id, which now stores the response ID.

For backwards compatibility, this module provides functions to convert between the
formats. The functions are used internally by ChatOpenAI.
"""  # noqa: E501

import copy
import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal, Optional, Union, cast

from langchain_core.messages import AIMessage, is_data_content_block
from langchain_core.messages import content_blocks as types
from langchain_core.v1.messages import AIMessage as AIMessageV1

_FUNCTION_CALL_IDS_MAP_KEY = "__openai_function_call_ids__"


# v0.3 / Responses
def _convert_to_v03_ai_message(
    message: AIMessage, has_reasoning: bool = False
) -> AIMessage:
    """Mutate an AIMessage to the old-style v0.3 format."""
    if isinstance(message.content, list):
        new_content: list[Union[dict, str]] = []
        for block in message.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    # Store a reasoning item in additional_kwargs (overwriting as in
                    # v0.3)
                    _ = block.pop("index", None)
                    if has_reasoning:
                        _ = block.pop("id", None)
                        _ = block.pop("type", None)
                    message.additional_kwargs["reasoning"] = block
                elif block.get("type") in (
                    "web_search_call",
                    "file_search_call",
                    "computer_call",
                    "code_interpreter_call",
                    "mcp_call",
                    "mcp_list_tools",
                    "mcp_approval_request",
                    "image_generation_call",
                ):
                    # Store built-in tool calls in additional_kwargs
                    if "tool_outputs" not in message.additional_kwargs:
                        message.additional_kwargs["tool_outputs"] = []
                    message.additional_kwargs["tool_outputs"].append(block)
                elif block.get("type") == "function_call":
                    # Store function call item IDs in additional_kwargs, otherwise
                    # discard function call items.
                    if _FUNCTION_CALL_IDS_MAP_KEY not in message.additional_kwargs:
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {}
                    if (call_id := block.get("call_id")) and (
                        function_call_id := block.get("id")
                    ):
                        message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][
                            call_id
                        ] = function_call_id
                elif (block.get("type") == "refusal") and (
                    refusal := block.get("refusal")
                ):
                    # Store a refusal item in additional_kwargs (overwriting as in
                    # v0.3)
                    message.additional_kwargs["refusal"] = refusal
                elif block.get("type") == "text":
                    # Store a message item ID on AIMessage.id
                    if "id" in block:
                        message.id = block["id"]
                    new_content.append({k: v for k, v in block.items() if k != "id"})
                elif (
                    set(block.keys()) == {"id", "index"}
                    and isinstance(block["id"], str)
                    and block["id"].startswith("msg_")
                ):
                    # Drop message IDs in streaming case
                    new_content.append({"index": block["index"]})
                else:
                    new_content.append(block)
            else:
                new_content.append(block)
        message.content = new_content
        if isinstance(message.id, str) and message.id.startswith("resp_"):
            message.id = None
    else:
        pass

    return message


def _convert_from_v03_ai_message(message: AIMessage) -> AIMessage:
    """Convert an old-style v0.3 AIMessage into the new content-block format."""
    # Only update ChatOpenAI v0.3 AIMessages
    # TODO: structure provenance into AIMessage
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


# v1 / Chat Completions
def _convert_from_v1_to_chat_completions(message: AIMessageV1) -> AIMessageV1:
    """Convert a v1 message to the Chat Completions format."""
    new_content: list[types.ContentBlock] = []
    for block in message.content:
        if block["type"] == "text":
            # Strip annotations
            new_content.append({"type": "text", "text": block["text"]})
        elif block["type"] in ("reasoning", "tool_call"):
            pass
        else:
            new_content.append(block)
    new_message = copy.copy(message)
    new_message.content = new_content

    return new_message


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
        url_citation = cast(types.Citation, {})
        for field in ("end_index", "start_index", "title"):
            if field in annotation:
                url_citation[field] = annotation[field]
        url_citation["type"] = "citation"
        url_citation["url"] = annotation["url"]
        for field in annotation:
            if field not in known_fields:
                if "extras" not in url_citation:
                    url_citation["extras"] = {}
                url_citation["extras"][field] = annotation[field]
        return url_citation

    elif annotation_type == "file_citation":
        known_fields = {"type", "title", "cited_text", "start_index", "end_index"}
        document_citation: types.Citation = {"type": "citation"}
        if "filename" in annotation:
            document_citation["title"] = annotation.pop("filename")
        for field in annotation:
            if field not in known_fields:
                if "extras" not in document_citation:
                    document_citation["extras"] = {}
                document_citation["extras"][field] = annotation[field]

        return document_citation

    # TODO: standardise container_file_citation?
    else:
        non_standard_annotation: types.NonStandardAnnotation = {
            "type": "non_standard_annotation",
            "value": annotation,
        }
        return non_standard_annotation


def _explode_reasoning(block: dict[str, Any]) -> Iterable[types.ReasoningContentBlock]:
    if "summary" not in block:
        yield cast(types.ReasoningContentBlock, block)
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
        _ = block.pop("summary", None)
        yield cast(types.ReasoningContentBlock, block)
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
        yield cast(types.ReasoningContentBlock, new_block)


def _convert_to_v1_from_responses(
    content: list[dict[str, Any]],
    tool_calls: Optional[list[types.ToolCall]] = None,
    invalid_tool_calls: Optional[list[types.InvalidToolCall]] = None,
) -> list[types.ContentBlock]:
    """Mutate a Responses message to v1 format."""

    def _iter_blocks() -> Iterable[types.ContentBlock]:
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                if "annotations" in block:
                    block["annotations"] = [
                        _convert_annotation_to_v1(a) for a in block["annotations"]
                    ]
                yield cast(types.TextContentBlock, block)

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
                    new_block["index"] = block["index"]
                for extra_key in (
                    "status",
                    "background",
                    "output_format",
                    "quality",
                    "revised_prompt",
                    "size",
                ):
                    if extra_key in block:
                        new_block[extra_key] = block[extra_key]
                yield cast(types.ImageContentBlock, new_block)

            elif block_type == "function_call":
                tool_call_block: Optional[types.ContentBlock] = None
                call_id = block.get("call_id", "")
                if call_id:
                    for tool_call in tool_calls or []:
                        if tool_call.get("id") == call_id:
                            tool_call_block = cast(types.ToolCall, tool_call.copy())
                            break
                    else:
                        for invalid_tool_call in invalid_tool_calls or []:
                            if invalid_tool_call.get("id") == call_id:
                                tool_call_block = cast(
                                    types.InvalidToolCall, invalid_tool_call.copy()
                                )
                                break
                if tool_call_block:
                    if "id" in block:
                        if "extras" not in tool_call_block:
                            tool_call_block["extras"] = {}
                        tool_call_block["extras"]["item_id"] = block["id"]  # type: ignore[typeddict-item]
                    if "index" in block:
                        tool_call_block["index"] = block["index"]
                    yield tool_call_block

            elif block_type == "web_search_call":
                web_search_call = {"type": "web_search_call", "id": block["id"]}
                if "index" in block:
                    web_search_call["index"] = block["index"]
                if (
                    "action" in block
                    and isinstance(block["action"], dict)
                    and block["action"].get("type") == "search"
                    and "query" in block["action"]
                ):
                    web_search_call["query"] = block["action"]["query"]
                for key in block:
                    if key not in ("type", "id"):
                        web_search_call[key] = block[key]

                web_search_result = {"type": "web_search_result", "id": block["id"]}
                if "index" in block:
                    web_search_result["index"] = block["index"] + 1
                yield cast(types.WebSearchCall, web_search_call)
                yield cast(types.WebSearchResult, web_search_result)

            elif block_type == "code_interpreter_call":
                code_interpreter_call = {
                    "type": "code_interpreter_call",
                    "id": block["id"],
                }
                if "code" in block:
                    code_interpreter_call["code"] = block["code"]
                if "container_id" in block:
                    code_interpreter_call["container_id"] = block["container_id"]
                if "index" in block:
                    code_interpreter_call["index"] = block["index"]

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
                if "index" in block:
                    code_interpreter_result["index"] = block["index"] + 1

                yield cast(types.CodeInterpreterCall, code_interpreter_call)
                yield cast(types.CodeInterpreterResult, code_interpreter_result)

            else:
                new_block = {"type": "non_standard", "value": block}
                if "index" in new_block["value"]:
                    new_block["index"] = new_block["value"].pop("index")
                yield cast(types.NonStandardContentBlock, new_block)

    return list(_iter_blocks())


def _convert_annotation_from_v1(annotation: types.Annotation) -> dict[str, Any]:
    if annotation["type"] == "citation":
        new_ann: dict[str, Any] = {}
        for field in ("end_index", "start_index"):
            if field in annotation:
                new_ann[field] = annotation[field]

        if "url" in annotation:
            # URL citation
            if "title" in annotation:
                new_ann["title"] = annotation["title"]
            new_ann["type"] = "url_citation"
            new_ann["url"] = annotation["url"]
        else:
            # Document citation
            new_ann["type"] = "file_citation"
            if "title" in annotation:
                new_ann["filename"] = annotation["title"]

        if extra_fields := annotation.get("extras"):
            for field, value in extra_fields.items():
                new_ann[field] = value

        return new_ann

    elif annotation["type"] == "non_standard_annotation":
        return annotation["value"]

    else:
        return dict(annotation)


def _implode_reasoning_blocks(blocks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    i = 0
    n = len(blocks)

    while i < n:
        block = blocks[i]

        # Skip non-reasoning blocks or blocks already in Responses format
        if block.get("type") != "reasoning" or "summary" in block:
            yield dict(block)
            i += 1
            continue
        elif "reasoning" not in block and "summary" not in block:
            # {"type": "reasoning", "id": "rs_..."}
            oai_format = {**block, "summary": []}
            if "extras" in oai_format:
                oai_format.update(oai_format.pop("extras"))
            oai_format["type"] = oai_format.pop("type", "reasoning")
            if "encrypted_content" in oai_format:
                oai_format["encrypted_content"] = oai_format.pop("encrypted_content")
            yield oai_format
            i += 1
            continue
        else:
            pass

        summary: list[dict[str, str]] = [
            {"type": "summary_text", "text": block.get("reasoning", "")}
        ]
        # 'common' is every field except the exploded 'reasoning'
        common = {k: v for k, v in block.items() if k != "reasoning"}
        if "extras" in common:
            common.update(common.pop("extras"))

        i += 1
        while i < n:
            next_ = blocks[i]
            if next_.get("type") == "reasoning" and "reasoning" in next_:
                summary.append(
                    {"type": "summary_text", "text": next_.get("reasoning", "")}
                )
                i += 1
            else:
                break

        merged = dict(common)
        merged["summary"] = summary
        merged["type"] = merged.pop("type", "reasoning")
        yield merged


def _consolidate_calls(
    items: Iterable[dict[str, Any]],
    call_name: Literal["web_search_call", "code_interpreter_call"],
    result_name: Literal["web_search_result", "code_interpreter_result"],
) -> Iterator[dict[str, Any]]:
    """
    Generator that walks through *items* and, whenever it meets the pair

        {"type": "web_search_call",    "id": X, ...}
        {"type": "web_search_result",  "id": X}

    merges them into

        {"id": X,
         "action": …,
         "status": …,
         "type": "web_search_call"}

    keeping every other element untouched.
    """
    items = iter(items)  # make sure we have a true iterator
    for current in items:
        # Only a call can start a pair worth collapsing
        if current.get("type") != call_name:
            yield current
            continue

        try:
            nxt = next(items)  # look-ahead one element
        except StopIteration:  # no “result” – just yield the call back
            yield current
            break

        # If this really is the matching “result” – collapse
        if nxt.get("type") == result_name and nxt.get("id") == current.get("id"):
            if call_name == "web_search_call":
                collapsed = {"id": current["id"]}
                if "action" in current:
                    collapsed["action"] = current["action"]
                collapsed["status"] = current["status"]
                collapsed["type"] = "web_search_call"

            if call_name == "code_interpreter_call":
                collapsed = {"id": current["id"]}
                for key in ("code", "container_id"):
                    if key in current:
                        collapsed[key] = current[key]

                for key in ("outputs", "status"):
                    if key in nxt:
                        collapsed[key] = nxt[key]
                collapsed["type"] = "code_interpreter_call"

            yield collapsed

        else:
            # Not a matching pair – emit both, in original order
            yield current
            yield nxt


def _convert_from_v1_to_responses(
    content: list[types.ContentBlock], tool_calls: list[types.ToolCall]
) -> list[dict[str, Any]]:
    new_content: list = []
    for block in content:
        if block["type"] == "text" and "annotations" in block:
            # Need a copy because we’re changing the annotations list
            new_block = dict(block)
            new_block["annotations"] = [
                _convert_annotation_from_v1(a) for a in block["annotations"]
            ]
            new_content.append(new_block)
        elif block["type"] == "tool_call":
            new_block = {"type": "function_call", "call_id": block["id"]}
            if "extras" in block and "item_id" in block["extras"]:
                new_block["id"] = block["extras"]["item_id"]
            if "name" in block:
                new_block["name"] = block["name"]
            if "extras" in block and "arguments" in block["extras"]:
                new_block["arguments"] = block["extras"]["arguments"]
            if any(key not in block for key in ("name", "arguments")):
                matching_tool_calls = [
                    call for call in tool_calls if call["id"] == block["id"]
                ]
                if matching_tool_calls:
                    tool_call = matching_tool_calls[0]
                    if "name" not in block:
                        new_block["name"] = tool_call["name"]
                    if "arguments" not in block:
                        new_block["arguments"] = json.dumps(tool_call["args"])
            new_content.append(new_block)
        elif (
            is_data_content_block(cast(dict, block))
            and block["type"] == "image"
            and "base64" in block
            and isinstance(block.get("id"), str)
            and block["id"].startswith("ig_")
        ):
            new_block = {"type": "image_generation_call", "result": block["base64"]}
            for extra_key in ("id", "status"):
                if extra_key in block:
                    new_block[extra_key] = block[extra_key]  # type: ignore[typeddict-item]
            new_content.append(new_block)
        elif block["type"] == "non_standard" and "value" in block:
            new_content.append(block["value"])
        else:
            new_content.append(block)

    new_content = list(_implode_reasoning_blocks(new_content))
    new_content = list(
        _consolidate_calls(new_content, "web_search_call", "web_search_result")
    )
    new_content = list(
        _consolidate_calls(
            new_content, "code_interpreter_call", "code_interpreter_result"
        )
    )

    return new_content
