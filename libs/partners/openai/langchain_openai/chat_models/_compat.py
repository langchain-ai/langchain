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

import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal, Union, cast

from langchain_core.messages import AIMessage, is_data_content_block
from langchain_core.messages import content as types

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


# v1 / Chat Completions
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
                    elif key in current.get("extras", {}):
                        collapsed[key] = current["extras"][key]
                    else:
                        pass

                for key in ("outputs", "status"):
                    if key in nxt:
                        collapsed[key] = nxt[key]
                    elif key in nxt.get("extras", {}):
                        collapsed[key] = nxt["extras"][key]
                    else:
                        pass
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
                elif extra_key in block.get("extras", {}):
                    new_block[extra_key] = block["extras"][extra_key]
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
