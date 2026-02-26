"""Derivations of standard content blocks from Groq content."""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.base import _extract_reasoning_from_additional_kwargs


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


def _parse_code_json(s: str) -> dict:
    """Extract Python code from Groq built-in tool content.

    Extracts the value of the 'code' field from a string of the form:
    {"code": some_arbitrary_text_with_unescaped_quotes}

    As Groq may not escape quotes in the executed tools, e.g.:
    ```
    '{"code": "import math; print("The square root of 101 is: "); print(math.sqrt(101))"}'
    ```
    """  # noqa: E501
    m = re.fullmatch(r'\s*\{\s*"code"\s*:\s*"(.*)"\s*\}\s*', s, flags=re.DOTALL)
    if not m:
        msg = (
            "Could not extract Python code from Groq tool arguments. "
            "Expected a JSON object with a 'code' field."
        )
        raise ValueError(msg)
    return {"code": m.group(1)}


def _convert_to_v1_from_groq(message: AIMessage) -> list[types.ContentBlock]:
    """Convert groq message content to v1 format."""
    content_blocks: list[types.ContentBlock] = []

    if reasoning_block := _extract_reasoning_from_additional_kwargs(message):
        content_blocks.append(reasoning_block)

    if executed_tools := message.additional_kwargs.get("executed_tools"):
        for idx, executed_tool in enumerate(executed_tools):
            args: dict[str, Any] | None = None
            if arguments := executed_tool.get("arguments"):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    if executed_tool.get("type") == "python":
                        try:
                            args = _parse_code_json(arguments)
                        except ValueError:
                            continue
                    elif (
                        executed_tool.get("type") == "function"
                        and executed_tool.get("name") == "python"
                    ):
                        # GPT-OSS
                        args = {"code": arguments}
                    else:
                        continue
            if isinstance(args, dict):
                name = ""
                if executed_tool.get("type") == "search":
                    name = "web_search"
                elif executed_tool.get("type") == "python" or (
                    executed_tool.get("type") == "function"
                    and executed_tool.get("name") == "python"
                ):
                    name = "code_interpreter"
                server_tool_call: types.ServerToolCall = {
                    "type": "server_tool_call",
                    "name": name,
                    "id": str(idx),
                    "args": args,
                }
                content_blocks.append(server_tool_call)
            if tool_output := executed_tool.get("output"):
                tool_result: types.ServerToolResult = {
                    "type": "server_tool_result",
                    "tool_call_id": str(idx),
                    "output": tool_output,
                    "status": "success",
                }
                known_fields = {"type", "arguments", "index", "output"}
                _populate_extras(tool_result, executed_tool, known_fields)
                content_blocks.append(tool_result)

    if isinstance(message.content, str) and message.content:
        content_blocks.append({"type": "text", "text": message.content})

    for tool_call in message.tool_calls:
        content_blocks.append(  # noqa: PERF401
            {
                "type": "tool_call",
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call.get("id"),
            }
        )

    return content_blocks


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with groq content.

    Args:
        message: The message to translate.

    Returns:
        The derived content blocks.
    """
    return _convert_to_v1_from_groq(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with groq content.

    Args:
        message: The message chunk to translate.

    Returns:
        The derived content blocks.
    """
    return _convert_to_v1_from_groq(message)


def _register_groq_translator() -> None:
    """Register the groq translator with the central registry.

    Run automatically when the module is imported.
    """
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("groq", translate_content, translate_content_chunk)


_register_groq_translator()
