"""Serialization helpers for client-executed Responses tools."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)

_NATIVE_CALL_TYPES = ("apply_patch_call", "shell_call", "local_shell_call")
_NATIVE_OUTPUT_TYPES = (
    "apply_patch_call_output",
    "shell_call_output",
    "local_shell_call_output",
)


def _native_tool_call(block: dict[str, Any]) -> ToolCall | None:
    """Normalize only complete, direct, client-executed calls."""
    block_type = block.get("type")
    if block_type not in _NATIVE_CALL_TYPES or block.get("status") != "completed":
        return None
    if (block.get("caller") or {}).get("type", "direct") != "direct":
        return None
    if (
        block_type == "shell_call"
        and (block.get("environment") or {}).get("type", "local") != "local"
    ):
        return None
    key = "operation" if block_type == "apply_patch_call" else "action"
    return ToolCall(
        name=block_type.removesuffix("_call"),
        args={key: block[key]},
        id=block["call_id"],
        type="tool_call",
    )


def _normalize_native_tool_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """Encode rejected or answered native calls without fabricating execution."""
    calls: dict[str, str] = {}
    normalized: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            blocks = list(message.content) if isinstance(message.content, list) else []
            blocks.extend(message.additional_kwargs.get("tool_outputs", []))
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "non_standard":
                    block = block.get("value", {})
                if isinstance(block, dict) and _native_tool_call(block) is not None:
                    calls[block["call_id"]] = block["type"]
        if not isinstance(message, ToolMessage) or message.tool_call_id not in calls:
            normalized.append(message)
            continue
        call_type = calls[message.tool_call_id]
        if structured_output := _native_tool_output(message):
            if structured_output["type"] != f"{call_type}_output":
                msg = "Native output type must match its originating tool call."
                raise ValueError(msg)
            normalized.append(message)
            continue
        text = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content)
        )
        output: dict[str, Any]
        if call_type == "apply_patch_call":
            output = {
                "type": "apply_patch_call_output",
                "status": "failed",
                "output": text,
            }
        elif call_type == "local_shell_call":
            output = {
                "type": "local_shell_call_output",
                "output": json.dumps({"error": text}),
            }
        else:
            output = {"type": "shell_call_output", "output": [], "status": "incomplete"}
        normalized.append(message.model_copy(update={"content": [output]}))
        if call_type == "shell_call":
            normalized.append(
                HumanMessage(
                    content=(
                        f"Shell call {message.tool_call_id} was not executed. "
                        f"Client response: {text}"
                    )
                )
            )
    return normalized


def _native_tool_output(message: ToolMessage) -> dict[str, Any] | None:
    """Serialize a structured native result without losing its call identity."""
    if not isinstance(message.content, list):
        return None
    outputs = []
    for block in message.content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "non_standard":
            block = block.get("value", {})
        if isinstance(block, dict) and block.get("type") in _NATIVE_OUTPUT_TYPES:
            outputs.append(block)
    if not outputs:
        return None
    if len(outputs) != 1 or len(message.content) != 1:
        msg = "A native ToolMessage must contain exactly one structured output."
        raise ValueError(msg)
    output = {key: value for key, value in outputs[0].items() if key != "index"}
    identity_key = "id" if output["type"] == "local_shell_call_output" else "call_id"
    if output.get(identity_key, message.tool_call_id) != message.tool_call_id:
        msg = "Native output identity must match ToolMessage.tool_call_id."
        raise ValueError(msg)
    output[identity_key] = message.tool_call_id
    if output["type"] == "apply_patch_call_output" and output.get("status") not in (
        "completed",
        "failed",
    ):
        msg = "apply_patch_call_output requires a completed or failed status."
        raise ValueError(msg)
    return output
