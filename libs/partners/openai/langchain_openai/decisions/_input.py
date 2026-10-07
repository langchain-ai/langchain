"""Convert `OpenAIDecisions` state to the Decisions API `input` field."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)

if TYPE_CHECKING:
    from langchain_openai.decisions.types import State


def to_decision_input(state: State) -> str | list[dict[str, Any]]:
    """Convert state to a string or a list of user messages.

    Args:
        state: Text, messages, or structured state to evaluate.

    Returns:
        A value suitable for the Decisions API `input` field.

    Raises:
        ValueError: If a message contains content the API does not accept.
        TypeError: If the state cannot be represented as JSON.
    """
    if isinstance(state, str):
        return state
    messages = [state] if isinstance(state, BaseMessage) else state
    if (
        isinstance(messages, Sequence)
        and messages
        and all(isinstance(message, HumanMessage) for message in messages)
    ):
        return [_to_user_message(message) for message in messages]  # type: ignore[arg-type]
    return _serialize_as_json_text(state)


def _to_user_message(message: HumanMessage) -> dict[str, Any]:
    content = convert_to_openai_messages(message)["content"]
    if isinstance(content, str):
        return {"role": "user", "content": content}
    return {"role": "user", "content": [_to_input_part(block) for block in content]}


def _to_input_part(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "text":
        return {"type": "input_text", "text": block["text"]}
    if block.get("type") == "image_url":
        url = block["image_url"]["url"]
        if not url.startswith("data:"):
            msg = (
                "The OpenAI Decisions API only accepts images as base64 data URLs. "
                "Hosted image URLs are not supported."
            )
            raise ValueError(msg)
        return {"type": "input_image", "image_url": url}
    msg = (
        f"Unsupported content block type for the OpenAI Decisions API: "
        f"{block.get('type')!r}. Only text and base64 images are supported."
    )
    raise ValueError(msg)


def _serialize_as_json_text(state: State) -> str:
    """Serialize state the API cannot accept natively to JSON text.

    Covers conversations with non-user messages and JSON objects or arrays. Messages
    at any depth are converted to role/content objects.
    """
    return json.dumps(_to_json_value(state), ensure_ascii=False)


def _to_json_value(value: object) -> Any:
    if isinstance(value, BaseMessage):
        return convert_to_openai_messages(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            msg = "OpenAI Decisions state object keys must be strings."
            raise TypeError(msg)
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    msg = f"Unsupported OpenAI Decisions state value: {type(value).__name__}."
    raise TypeError(msg)
