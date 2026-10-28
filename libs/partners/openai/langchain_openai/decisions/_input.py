"""Convert `OpenAIDecisions` state to the Decisions API `input` field."""

from __future__ import annotations

import json
import math
import re
import uuid
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

    Strings are sent as is, and sequences of `HumanMessage` objects as user messages.
    Anything else, such as conversations with system, AI, or tool messages, is
    serialized to JSON text in a single user message, with base64 images from any
    message kept in place as image parts.

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
    if _supports_native_json(state):
        return [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
    return [_to_json_user_message(state)]


def _supports_native_json(state: object) -> bool:
    if not isinstance(state, dict) and not (
        isinstance(state, list) and all(isinstance(item, dict) for item in state)
    ):
        return False
    pending = [(state, 0)]
    count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > 10_000:
            return False
        if isinstance(value, (dict, list)):
            if depth >= 32:
                return False
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    return False
                children = list(value.values())
            else:
                children = value
            pending.extend((child, depth + 1) for child in children)
        elif not (
            value is None
            or isinstance(value, (str, int, bool))
            or (isinstance(value, float) and math.isfinite(value))
        ):
            return False
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return len(encoded.encode("utf-8")) <= 256 * 1024


def _to_user_message(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}
    return {
        "role": "user",
        "content": [_to_message_part(block) for block in message.content],
    }


def _to_message_part(block: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(block, str):
        return {"type": "input_text", "text": block}
    if block.get("type") == "image":
        converted = convert_to_openai_messages(HumanMessage(content=[block]))[
            "content"
        ][0]
        part = _to_input_part(converted)
        if "prompt_cache_breakpoint" in block:
            part["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
        return part
    return _to_input_part(block)


def _to_input_part(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") in {"text", "input_text"}:
        part = {"type": "input_text", "text": block["text"]}
    elif block.get("type") in {"image_url", "input_image"}:
        image = block["image_url"]
        url = image["url"] if isinstance(image, dict) else image
        if not url.startswith(("data:", "https://", "http://")):
            msg = "OpenAI Decisions images require data URLs or hosted HTTP(S) URLs."
            raise ValueError(msg)
        part = {"type": "input_image", "image_url": url}
        detail = image.get("detail") if isinstance(image, dict) else block.get("detail")
        if detail is not None:
            part["detail"] = detail
    else:
        msg = (
            "Unsupported content block type for the OpenAI Decisions API: "
            f"{block.get('type')!r}."
        )
        raise ValueError(msg)
    if "prompt_cache_breakpoint" in block:
        part["prompt_cache_breakpoint"] = block["prompt_cache_breakpoint"]
    return part


def _to_json_user_message(state: State) -> dict[str, Any]:
    """Serialize state the API cannot accept natively to a single user message.

    Covers conversations with non-user messages and JSON objects or arrays. Messages
    at any depth are converted to role/content objects. Base64 images in any message
    are sent as `input_image` parts at the position where they appear in the text, so
    the format stays the same whether or not a conversation contains images.
    """
    images: list[dict[str, Any]] = []
    token = f"lc-image-{uuid.uuid4().hex}"
    text = json.dumps(_to_json_value(state, images, token), ensure_ascii=False)
    content: list[dict[str, Any]] = []
    position = 0
    for match in re.finditer(rf"{token}-(\d+)", text):
        content.append({"type": "input_text", "text": text[position : match.start()]})
        content.append(images[int(match[1])])
        position = match.end()
    content.append({"type": "input_text", "text": text[position:]})
    return {"role": "user", "content": content}


def _to_json_value(value: object, images: list[dict[str, Any]], token: str) -> Any:
    if isinstance(value, BaseMessage):
        return convert_to_openai_messages(_with_image_tokens(value, images, token))
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            msg = "OpenAI Decisions state object keys must be strings."
            raise TypeError(msg)
        return {key: _to_json_value(item, images, token) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_value(item, images, token) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    msg = f"Unsupported OpenAI Decisions state value: {type(value).__name__}."
    raise TypeError(msg)


def _with_image_tokens(
    message: BaseMessage, images: list[dict[str, Any]], token: str
) -> BaseMessage:
    """Replace base64 image blocks with tokens marking where each image belongs."""
    if isinstance(message.content, str):
        return message
    content: list[str | dict[str, Any]] = []
    for block in message.content:
        url = _image_part(block) if isinstance(block, dict) else None
        if url is None:
            content.append(block)
        else:
            content.append({"type": "text", "text": f"{token}-{len(images)}"})
            images.append(url)
    return message.model_copy(update={"content": content})


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an image block, or return `None` for other blocks."""
    if block.get("type") not in {"image", "image_url", "input_image"}:
        return None
    return _to_message_part(block)
