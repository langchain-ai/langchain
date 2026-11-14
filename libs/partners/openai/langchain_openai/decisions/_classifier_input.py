"""Convert `OpenAIClassifier` requests to the OpenAI Decisions API format."""

from __future__ import annotations

import json
import re
import uuid
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.decisions import Box, Choice, Noul
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
    is_data_content_block,
)

from langchain_openai.decisions._input import _to_json_user_message, _to_json_value

if TYPE_CHECKING:
    from langchain_core.decisions import Question, State
    from pydantic import JsonValue


def _to_classifier_input(state: State) -> str | list[dict[str, Any]]:
    """Convert state to a string or a list of Decisions API input items.

    Strings are sent as-is. A message or sequence of messages is sent as native input
    items: human messages as user messages, system messages as system messages, AI
    messages as assistant messages plus one `function_call` item per tool call, and
    tool messages as `function_call_output` items. Any other state is sent as a
    single user message: an `input_json` part when it is a JSON object or array of
    objects, or JSON text otherwise. `input_json` cannot carry media, so media blocks
    in such state, including inside nested messages, are dropped with a warning.

    Args:
        state: Text, messages, or structured state to classify.

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
        and all(isinstance(message, BaseMessage) for message in messages)
    ):
        return [item for message in messages for item in _to_items(message)]  # type: ignore[arg-type]
    if state is None or isinstance(state, (int, float, bool)):
        msg = (
            "Classifier state must be a string, object, array, BaseMessage, or "
            "sequence of BaseMessage objects."
        )
        raise TypeError(msg)
    dropped: list[dict[str, Any]] = []
    state = _without_media(state, dropped)
    if dropped:
        warnings.warn(
            f"Dropped {len(dropped)} media block(s) from classifier state. The OpenAI "
            "Decisions API accepts media only in messages passed as the root state.",
            UserWarning,
            stacklevel=2,
        )
    data = _to_json_value(state, [], f"lc-image-{uuid.uuid4().hex}")
    if _is_json_object_data(data):
        return [{"role": "user", "content": [{"type": "input_json", "data": data}]}]
    return [_to_json_user_message(state)]  # type: ignore[arg-type]


def _without_media(value: Any, dropped: list[dict[str, Any]]) -> Any:
    """Return `value` without media blocks, collecting the dropped blocks."""
    if isinstance(value, BaseMessage):
        if isinstance(value.content, str):
            return value
        return value.model_copy(
            update={"content": _without_media(value.content, dropped)}
        )
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if isinstance(item, dict) and _is_media_block(item):
                dropped.append(item)
            else:
                result[key] = _without_media(item, dropped)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        kept = []
        for item in value:
            if isinstance(item, dict) and _is_media_block(item):
                dropped.append(item)
            else:
                kept.append(_without_media(item, dropped))
        return kept
    return value


def _to_classifier_questions(questions: dict[str, Question]) -> list[dict[str, Any]]:
    """Convert validated questions to the Decisions API `questions` field.

    A `Noul` is sent as a `predicate`. Predicates have no field for outcome
    descriptions, so `Noul` criteria are appended to the instructions. Image blocks
    in instructions are sent as `input_image` parts.

    Args:
        questions: Validated questions keyed by answer ID.

    Returns:
        Questions in the Decisions API format, in the order supplied.

    Raises:
        ValueError: If a question is a `Box`, a `Choice` has fewer than two labels,
            instructions contain media other than images, or criteria contain media.
    """
    return [_question_to_api(name, question) for name, question in questions.items()]


def _question_to_api(name: str, question: Question) -> dict[str, Any]:
    if isinstance(question, Box):
        msg = (
            f"Question {name!r}: the OpenAI Decisions API does not support Box "
            "questions."
        )
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "name": name,
        "instructions": _to_instructions(question.instructions),
    }
    if isinstance(question, Noul):
        if criteria_text := _noul_criteria_text(question):
            payload["instructions"] = _append_text(
                payload["instructions"], criteria_text
            )
        return {"type": "predicate", **payload}
    if isinstance(question, Choice):
        if len(question.criteria) < 2:
            msg = (
                f"Question {name!r}: the OpenAI Decisions API requires at least two "
                "Choice labels."
            )
            raise ValueError(msg)
        payload["choices"] = [
            _with_description({"value": label}, description)
            for label, description in question.criteria.items()
        ]
        return {"type": "choice", **payload}
    payload["levels"] = [{"label": _to_text(level)} for level in question.criteria]
    return {"type": "score", **payload}


def _noul_criteria_text(question: Noul) -> str:
    """Return `Noul` criteria as text to append to the instructions, if any."""
    if question.criteria is None:
        return ""
    outcomes = [("true", question.criteria.true), ("false", question.criteria.false)]
    return "\n".join(
        f"Answer {outcome} if: {_to_text(description)}"
        for outcome, description in outcomes
        if description is not None
    )


def _append_text(
    instructions: str | list[dict[str, Any]], text: str
) -> str | list[dict[str, Any]]:
    if isinstance(instructions, str):
        return f"{instructions}\n\n{text}"
    return [*instructions, {"type": "input_text", "text": text}]


def _with_description(item: dict[str, Any], description: JsonValue) -> dict[str, Any]:
    if description is not None:
        item["description"] = _to_text(description)
    return item


def _to_instructions(content: JsonValue) -> str | list[dict[str, Any]]:
    """Send text as-is, JSON objects as `input_json` parts, and other JSON as text.

    Image blocks are sent as `input_image` parts. An image block nested in other JSON
    is sent at the position where it appears in the JSON text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and _is_media_block(content):
        return [{"type": "input_image", "image_url": _image_url(content)}]
    images: list[str] = []
    token = f"lc-image-{uuid.uuid4().hex}"
    data = _replace_images(content, images, token)
    if not images:
        if _is_json_object_data(content):
            return [{"type": "input_json", "data": content}]
        return _to_text(content)
    text = json.dumps(data, ensure_ascii=False)
    parts: list[dict[str, Any]] = []
    position = 0
    for match in re.finditer(rf"{token}-(\d+)", text):
        parts.append({"type": "input_text", "text": text[position : match.start()]})
        parts.append({"type": "input_image", "image_url": images[int(match[1])]})
        position = match.end()
    parts.append({"type": "input_text", "text": text[position:]})
    return parts


def _replace_images(value: JsonValue, images: list[str], token: str) -> JsonValue:
    """Replace image blocks with tokens marking where each image belongs."""
    if isinstance(value, dict):
        if _is_media_block(value):
            images.append(_image_url(value))
            return f"{token}-{len(images) - 1}"
        return {
            key: _replace_images(item, images, token) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_images(item, images, token) for item in value]
    return value


def _is_media_block(block: dict[str, Any]) -> bool:
    return block.get("type") == "image_url" or is_data_content_block(block)


def _image_url(block: dict[str, Any]) -> str:
    """Return the data or `http(s)` URL of an image block."""
    if block.get("type") in {"image", "image_url"}:
        part = convert_to_openai_messages(HumanMessage(content=[block]))["content"][0]
        if isinstance(part, dict) and part.get("type") == "image_url":
            return part["image_url"]["url"]
    msg = (
        f"Unsupported content block type in question instructions for the OpenAI "
        f"Decisions API: {block.get('type')!r}. Only text and images are supported."
    )
    raise ValueError(msg)


def _is_json_object_data(value: object) -> bool:
    """Return whether `value` fits `input_json`, which takes an object or objects."""
    if isinstance(value, Mapping):
        return True
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, Mapping) for item in value)
    )


def _to_text(content: JsonValue) -> str:
    """Return text content as-is and other JSON content as JSON text."""
    if isinstance(content, str):
        return content
    if _contains_media(content):
        msg = (
            "The OpenAI Decisions API accepts only text in question criteria, so "
            "media blocks are not supported there."
        )
        raise ValueError(msg)
    return json.dumps(content, ensure_ascii=False)


def _contains_media(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return _is_media_block(value) or any(map(_contains_media, value.values()))
    if isinstance(value, list):
        return any(map(_contains_media, value))
    return False


def _to_items(message: BaseMessage) -> list[dict[str, Any]]:
    if isinstance(message, HumanMessage):
        return [{"role": "user", "content": _user_content(message)}]
    if isinstance(message, SystemMessage):
        return [{"role": "system", "content": _text_only(message)}]
    if isinstance(message, AIMessage):
        return _assistant_items(message)
    if isinstance(message, ToolMessage):
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id,
                "output": _user_content(message),
            }
        ]
    msg = (
        f"Unsupported message type for the OpenAI Decisions API: "
        f"{type(message).__name__}."
    )
    raise ValueError(msg)


def _assistant_items(message: AIMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if text := message.text:
        items.append({"role": "assistant", "content": text})
    for tool_call in message.tool_calls:
        if not tool_call.get("id"):
            msg = "The OpenAI Decisions API requires an ID for every tool call."
            raise ValueError(msg)
        items.append(
            {
                "type": "function_call",
                "call_id": tool_call["id"],
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
            }
        )
    return items


def _user_content(message: BaseMessage) -> str | list[dict[str, Any]]:
    content = convert_to_openai_messages(message)["content"]
    if isinstance(content, str):
        return content
    return [_to_input_part(block) for block in content]


def _text_only(message: BaseMessage) -> str:
    content = convert_to_openai_messages(message)["content"]
    if isinstance(content, str):
        return content
    if any(block.get("type") != "text" for block in content):
        msg = "The OpenAI Decisions API accepts only text in system messages."
        raise ValueError(msg)
    return "".join(block["text"] for block in content)


def _to_input_part(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "text":
        return {"type": "input_text", "text": block["text"]}
    if block.get("type") == "image_url":
        image = block["image_url"]
        part = {"type": "input_image", "image_url": image["url"]}
        if detail := image.get("detail"):
            part["detail"] = detail
        return part
    msg = (
        f"Unsupported content block type for the OpenAI Decisions API: "
        f"{block.get('type')!r}. Only text and images are supported."
    )
    raise ValueError(msg)
