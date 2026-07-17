"""Helpers for OpenRouter's beta Responses API.

OpenRouter's Responses API is OpenAI-compatible in shape but **stateless**:
`store` and `previous_response_id` are rejected by the server. Sticky routing /
prompt-cache affinity should use `session_id` instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_STATEFUL_UNSUPPORTED_MSG = (
    "OpenRouter's Responses API is stateless and does not support `{param}`. "
    "Requests that set `store: true` or a non-null `previous_response_id` are "
    "rejected by OpenRouter with a 400 error. "
    "Send the full conversation history on each request, and use `session_id` "
    "for sticky routing / prompt-cache affinity. "
    "See https://openrouter.ai/docs/api/reference/responses/overview"
)

# Params that must never be forwarded to OpenRouter Responses.
_RESPONSES_DROP_PARAMS = frozenset(
    {
        "stop",
        "n",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "seed",  # not in Responses request surface
        "route",  # chat-completions-oriented OpenRouter param
    }
)

_VALID_MESSAGE_STATUSES = frozenset({"completed", "incomplete", "in_progress"})


def _assert_stateless_responses_supported(
    params: dict[str, Any],
    *,
    use_previous_response_id: bool = False,
) -> None:
    """Raise if unsupported stateful Responses parameters are present.

    Args:
        params: Merged invocation / model kwargs.
        use_previous_response_id: Constructor flag mirroring OpenAI's API.

    Raises:
        ValueError: If a stateful Responses parameter is set.
    """
    if use_previous_response_id:
        msg = _STATEFUL_UNSUPPORTED_MSG.format(param="use_previous_response_id")
        raise ValueError(msg)
    if params.get("previous_response_id") is not None:
        msg = _STATEFUL_UNSUPPORTED_MSG.format(param="previous_response_id")
        raise ValueError(msg)
    if params.get("store") is True:
        msg = _STATEFUL_UNSUPPORTED_MSG.format(param="store")
        raise ValueError(msg)


def _block_to_input_part(block: Any) -> dict[str, Any] | None:  # noqa: PLR0911
    """Convert a single content block to a Responses input part."""
    if isinstance(block, str):
        return {"type": "input_text", "text": block} if block else None
    if not isinstance(block, dict):
        return None

    block_type = block.get("type")
    if block_type in ("text", "input_text"):
        text = block.get("text", "")
        return {"type": "input_text", "text": text} if text else None
    if block_type == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        return {"type": "input_image", "image_url": url} if url else None
    if block_type in ("input_image", "input_file"):
        return block
    if block_type != "file":
        return None

    file_obj = block.get("file") or {}
    file_part: dict[str, Any] = {"type": "input_file"}
    if file_data := file_obj.get("file_data"):
        file_part["file_data"] = file_data
    if filename := file_obj.get("filename"):
        file_part["filename"] = filename
    return file_part


def _content_to_input_parts(content: Any) -> list[dict[str, Any]]:
    """Convert message content into Responses `input_*` content parts."""
    # Lazy import avoids a circular dependency with chat_models.
    from langchain_openrouter.chat_models import (  # noqa: PLC0415
        _format_message_content,
    )

    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}] if content else []

    return [
        part
        for block in _format_message_content(content) or []
        if (part := _block_to_input_part(block)) is not None
    ]


def _assistant_text_parts(content: Any) -> list[dict[str, Any]]:
    """Convert assistant content into Responses `output_text` parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return (
            [{"type": "output_text", "text": content, "annotations": []}]
            if content
            else []
        )

    parts: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append({"type": "output_text", "text": block, "annotations": []})
            continue
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "output_text"):
            text = block.get("text")
            if text is None:
                continue
            parts.append(
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": block.get("annotations") or [],
                }
            )
        elif block_type == "refusal":
            parts.append({"type": "refusal", "refusal": block["refusal"]})
    return parts


def _resolve_assistant_message_id(message: AIMessage) -> str:
    """Resolve a `msg_*` id for multi-turn Responses replay."""
    msg_id = message.id
    if isinstance(msg_id, str) and msg_id.startswith("msg_"):
        return msg_id
    if isinstance(message.response_metadata, dict):
        stored = message.response_metadata.get("message_id")
        if isinstance(stored, str) and stored:
            return stored
    return f"msg_{uuid4().hex}"


def _resolve_assistant_status(message: AIMessage) -> str:
    """Resolve assistant item status for multi-turn Responses replay."""
    status = "completed"
    if isinstance(message.response_metadata, dict):
        status = (
            message.response_metadata.get("message_status")
            or message.response_metadata.get("status")
            or status
        )
    if status not in _VALID_MESSAGE_STATUSES:
        return "completed"
    return status


def _append_ai_message_items(
    message: AIMessage, input_items: list[dict[str, Any]]
) -> None:
    """Append Responses input items derived from an `AIMessage`."""
    text_parts = _assistant_text_parts(message.content)
    msg_id = _resolve_assistant_message_id(message)
    status = _resolve_assistant_status(message)

    if text_parts:
        input_items.append(
            {
                "type": "message",
                "role": "assistant",
                "id": msg_id,
                "status": status,
                "content": text_parts,
            }
        )

    for tool_call in message.tool_calls:
        args = tool_call.get("args")
        if isinstance(args, str):
            arguments = args
        else:
            arguments = json.dumps(args or {}, ensure_ascii=False)
        input_items.append(
            {
                "type": "function_call",
                "name": tool_call["name"],
                "arguments": arguments,
                "call_id": tool_call.get("id") or f"call_{uuid4().hex}",
            }
        )

    reasoning_details = message.additional_kwargs.get("reasoning_details")
    if isinstance(reasoning_details, list):
        input_items.extend(
            detail
            for detail in reasoning_details
            if isinstance(detail, dict) and detail.get("type") == "reasoning"
        )


def _construct_responses_api_input(
    messages: Sequence[BaseMessage],
) -> list[dict[str, Any]]:
    """Convert LangChain messages into OpenRouter Responses `input` items.

    Args:
        messages: Conversation history.

    Returns:
        A list of Responses API input items.
    """
    input_items: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ToolMessage):
            output = message.content
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": output,
                }
            )
            continue

        if isinstance(message, AIMessage):
            _append_ai_message_items(message, input_items)
            continue

        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ChatMessage):
            role = message.role
        else:
            msg = f"Got unknown type {message}"
            raise TypeError(msg)

        parts = _content_to_input_parts(message.content)
        # OpenRouter accepts empty content for some roles; still send a message.
        input_items.append(
            {
                "type": "message",
                "role": role,
                "content": parts or [{"type": "input_text", "text": ""}],
            }
        )

    return input_items


def _flatten_responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Chat Completions-style tools to Responses shape."""
    if tool.get("type") == "function" and "function" in tool:
        extra = {k: v for k, v in tool.items() if k not in ("type", "function")}
        return {"type": "function", **tool["function"], **extra}
    return tool


def _flatten_responses_tool_choice(tool_choice: Any) -> Any:
    """Convert Chat Completions-style tool_choice to Responses shape."""
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and "function" in tool_choice
    ):
        return {"type": "function", **tool_choice["function"]}
    return tool_choice


def _apply_response_format(payload: dict[str, Any], schema: Any) -> None:
    """Map Chat Completions `response_format` onto Responses `text.format`."""
    strict = payload.pop("strict", None)
    if schema == {"type": "json_object"}:
        payload["text"] = {"format": {"type": "json_object"}}
        return
    if isinstance(schema, dict) and schema.get("type") == "json_schema":
        payload["text"] = {"format": {"type": "json_schema", **schema["json_schema"]}}
        return
    if isinstance(schema, dict):
        name = schema.get("title") or "response"
        text_format: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if strict is not None:
            text_format["strict"] = strict
        payload["text"] = {"format": text_format}


def _construct_responses_api_payload(
    messages: Sequence[BaseMessage],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build an OpenRouter Responses API request payload.

    Args:
        messages: Conversation history.
        params: Merged model / invocation parameters.

    Returns:
        Keyword arguments suitable for `client.beta.responses.send`.
    """
    payload = dict(params)

    for legacy_token_param in ("max_tokens", "max_completion_tokens"):
        if legacy_token_param in payload and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = payload[legacy_token_param]

    for key in _RESPONSES_DROP_PARAMS:
        payload.pop(key, None)

    # Stateful knobs must never be forwarded.
    payload.pop("previous_response_id", None)
    payload.pop("store", None)
    payload.pop("use_previous_response_id", None)

    payload["input"] = _construct_responses_api_input(messages)

    if tools := payload.pop("tools", None):
        payload["tools"] = [_flatten_responses_tool(tool) for tool in tools]

    if "tool_choice" in payload:
        payload["tool_choice"] = _flatten_responses_tool_choice(payload["tool_choice"])

    if schema := payload.pop("response_format", None):
        _apply_response_format(payload, schema)

    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize SDK models / dicts to a plain dict."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=False)
    msg = f"Expected dict or pydantic model, got {type(value)!r}"
    raise TypeError(msg)


def _extract_output_text(response: dict[str, Any]) -> str:
    """Join all `output_text` parts from a Responses result."""
    texts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        texts.extend(
            part.get("text") or ""
            for part in item.get("content") or []
            if isinstance(part, dict) and part.get("type") == "output_text"
        )
    return "".join(texts)


def _raise_response_error(error: Any, *, streaming: bool = False) -> None:
    """Raise a `ValueError` for an OpenRouter Responses error payload."""
    if isinstance(error, dict):
        err_msg = error.get("message", str(error))
        code = error.get("code", "unknown")
    else:
        err_msg = str(error)
        code = "unknown"
    prefix = (
        "OpenRouter API returned an error during streaming"
        if streaming
        else "OpenRouter API returned an error"
    )
    msg = f"{prefix}: {err_msg} (code: {code})"
    raise ValueError(msg)


def _parse_function_call_item(
    item: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    invalid_tool_calls: list[dict[str, Any]],
) -> None:
    """Parse a Responses `function_call` output item into tool call lists."""
    raw_args = item.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        tool_calls.append(
            {
                "type": "tool_call",
                "name": item.get("name") or "",
                "args": args,
                "id": item.get("call_id"),
            }
        )
    except json.JSONDecodeError as e:
        invalid_tool_calls.append(
            {
                "type": "invalid_tool_call",
                "name": item.get("name") or "",
                "args": raw_args,
                "id": item.get("call_id"),
                "error": str(e),
            }
        )


def _consume_output_items(
    output_items: list[Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    str | None,
    str | None,
]:
    """Parse Responses `output` into tool calls, kwargs, and message ids."""
    tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[dict[str, Any]] = []
    additional_kwargs: dict[str, Any] = {}
    message_id: str | None = None
    message_status: str | None = None

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            message_id = item.get("id") or message_id
            message_status = item.get("status") or message_status
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "refusal":
                    additional_kwargs["refusal"] = part.get("refusal")
        elif item_type == "function_call":
            _parse_function_call_item(item, tool_calls, invalid_tool_calls)
        elif item_type == "reasoning":
            details = additional_kwargs.setdefault("reasoning_details", [])
            details.append(item)

    return tool_calls, invalid_tool_calls, additional_kwargs, message_id, message_status


def _build_response_metadata(
    response_dict: dict[str, Any],
    *,
    model_name: str,
    message_id: str | None,
    message_status: str | None,
) -> dict[str, Any]:
    """Build `AIMessage.response_metadata` from a Responses result."""
    response_metadata: dict[str, Any] = {
        "model_provider": "openrouter",
        "model_name": response_dict.get("model") or model_name,
    }
    if response_id := response_dict.get("id"):
        response_metadata["id"] = response_id
    if created_at := response_dict.get("created_at"):
        response_metadata["created_at"] = created_at
    if status := response_dict.get("status"):
        response_metadata["status"] = status
    if message_id:
        response_metadata["message_id"] = message_id
    if message_status:
        response_metadata["message_status"] = message_status
    if object_ := response_dict.get("object"):
        response_metadata["object"] = object_
    return response_metadata


def _construct_lc_result_from_responses_api(
    response: Any,
    *,
    model_name: str,
) -> ChatResult:
    """Convert an OpenRouter Responses result into a `ChatResult`.

    Args:
        response: SDK `OpenResponsesResult` or equivalent dict.
        model_name: Fallback model name when the response omits `model`.

    Returns:
        A LangChain `ChatResult`.
    """
    response_dict = _as_dict(response)

    if error := response_dict.get("error"):
        _raise_response_error(error)

    (
        tool_calls,
        invalid_tool_calls,
        additional_kwargs,
        message_id,
        message_status,
    ) = _consume_output_items(response_dict.get("output") or [])

    content = _extract_output_text(response_dict)
    # Prefer empty string over None for tool-only responses (matches chat path).
    if tool_calls and not content:
        content = ""

    response_metadata = _build_response_metadata(
        response_dict,
        model_name=model_name,
        message_id=message_id,
        message_status=message_status,
    )

    usage_metadata = None
    if usage := response_dict.get("usage"):
        from langchain_openrouter.chat_models import (  # noqa: PLC0415
            _create_usage_metadata,
        )

        usage_metadata = _create_usage_metadata(usage)
        if isinstance(usage, dict):
            if "cost" in usage:
                response_metadata["cost"] = usage["cost"]
            if "cost_details" in usage:
                response_metadata["cost_details"] = usage["cost_details"]

    # Use msg_* as AIMessage.id for multi-turn Responses replay; fall back to
    # the response id when no message item is present.
    ai_id = message_id or response_dict.get("id")

    message = AIMessage(
        content=content,
        id=ai_id,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,  # type: ignore[arg-type]
        invalid_tool_calls=invalid_tool_calls,  # type: ignore[arg-type]
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )

    llm_output: dict[str, Any] = {
        "model_name": response_metadata["model_name"],
    }
    if response_id := response_dict.get("id"):
        llm_output["id"] = response_id
    if object_ := response_dict.get("object"):
        llm_output["object"] = object_

    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output=llm_output,
    )


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Normalize a streaming event to a dict."""
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump(by_alias=True, exclude_none=False)
    # Fallback for attribute-style SDK events.
    data: dict[str, Any] = {}
    for key in (
        "type",
        "delta",
        "output_index",
        "content_index",
        "item_id",
        "response",
        "item",
        "name",
        "arguments",
        "call_id",
    ):
        if hasattr(chunk, key):
            value = getattr(chunk, key)
            if hasattr(value, "model_dump"):
                value = value.model_dump(by_alias=True, exclude_none=False)
            data[key] = value
    return data


def _text_delta_chunk(delta: Any) -> ChatGenerationChunk | None:
    """Build a chunk from a text delta event payload."""
    if isinstance(delta, dict):
        delta = delta.get("text") or delta.get("delta") or ""
    if not delta:
        return None
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=delta,
            response_metadata={"model_provider": "openrouter"},
        )
    )


def _function_call_added_chunk(
    item: dict[str, Any], output_index: Any
) -> ChatGenerationChunk:
    """Build a chunk for a newly added function_call output item."""
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            tool_call_chunks=[
                tool_call_chunk(
                    name=item.get("name"),
                    args=item.get("arguments") or "",
                    id=item.get("call_id"),
                    index=output_index,
                )
            ],
            response_metadata={"model_provider": "openrouter"},
        )
    )


def _completed_response_chunk(
    response: dict[str, Any], *, model_name: str
) -> ChatGenerationChunk | None:
    """Build a terminal chunk from a completed/incomplete Responses event."""
    result = _construct_lc_result_from_responses_api(response, model_name=model_name)
    message = result.generations[0].message
    if not isinstance(message, AIMessage):
        return None
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content="",
            usage_metadata=message.usage_metadata,
            response_metadata={
                k: v
                for k, v in message.response_metadata.items()
                if k != "model_provider"
            }
            | {"model_provider": "openrouter"},
            id=message.id,
        ),
        generation_info={"finish_reason": response.get("status")},
    )


def _convert_responses_chunk_to_generation_chunk(  # noqa: C901, PLR0911
    chunk: Any,
    *,
    model_name: str,
) -> ChatGenerationChunk | None:
    """Convert a Responses SSE event into a `ChatGenerationChunk`.

    Args:
        chunk: Streaming event from `client.beta.responses.send(stream=True)`.
        model_name: Fallback model name for metadata.

    Returns:
        A generation chunk, or `None` for events that should be skipped.
    """
    event = _chunk_to_dict(chunk)
    event_type = event.get("type")

    if event_type in (
        "response.output_text.delta",
        "response.content_part.delta",
    ):
        return _text_delta_chunk(event.get("delta") or "")

    if event_type == "response.function_call_arguments.delta":
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    tool_call_chunk(
                        args=event.get("delta") or "",
                        index=event.get("output_index"),
                    )
                ],
                response_metadata={"model_provider": "openrouter"},
            )
        )

    if event_type == "response.output_item.added":
        item = event.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "function_call":
            return _function_call_added_chunk(item, event.get("output_index"))
        return None

    if event_type == "response.created":
        response = event.get("response") or {}
        if not isinstance(response, dict):
            response = _as_dict(response)
        response_id = response.get("id")
        if not response_id:
            return None
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                id=response_id,
                response_metadata={
                    "model_provider": "openrouter",
                    "id": response_id,
                },
            )
        )

    if event_type in ("response.completed", "response.done", "response.incomplete"):
        response = event.get("response") or {}
        if not isinstance(response, dict):
            response = _as_dict(response)
        return _completed_response_chunk(response, model_name=model_name)

    if event_type == "error" or event.get("error"):
        _raise_response_error(event.get("error") or event, streaming=True)

    return None
