"""Tests for ModelFallbackMiddleware response normalisation."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from langchain.agents.middleware.model_fallback import (
    _normalise_ai_message,
    _normalise_fallback_response,
)
from langchain.agents.middleware.types import ModelResponse


# ---------------------------------------------------------------------------
# _normalise_ai_message
# ---------------------------------------------------------------------------


def test_normalise_ai_message_string_content_unchanged() -> None:
    """String content is returned as-is."""
    msg = AIMessage(content="hello")
    assert _normalise_ai_message(msg) is msg


def test_normalise_ai_message_empty_content_unchanged() -> None:
    """Empty list content is returned as-is."""
    msg = AIMessage(content=[])
    assert _normalise_ai_message(msg) is msg


def test_normalise_ai_message_no_function_call_unchanged() -> None:
    """Content with no function_call blocks is returned as-is."""
    msg = AIMessage(content=[{"type": "text", "text": "hi"}])
    assert _normalise_ai_message(msg) is msg


def test_normalise_ai_message_function_call_converted() -> None:
    """function_call block is converted to tool_use."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            }
        ]
    )
    result = _normalise_ai_message(msg)
    assert result is not msg
    assert result.content == [
        {
            "type": "tool_use",
            "id": "call_abc",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]


def test_normalise_ai_message_function_call_call_id() -> None:
    """call_id is used as fallback for toolUseId."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "call_id": "c_xyz",
                "name": "search",
                "arguments": "{}",
            }
        ]
    )
    result = _normalise_ai_message(msg)
    assert result.content[0]["id"] == "c_xyz"


def test_normalise_ai_message_function_call_callId() -> None:
    """callId (camelCase) is used as fallback for toolUseId."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "callId": "c_camel",
                "name": "noop",
                "arguments": "{}",
            }
        ]
    )
    result = _normalise_ai_message(msg)
    assert result.content[0]["id"] == "c_camel"


def test_normalise_ai_message_function_call_dict_arguments() -> None:
    """arguments already a dict is preserved."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "c1",
                "name": "noop",
                "arguments": {"x": 1},
            }
        ]
    )
    result = _normalise_ai_message(msg)
    assert result.content[0]["input"] == {"x": 1}


def test_normalise_ai_message_mixed_blocks() -> None:
    """text block is preserved alongside converted function_call block."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Sure."},
            {
                "type": "function_call",
                "id": "call_m",
                "name": "get_weather",
                "arguments": '{"city": "London"}',
            },
        ]
    )
    result = _normalise_ai_message(msg)
    assert result.content == [
        {"type": "text", "text": "Sure."},
        {
            "type": "tool_use",
            "id": "call_m",
            "name": "get_weather",
            "input": {"city": "London"},
        },
    ]


def test_normalise_ai_message_invalid_json_arguments() -> None:
    """Malformed JSON in arguments produces empty dict input without raising."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "c1",
                "name": "bad",
                "arguments": "{not valid json",
            }
        ]
    )
    result = _normalise_ai_message(msg)
    assert result.content[0]["input"] == {}


# ---------------------------------------------------------------------------
# _normalise_fallback_response
# ---------------------------------------------------------------------------


def test_normalise_fallback_response_ai_message() -> None:
    """AIMessage response is normalised directly."""
    msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "c1",
                "name": "fn",
                "arguments": "{}",
            }
        ]
    )
    result = _normalise_fallback_response(msg)
    assert isinstance(result, AIMessage)
    assert result.content[0]["type"] == "tool_use"


def test_normalise_fallback_response_model_response() -> None:
    """ModelResponse result messages are normalised."""
    ai_msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "c2",
                "name": "fn",
                "arguments": '{"k": "v"}',
            }
        ]
    )
    response = ModelResponse(result=[ai_msg])
    result = _normalise_fallback_response(response)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content[0]["type"] == "tool_use"
    assert result.result[0].content[0]["input"] == {"k": "v"}


def test_normalise_fallback_response_no_function_call_unchanged() -> None:
    """ModelResponse with no function_call blocks passes through unchanged content."""
    ai_msg = AIMessage(content="plain text")
    response = ModelResponse(result=[ai_msg])
    result = _normalise_fallback_response(response)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "plain text"


def test_normalise_fallback_response_primary_not_normalised() -> None:
    """Primary model responses are NOT normalised (normalisation only in fallback path)."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
    from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse

    from typing import cast
    from langgraph.runtime import Runtime

    primary_msg = AIMessage(
        content=[
            {
                "type": "function_call",
                "id": "c_primary",
                "name": "fn",
                "arguments": "{}",
            }
        ]
    )
    primary_model = GenericFakeChatModel(messages=iter([primary_msg]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback")]))

    middleware = ModelFallbackMiddleware(fallback_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    request = ModelRequest(
        model=primary_model,
        system_prompt=None,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=[]),
        runtime=cast("Runtime", object()),
        model_settings={},
    )

    response = middleware.wrap_model_call(request, mock_handler)
    assert isinstance(response, ModelResponse)
    # Primary response is NOT normalised
    assert response.result[0].content[0]["type"] == "function_call"
