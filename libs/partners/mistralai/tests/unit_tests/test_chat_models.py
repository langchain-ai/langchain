"""Test MistralAI Chat API wrapper."""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)
from pydantic import SecretStr

from langchain_mistralai.chat_models import (  # type: ignore[import]
    ChatMistralAI,
    _convert_message_to_mistral_chat_message,
    _convert_mistral_chat_message_to_message,
    _convert_tool_call_id_to_mistral_compatible,
    _is_valid_mistral_tool_call_id,
)

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistralai_model_param() -> None:
    llm = ChatMistralAI(model="foo")  # type: ignore[call-arg]
    assert llm.model == "foo"


def test_mistralai_initialization() -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatMistralAI(model="test", mistral_api_key="test"),  # type: ignore[call-arg, call-arg]
        ChatMistralAI(model="test", api_key="test"),  # type: ignore[call-arg, arg-type]
    ]:
        assert cast("SecretStr", model.mistral_api_key).get_secret_value() == "test"


@pytest.mark.parametrize(
    ("model", "expected_url"),
    [
        (ChatMistralAI(model="test"), "https://api.mistral.ai/v1"),  # type: ignore[call-arg, arg-type]
        (ChatMistralAI(model="test", endpoint="baz"), "baz"),  # type: ignore[call-arg, arg-type]
    ],
)
def test_mistralai_initialization_baseurl(
    model: ChatMistralAI, expected_url: str
) -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized providing endpoint, but also
    # with default

    assert model.endpoint == expected_url


@pytest.mark.parametrize(
    "env_var_name",
    [
        ("MISTRAL_BASE_URL"),
    ],
)
def test_mistralai_initialization_baseurl_env(env_var_name: str) -> None:
    """Test ChatMistralAI initialization."""
    # Verify that ChatMistralAI can be initialized using env variable
    import os

    os.environ[env_var_name] = "boo"
    model = ChatMistralAI(model="test")  # type: ignore[call-arg]
    assert model.endpoint == "boo"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="Hello"),
            {"role": "system", "content": "Hello"},
        ),
        (
            HumanMessage(content="Hello"),
            {"role": "user", "content": "Hello"},
        ),
        (
            AIMessage(content="Hello"),
            {"role": "assistant", "content": "Hello"},
        ),
        (
            AIMessage(content="{", additional_kwargs={"prefix": True}),
            {"role": "assistant", "content": "{", "prefix": True},
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            {"role": "assistant", "content": "Hello"},
        ),
    ],
)
def test_convert_message_to_mistral_chat_message(
    message: BaseMessage, expected: dict
) -> None:
    result = _convert_message_to_mistral_chat_message(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> dict:
    return {
        "id": "abc123",
        "model": "fake_model",
        "choices": [
            {
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None,
            }
        ],
    }


def mock_chat_stream(*args: Any, **kwargs: Any) -> Generator:
    def it() -> Generator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


async def mock_chat_astream(*args: Any, **kwargs: Any) -> AsyncGenerator:
    async def it() -> AsyncGenerator:
        for token in ["Hello", " how", " can", " I", " help", "?"]:
            yield _make_completion_response_from_token(token)

    return it()


class MyCustomHandler(BaseCallbackHandler):
    last_token: str = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.last_token = token


@patch(
    "langchain_mistralai.chat_models.ChatMistralAI.completion_with_retry",
    new=mock_chat_stream,
)
def test_stream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    for token in chat.stream("Hello"):
        assert callback.last_token == token.content


@patch("langchain_mistralai.chat_models.acompletion_with_retry", new=mock_chat_astream)
async def test_astream_with_callback() -> None:
    callback = MyCustomHandler()
    chat = ChatMistralAI(callbacks=[callback])
    async for token in chat.astream("Hello"):
        assert callback.last_token == token.content


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "ssAbar4Dr",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
    }
    message = {"role": "assistant", "content": "", "tool_calls": [raw_tool_call]}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="ssAbar4Dr",
                type="tool_call",
            )
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output
    assert _convert_message_to_mistral_chat_message(expected_output) == message

    # Test malformed tool call
    raw_tool_calls = [
        {
            "id": "pL5rEGzxe",
            "function": {
                "arguments": '{"name": "Sally", "hair_color": "green"}',
                "name": "GenerateUsername",
            },
        },
        {
            "id": "ssAbar4Dr",
            "function": {
                "arguments": "oops",
                "name": "GenerateUsername",
            },
        },
    ]
    message = {"role": "assistant", "content": "", "tool_calls": raw_tool_calls}
    result = _convert_mistral_chat_message_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
                id="ssAbar4Dr",
                type="invalid_tool_call",
            ),
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="pL5rEGzxe",
                type="tool_call",
            ),
        ],
        response_metadata={"model_provider": "mistralai"},
    )
    assert result == expected_output
    assert _convert_message_to_mistral_chat_message(expected_output) == message


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> list[int]:
        return [1, 2, 3]

    llm = ChatMistralAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]


def test_tool_id_conversion() -> None:
    assert _is_valid_mistral_tool_call_id("ssAbar4Dr")
    assert not _is_valid_mistral_tool_call_id("abc123")
    assert not _is_valid_mistral_tool_call_id("call_JIIjI55tTipFFzpcP8re3BpM")

    result_map = {
        "ssAbar4Dr": "ssAbar4Dr",
        "abc123": "pL5rEGzxe",
        "call_JIIjI55tTipFFzpcP8re3BpM": "8kxAQvoED",
    }
    for input_id, expected_output in result_map.items():
        assert _convert_tool_call_id_to_mistral_compatible(input_id) == expected_output
        assert _is_valid_mistral_tool_call_id(expected_output)


def test_extra_kwargs() -> None:
    # Check that foo is saved in extra_kwargs.
    llm = ChatMistralAI(model="my-model", foo=3, max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatMistralAI(model="my-model", foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatMistralAI(model="my-model", foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]


def test_retry_with_failure_then_success() -> None:
    """Test retry mechanism works correctly when fiest request fails, second succeed."""
    # Create a real ChatMistralAI instance
    chat = ChatMistralAI(max_retries=3)

    # Set up the actual retry mechanism (not just mocking it)
    # We'll track how many times the function is called
    call_count = 0

    def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            msg = "Connection error"
            raise httpx.RequestError(msg, request=MagicMock())

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        return mock_response

    with patch.object(chat.client, "post", side_effect=mock_post):
        result = chat.invoke("Hello")
        assert result.content == "Hello!"
        assert call_count == 2, f"Expected 2 calls, but got {call_count}"


def test_no_duplicate_tool_calls_when_multiple_tools() -> None:
    """
    Tests wether the conversion of an AIMessage with more than one tool call
    to a Mistral assistant message correctly returns each tool call exactly
    once in the final payload.

    The current implementation uses a faulty for loop which produces N*N entries in the
    final tool_calls array of the payload (and thus duplicates tool call ids).
    """
    msg = AIMessage(
        content="",  # content should be blank when tool_calls are present
        tool_calls=[
            ToolCall(name="tool_a", args={"x": 1}, id="id_a", type="tool_call"),
            ToolCall(name="tool_b", args={"y": 2}, id="id_b", type="tool_call"),
        ],
        response_metadata={"model_provider": "mistralai"},
    )

    mistral_msg = _convert_message_to_mistral_chat_message(msg)

    assert mistral_msg["role"] == "assistant"
    assert "tool_calls" in mistral_msg, "Expected tool_calls to be present."

    tool_calls = mistral_msg["tool_calls"]
    # With the bug, this would be 4 (2x2); we expect exactly 2 entries.
    assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

    # Ensure there are no duplicate ids
    ids = [tc.get("id") for tc in tool_calls if isinstance(tc, dict)]
    assert len(ids) == 2
    assert len(set(ids)) == 2, f"Duplicate tool call IDs found: {ids}"


def test_profile() -> None:
    model = ChatMistralAI(model="mistral-large-latest")  # type: ignore[call-arg]
    assert model.profile
