"""Test MistralAI Chat API wrapper."""

import os
from typing import Any, AsyncGenerator, Dict, Generator, List, cast
from unittest.mock import patch

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
        assert cast(SecretStr, model.mistral_api_key).get_secret_value() == "test"


@pytest.mark.parametrize(
    "model,expected_url",
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
            dict(role="system", content="Hello"),
        ),
        (
            HumanMessage(content="Hello"),
            dict(role="user", content="Hello"),
        ),
        (
            AIMessage(content="Hello"),
            dict(role="assistant", content="Hello"),
        ),
        (
            AIMessage(content="{", additional_kwargs={"prefix": True}),
            dict(role="assistant", content="{", prefix=True),
        ),
        (
            ChatMessage(role="assistant", content="Hello"),
            dict(role="assistant", content="Hello"),
        ),
    ],
)
def test_convert_message_to_mistral_chat_message(
    message: BaseMessage, expected: Dict
) -> None:
    result = _convert_message_to_mistral_chat_message(message)
    assert result == expected


def _make_completion_response_from_token(token: str) -> Dict:
    return dict(
        id="abc123",
        model="fake_model",
        choices=[
            dict(
                index=0,
                delta=dict(content=token),
                finish_reason=None,
            )
        ],
    )


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
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
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
    )
    assert result == expected_output
    assert _convert_message_to_mistral_chat_message(expected_output) == message


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> List[int]:
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
