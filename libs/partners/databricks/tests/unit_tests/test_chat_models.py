"""Test chat model integration."""

import json
from typing import Generator
from unittest import mock

import mlflow  # type: ignore # noqa: F401
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_databricks.chat_models import (
    ChatDatabricks,
    _convert_dict_to_message,
    _convert_dict_to_message_chunk,
    _convert_message_to_dict,
)

_MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl_id",
    "object": "chat.completion",
    "created": 1721875529,
    "model": "meta-llama-3.1-70b-instruct-072424",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "To calculate the result of 36939 multiplied by 8922.4, "
                "I get:\n\n36939 x 8922.4 = 329,511,111.6",
            },
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
}

_MOCK_STREAM_RESPONSE = [
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "36939"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "x"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 22, "total_tokens": 52},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "8922.4"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 24, "total_tokens": 54},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": " = "},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 28, "total_tokens": 58},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "329,511,111.6"},
                "finish_reason": None,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 30, "total_tokens": 60},
    },
    {
        "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
        "object": "chat.completion.chunk",
        "created": 1721877054,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
    },
]


@pytest.fixture(autouse=True)
def mock_client() -> Generator:
    client = mock.MagicMock()
    client.predict.return_value = _MOCK_CHAT_RESPONSE
    client.predict_stream.return_value = _MOCK_STREAM_RESPONSE
    with mock.patch("mlflow.deployments.get_deploy_client", return_value=client):
        yield


@pytest.fixture
def llm() -> ChatDatabricks:
    return ChatDatabricks(
        endpoint="databricks-meta-llama-3-70b-instruct", target_uri="databricks"
    )


def test_chat_mlflow_predict(llm: ChatDatabricks) -> None:
    res = llm.invoke(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ]
    )
    assert res.content == _MOCK_CHAT_RESPONSE["choices"][0]["message"]["content"]  # type: ignore[index]


def test_chat_mlflow_stream(llm: ChatDatabricks) -> None:
    res = llm.stream(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "36939 * 8922.4"},
        ]
    )
    for chunk, expected in zip(res, _MOCK_STREAM_RESPONSE):
        assert chunk.content == expected["choices"][0]["delta"]["content"]  # type: ignore[index]


def test_chat_mlflow_bind_tools(llm: ChatDatabricks) -> None:
    class GetWeather(BaseModel):
        """Get the current weather in a given location"""

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )

    class GetPopulation(BaseModel):
        """Get the current population in a given location"""

        location: str = Field(
            ..., description="The city and state, e.g. San Francisco, CA"
        )

    llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
    response = llm_with_tools.invoke(
        "Which city is hotter today and which is bigger: LA or NY?"
    )
    assert isinstance(response, AIMessage)


### Test data conversion functions ###


@pytest.mark.parametrize(
    ("role", "expected_output"),
    [
        ("user", HumanMessage("foo")),
        ("system", SystemMessage("foo")),
        ("assistant", AIMessage("foo")),
        ("any_role", ChatMessage(content="foo", role="any_role")),
    ],
)
def test_convert_message(role: str, expected_output: BaseMessage) -> None:
    message = {"role": role, "content": "foo"}
    result = _convert_dict_to_message(message)
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == message


def test_convert_message_with_tool_calls() -> None:
    ID = "call_fb5f5e1a-bac0-4422-95e9-d06e6022ad12"
    tool_calls = [
        {
            "id": ID,
            "type": "function",
            "function": {
                "name": "main__test__python_exec",
                "arguments": '{"code": "result = 36939 * 8922.4"}',
            },
        }
    ]
    message_with_tools = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
        "id": ID,
    }
    result = _convert_dict_to_message(message_with_tools)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": tool_calls},
        id=ID,
        tool_calls=[
            {
                "name": tool_calls[0]["function"]["name"],  # type: ignore[index]
                "args": json.loads(tool_calls[0]["function"]["arguments"]),  # type: ignore[index]
                "id": ID,
                "type": "tool_call",
            }
        ],
    )
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == message_with_tools


@pytest.mark.parametrize(
    ("role", "expected_output"),
    [
        ("user", HumanMessageChunk(content="foo")),
        ("system", SystemMessageChunk(content="foo")),
        ("assistant", AIMessageChunk(content="foo")),
        ("any_role", ChatMessageChunk(content="foo", role="any_role")),
    ],
)
def test_convert_message_chunk(role: str, expected_output: BaseMessage) -> None:
    delta = {"role": role, "content": "foo"}
    result = _convert_dict_to_message_chunk(delta, "default_role")
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == delta


def test_convert_message_chunk_with_tool_calls() -> None:
    delta_with_tools = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"index": 0, "function": {"arguments": " }"}}],
    }
    result = _convert_dict_to_message_chunk(delta_with_tools, "role")
    expected_output = AIMessageChunk(
        content="",
        additional_kwargs={"tool_calls": delta_with_tools["tool_calls"]},
        id=None,
        tool_call_chunks=[ToolCallChunk(name=None, args=" }", id=None, index=0)],
    )
    assert result == expected_output


def test_convert_tool_message_chunk() -> None:
    delta = {
        "role": "tool",
        "content": "foo",
        "tool_call_id": "tool_call_id",
        "id": "some_id",
    }
    result = _convert_dict_to_message_chunk(delta, "default_role")
    expected_output = ToolMessageChunk(
        content="foo", id="some_id", tool_call_id="tool_call_id"
    )
    assert result == expected_output

    # convert back
    dict_result = _convert_message_to_dict(result)
    assert dict_result == delta


def test_convert_message_to_dict_function() -> None:
    with pytest.raises(ValueError, match="Function messages are not supported"):
        _convert_message_to_dict(FunctionMessage(content="", name="name"))
