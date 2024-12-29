"""Test OpenAI Chat API wrapper."""

import json
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    _convert_to_openai_response_format,
    _create_usage_metadata,
    _format_message_content,
)


def test_openai_model_param() -> None:
    llm = ChatOpenAI(model="foo")
    assert llm.model_name == "foo"
    llm = ChatOpenAI(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"

    llm = ChatOpenAI(max_tokens=10)  # type: ignore[call-arg]
    assert llm.max_tokens == 10
    llm = ChatOpenAI(max_completion_tokens=10)
    assert llm.max_tokens == 10


def test_openai_o1_temperature() -> None:
    llm = ChatOpenAI(model="o1-preview")
    assert llm.temperature == 1
    llm = ChatOpenAI(model_name="o1-mini")  # type: ignore[call-arg]
    assert llm.temperature == 1


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {"role": "function", "name": name, "content": content}
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_human_with_name() -> None:
    message = {"role": "user", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_ai_with_name() -> None:
    message = {"role": "assistant", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_developer() -> None:
    message = {"role": "developer", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(
        content="foo", additional_kwargs={"__openai_role__": "developer"}
    )
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
        "function": {
            "arguments": '{"name": "Sally", "hair_color": "green"}',
            "name": "GenerateUsername",
        },
        "type": "function",
    }
    message = {"role": "assistant", "content": None, "tool_calls": [raw_tool_call]}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": [raw_tool_call]},
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message

    # Test malformed tool call
    raw_tool_calls: list = [
        {
            "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
            "function": {"arguments": "oops", "name": "GenerateUsername"},
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"name": "Sally", "hair_color": "green"}',
                "name": "GenerateUsername",
            },
            "type": "function",
        },
    ]
    raw_tool_calls = list(sorted(raw_tool_calls, key=lambda x: x["id"]))
    message = {"role": "assistant", "content": None, "tool_calls": raw_tool_calls}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": raw_tool_calls},
        invalid_tool_calls=[
            InvalidToolCall(
                name="GenerateUsername",
                args="oops",
                id="call_wm0JY6CdwOMZ4eTxHWUThDNz",
                error=(
                    "Function GenerateUsername arguments:\n\noops\n\nare not "
                    "valid JSON. Received JSONDecodeError Expecting value: line 1 "
                    "column 1 (char 0)\nFor troubleshooting, visit: https://python"
                    ".langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE "
                ),
                type="invalid_tool_call",
            )
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_abc123",
                type="tool_call",
            )
        ],
    )
    assert result == expected_output
    reverted_message_dict = _convert_message_to_dict(expected_output)
    reverted_message_dict["tool_calls"] = list(
        sorted(reverted_message_dict["tool_calls"], key=lambda x: x["id"])
    )
    assert reverted_message_dict == message


class MockAsyncContextManager:
    def __init__(self, chunk_list: list):
        self.current_chunk = 0
        self.chunk_list = chunk_list
        self.chunk_num = len(chunk_list)

    async def __aenter__(self) -> "MockAsyncContextManager":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        pass

    def __aiter__(self) -> "MockAsyncContextManager":
        return self

    async def __anext__(self) -> dict:
        if self.current_chunk < self.chunk_num:
            chunk = self.chunk_list[self.current_chunk]
            self.current_chunk += 1
            return chunk
        else:
            raise StopAsyncIteration


class MockSyncContextManager:
    def __init__(self, chunk_list: list):
        self.current_chunk = 0
        self.chunk_list = chunk_list
        self.chunk_num = len(chunk_list)

    def __enter__(self) -> "MockSyncContextManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        pass

    def __iter__(self) -> "MockSyncContextManager":
        return self

    def __next__(self) -> dict:
        if self.current_chunk < self.chunk_num:
            chunk = self.chunk_list[self.current_chunk]
            self.current_chunk += 1
            return chunk
        else:
            raise StopIteration


GLM4_STREAM_META = """{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"\u4eba\u5de5\u667a\u80fd"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"\u52a9\u624b"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"，"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"\u4f60\u53ef\u4ee5"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"\u53eb\u6211"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"AI"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"\u52a9\u624b"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"delta":{"role":"assistant","content":"。"}}]}
{"id":"20240722102053e7277a4f94e848248ff9588ed37fb6e6","created":1721614853,"model":"glm-4","choices":[{"index":0,"finish_reason":"stop","delta":{"role":"assistant","content":""}}],"usage":{"prompt_tokens":13,"completion_tokens":10,"total_tokens":23}}
[DONE]"""  # noqa: E501


@pytest.fixture
def mock_glm4_completion() -> list:
    list_chunk_data = GLM4_STREAM_META.split("\n")
    result_list = []
    for msg in list_chunk_data:
        if msg != "[DONE]":
            result_list.append(json.loads(msg))

    return result_list


async def test_glm4_astream(mock_glm4_completion: list) -> None:
    llm_name = "glm-4"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = AsyncMock()

    async def mock_create(*args: Any, **kwargs: Any) -> MockAsyncContextManager:
        return MockAsyncContextManager(mock_glm4_completion)

    mock_client.create = mock_create
    usage_chunk = mock_glm4_completion[-1]

    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "async_client", mock_client):
        async for chunk in llm.astream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


def test_glm4_stream(mock_glm4_completion: list) -> None:
    llm_name = "glm-4"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(mock_glm4_completion)

    mock_client.create = mock_create
    usage_chunk = mock_glm4_completion[-1]

    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "client", mock_client):
        for chunk in llm.stream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


DEEPSEEK_STREAM_DATA = """{"id":"d3610c24e6b42518a7883ea57c3ea2c3","choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":null,"logprobs":null}],"created":1721630271,"model":"deepseek-chat","system_fingerprint":"fp_7e0991cad4","object":"chat.completion.chunk","usage":null}
{"choices":[{"delta":{"content":"我是","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"Deep","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"Seek","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":" Chat","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"，","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"一个","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"由","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"深度","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"求","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"索","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"公司","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"开发的","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"智能","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"助手","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"。","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":null}
{"choices":[{"delta":{"content":"","role":null},"finish_reason":"stop","index":0,"logprobs":null}],"created":1721630271,"id":"d3610c24e6b42518a7883ea57c3ea2c3","model":"deepseek-chat","object":"chat.completion.chunk","system_fingerprint":"fp_7e0991cad4","usage":{"completion_tokens":15,"prompt_tokens":11,"total_tokens":26}}
[DONE]"""  # noqa: E501


@pytest.fixture
def mock_deepseek_completion() -> List[Dict]:
    list_chunk_data = DEEPSEEK_STREAM_DATA.split("\n")
    result_list = []
    for msg in list_chunk_data:
        if msg != "[DONE]":
            result_list.append(json.loads(msg))

    return result_list


async def test_deepseek_astream(mock_deepseek_completion: list) -> None:
    llm_name = "deepseek-chat"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = AsyncMock()

    async def mock_create(*args: Any, **kwargs: Any) -> MockAsyncContextManager:
        return MockAsyncContextManager(mock_deepseek_completion)

    mock_client.create = mock_create
    usage_chunk = mock_deepseek_completion[-1]
    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "async_client", mock_client):
        async for chunk in llm.astream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


def test_deepseek_stream(mock_deepseek_completion: list) -> None:
    llm_name = "deepseek-chat"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(mock_deepseek_completion)

    mock_client.create = mock_create
    usage_chunk = mock_deepseek_completion[-1]
    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "client", mock_client):
        for chunk in llm.stream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


OPENAI_STREAM_DATA = """{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"usage":null}
{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[{"index":0,"delta":{"content":"我是"},"logprobs":null,"finish_reason":null}],"usage":null}
{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[{"index":0,"delta":{"content":"助手"},"logprobs":null,"finish_reason":null}],"usage":null}
{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[{"index":0,"delta":{"content":"。"},"logprobs":null,"finish_reason":null}],"usage":null}
{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}],"usage":null}
{"id":"chatcmpl-9nhARrdUiJWEMd5plwV1Gc9NCjb9M","object":"chat.completion.chunk","created":1721631035,"model":"gpt-4o-2024-05-13","system_fingerprint":"fp_18cc0f1fa0","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":3,"total_tokens":17}}
[DONE]"""  # noqa: E501


@pytest.fixture
def mock_openai_completion() -> List[Dict]:
    list_chunk_data = OPENAI_STREAM_DATA.split("\n")
    result_list = []
    for msg in list_chunk_data:
        if msg != "[DONE]":
            result_list.append(json.loads(msg))

    return result_list


async def test_openai_astream(mock_openai_completion: list) -> None:
    llm_name = "gpt-4o"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = AsyncMock()

    async def mock_create(*args: Any, **kwargs: Any) -> MockAsyncContextManager:
        return MockAsyncContextManager(mock_openai_completion)

    mock_client.create = mock_create
    usage_chunk = mock_openai_completion[-1]
    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "async_client", mock_client):
        async for chunk in llm.astream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


def test_openai_stream(mock_openai_completion: list) -> None:
    llm_name = "gpt-4o"
    llm = ChatOpenAI(model=llm_name, stream_usage=True)
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(mock_openai_completion)

    mock_client.create = mock_create
    usage_chunk = mock_openai_completion[-1]
    usage_metadata: Optional[UsageMetadata] = None
    with patch.object(llm, "client", mock_client):
        for chunk in llm.stream("你的名字叫什么？只回答名字"):
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                usage_metadata = chunk.usage_metadata

    assert usage_metadata is not None

    assert usage_metadata["input_tokens"] == usage_chunk["usage"]["prompt_tokens"]
    assert usage_metadata["output_tokens"] == usage_chunk["usage"]["completion_tokens"]
    assert usage_metadata["total_tokens"] == usage_chunk["usage"]["total_tokens"]


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Bar Baz", "name": "Erick"},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def mock_client(mock_completion: dict) -> MagicMock:
    rtn = MagicMock()

    mock_create = MagicMock()

    mock_resp = MagicMock()
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.parse.return_value = mock_completion
    mock_create.return_value = mock_resp

    rtn.with_raw_response.create = mock_create
    rtn.create.return_value = mock_completion
    return rtn


@pytest.fixture
def mock_async_client(mock_completion: dict) -> AsyncMock:
    rtn = AsyncMock()

    mock_create = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.parse.return_value = mock_completion
    mock_create.return_value = mock_resp

    rtn.with_raw_response.create = mock_create
    rtn.create.return_value = mock_completion
    return rtn


def test_openai_invoke(mock_client: MagicMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "client", mock_client):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"

        # headers are not in response_metadata if include_response_headers not set
        assert "headers" not in res.response_metadata
    assert mock_client.create.called


async def test_openai_ainvoke(mock_async_client: AsyncMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "async_client", mock_async_client):
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"

        # headers are not in response_metadata if include_response_headers not set
        assert "headers" not in res.response_metadata
    assert mock_async_client.create.called


@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-3.5-0125",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview",
    ],
)
def test__get_encoding_model(model: str) -> None:
    ChatOpenAI(model=model)._get_encoding_model()
    return


def test_openai_invoke_name(mock_client: MagicMock) -> None:
    llm = ChatOpenAI()

    with patch.object(llm, "client", mock_client):
        messages = [HumanMessage(content="Foo", name="Katie")]
        res = llm.invoke(messages)
        call_args, call_kwargs = mock_client.create.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Katie"

        # check return type has name
        assert res.content == "Bar Baz"
        assert res.name == "Erick"


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> List[int]:
        return [1, 2, 3]

    llm = ChatOpenAI(custom_get_token_ids=token_encoder)
    assert llm.get_token_ids("foo") == [1, 2, 3]


def test_format_message_content() -> None:
    content: Any = "hello"
    assert content == _format_message_content(content)

    content = None
    assert content == _format_message_content(content)

    content = []
    assert content == _format_message_content(content)

    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "url.com"}},
    ]
    assert content == _format_message_content(content)

    content = [
        {"type": "text", "text": "hello"},
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA", "unit": "celsius"},
        },
    ]
    assert [{"type": "text", "text": "hello"}] == _format_message_content(content)


class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


class MakeASandwich(BaseModel):
    "Make a sandwich given a list of ingredients."

    bread_type: str
    cheese_type: str
    condiments: List[str]
    vegetables: List[str]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "any",
        "none",
        "auto",
        "required",
        "GenerateUsername",
        {"type": "function", "function": {"name": "MakeASandwich"}},
        False,
        None,
    ],
)
@pytest.mark.parametrize("strict", [True, False, None])
def test_bind_tools_tool_choice(tool_choice: Any, strict: Optional[bool]) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm.bind_tools(
        tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice, strict=strict
    )


@pytest.mark.parametrize(
    "schema", [GenerateUsername, GenerateUsername.model_json_schema()]
)
@pytest.mark.parametrize("method", ["json_schema", "function_calling", "json_mode"])
@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("strict", [True, False, None])
def test_with_structured_output(
    schema: Union[Type, Dict[str, Any], None],
    method: Literal["function_calling", "json_mode", "json_schema"],
    include_raw: bool,
    strict: Optional[bool],
) -> None:
    """Test passing in manually construct tool call message."""
    if method == "json_mode":
        strict = None
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm.with_structured_output(
        schema, method=method, strict=strict, include_raw=include_raw
    )


def test_get_num_tokens_from_messages() -> None:
    llm = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage("you're a good assistant"),
        HumanMessage("how are you"),
        HumanMessage(
            [
                {"type": "text", "text": "what's in this image"},
                {"type": "image_url", "image_url": {"url": "https://foobar.com"}},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://foobar.com", "detail": "low"},
                },
            ]
        ),
        AIMessage("a nice bird"),
        AIMessage(
            "",
            tool_calls=[
                ToolCall(id="foo", name="bar", args={"arg1": "arg1"}, type="tool_call")
            ],
        ),
        AIMessage(
            "",
            additional_kwargs={
                "function_call": {
                    "arguments": json.dumps({"arg1": "arg1"}),
                    "name": "fun",
                }
            },
        ),
        AIMessage(
            "text",
            tool_calls=[
                ToolCall(id="foo", name="bar", args={"arg1": "arg1"}, type="tool_call")
            ],
        ),
        ToolMessage("foobar", tool_call_id="foo"),
    ]
    expected = 176
    actual = llm.get_num_tokens_from_messages(messages)
    assert expected == actual


class Foo(BaseModel):
    bar: int


# class FooV1(BaseModelV1):
#     bar: int


@pytest.mark.parametrize(
    "schema",
    [
        Foo
        # FooV1
    ],
)
def test_schema_from_with_structured_output(schema: Type) -> None:
    """Test schema from with_structured_output."""

    llm = ChatOpenAI()

    structured_llm = llm.with_structured_output(
        schema, method="json_schema", strict=True
    )

    expected = {
        "properties": {"bar": {"title": "Bar", "type": "integer"}},
        "required": ["bar"],
        "title": schema.__name__,
        "type": "object",
    }
    actual = structured_llm.get_output_schema().model_json_schema()
    assert actual == expected


def test__create_usage_metadata() -> None:
    usage_metadata = {
        "completion_tokens": 15,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
        "prompt_tokens": 11,
        "total_tokens": 26,
    }
    result = _create_usage_metadata(usage_metadata)
    assert result == UsageMetadata(
        output_tokens=15,
        input_tokens=11,
        total_tokens=26,
        input_token_details={},
        output_token_details={},
    )


def test__convert_to_openai_response_format() -> None:
    # Test response formats that aren't tool-like.
    response_format: dict = {
        "type": "json_schema",
        "json_schema": {
            "name": "math_reasoning",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"},
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False,
                        },
                    },
                    "final_answer": {"type": "string"},
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    actual = _convert_to_openai_response_format(response_format)
    assert actual == response_format

    actual = _convert_to_openai_response_format(response_format["json_schema"])
    assert actual == response_format

    actual = _convert_to_openai_response_format(response_format, strict=True)
    assert actual == response_format

    with pytest.raises(ValueError):
        _convert_to_openai_response_format(response_format, strict=False)


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
@pytest.mark.parametrize("strict", [True, None])
def test_structured_output_strict(
    method: Literal["function_calling", "json_schema"], strict: Optional[bool]
) -> None:
    """Test to verify structured output with strict=True."""

    llm = ChatOpenAI(model="gpt-4o-2024-08-06")

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    llm.with_structured_output(Joke, method=method, strict=strict)
    # Schema
    llm.with_structured_output(Joke.model_json_schema(), method=method, strict=strict)


def test_nested_structured_output_strict() -> None:
    """Test to verify structured output with strict=True for nested object."""

    llm = ChatOpenAI(model="gpt-4o-2024-08-06")

    class SelfEvaluation(TypedDict):
        score: int
        text: str

    class JokeWithEvaluation(TypedDict):
        """Joke to tell user."""

        setup: str
        punchline: str
        self_evaluation: SelfEvaluation

    llm.with_structured_output(JokeWithEvaluation, method="json_schema")


def test__get_request_payload() -> None:
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
    messages: list = [
        SystemMessage("hello"),
        SystemMessage("bye", additional_kwargs={"__openai_role__": "developer"}),
        {"role": "human", "content": "how are you"},
    ]
    expected = {
        "messages": [
            {"role": "system", "content": "hello"},
            {"role": "developer", "content": "bye"},
            {"role": "user", "content": "how are you"},
        ],
        "model": "gpt-4o-2024-08-06",
        "stream": False,
        "n": 1,
        "temperature": 0.7,
    }
    payload = llm._get_request_payload(messages)
    assert payload == expected


def test_init_o1() -> None:
    with pytest.warns(None) as record:  # type: ignore[call-overload]
        ChatOpenAI(model="o1", reasoning_effort="medium")
    assert len(record) == 0
