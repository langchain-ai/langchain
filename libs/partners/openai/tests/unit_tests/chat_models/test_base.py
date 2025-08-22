"""Test OpenAI Chat API wrapper."""

import json
from functools import partial
from types import TracebackType
from typing import Any, Literal, Optional, Union, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response import IncompleteDetails, Response, ResponseUsage
from openai.types.responses.response_error import ResponseError
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
    Result,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._compat import (
    _FUNCTION_CALL_IDS_MAP_KEY,
    _convert_from_v03_ai_message,
    _convert_to_v03_ai_message,
)
from langchain_openai.chat_models.base import (
    _construct_lc_result_from_responses_api,
    _construct_responses_api_input,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _convert_to_openai_response_format,
    _create_usage_metadata,
    _create_usage_metadata_responses,
    _format_message_content,
    _get_last_messages,
    _make_computer_call_output_from_message,
    _oai_structured_outputs_parser,
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


def test_openai_client_caching() -> None:
    """Test that the OpenAI client is cached."""
    llm1 = ChatOpenAI(model="gpt-4.1-mini")
    llm2 = ChatOpenAI(model="gpt-4.1-mini")
    assert llm1.root_client._client is llm2.root_client._client

    llm3 = ChatOpenAI(model="gpt-4.1-mini", base_url="foo")
    assert llm1.root_client._client is not llm3.root_client._client

    llm4 = ChatOpenAI(model="gpt-4.1-mini", timeout=None)
    assert llm1.root_client._client is llm4.root_client._client

    llm5 = ChatOpenAI(model="gpt-4.1-mini", timeout=3)
    assert llm1.root_client._client is not llm5.root_client._client

    llm6 = ChatOpenAI(
        model="gpt-4.1-mini", timeout=httpx.Timeout(timeout=60.0, connect=5.0)
    )
    assert llm1.root_client._client is not llm6.root_client._client

    llm7 = ChatOpenAI(model="gpt-4.1-mini", timeout=(5, 1))
    assert llm1.root_client._client is not llm7.root_client._client


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
        exc_type: Optional[type[BaseException]],
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
        exc_type: Optional[type[BaseException]],
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
def mock_deepseek_completion() -> list[dict]:
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
def mock_openai_completion() -> list[dict]:
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


def test_function_calls_with_tool_calls(mock_client: MagicMock) -> None:
    # Test that we ignore function calls if tool_calls are present
    llm = ChatOpenAI(model="gpt-4.1-mini")
    tool_call_message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "Boston"}',
            }
        },
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "Boston"},
                "id": "abc123",
                "type": "tool_call",
            }
        ],
    )
    messages = [
        HumanMessage("What's the weather in Boston?"),
        tool_call_message,
        ToolMessage(content="It's sunny.", name="get_weather", tool_call_id="abc123"),
    ]
    with patch.object(llm, "client", mock_client):
        _ = llm.invoke(messages)
        _, call_kwargs = mock_client.create.call_args
        call_messages = call_kwargs["messages"]
        tool_call_message_payload = call_messages[1]
        assert "tool_calls" in tool_call_message_payload
        assert "function_call" not in tool_call_message_payload

    # Test we don't ignore function calls if tool_calls are not present
    cast(AIMessage, messages[1]).tool_calls = []
    with patch.object(llm, "client", mock_client):
        _ = llm.invoke(messages)
        _, call_kwargs = mock_client.create.call_args
        call_messages = call_kwargs["messages"]
        tool_call_message_payload = call_messages[1]
        assert "function_call" in tool_call_message_payload
        assert "tool_calls" not in tool_call_message_payload


def test_custom_token_counting() -> None:
    def token_encoder(text: str) -> list[int]:
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

    # Standard multi-modal inputs
    content = [{"type": "image", "source_type": "url", "url": "https://..."}]
    expected = [{"type": "image_url", "image_url": {"url": "https://..."}}]
    assert expected == _format_message_content(content)

    content = [
        {
            "type": "image",
            "source_type": "base64",
            "data": "<base64 data>",
            "mime_type": "image/png",
        }
    ]
    expected = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,<base64 data>"},
        }
    ]
    assert expected == _format_message_content(content)

    content = [
        {
            "type": "file",
            "source_type": "base64",
            "data": "<base64 data>",
            "mime_type": "application/pdf",
            "filename": "my_file",
        }
    ]
    expected = [
        {
            "type": "file",
            "file": {
                "filename": "my_file",
                "file_data": "data:application/pdf;base64,<base64 data>",
            },
        }
    ]
    assert expected == _format_message_content(content)

    content = [{"type": "file", "source_type": "id", "id": "file-abc123"}]
    expected = [{"type": "file", "file": {"file_id": "file-abc123"}}]
    assert expected == _format_message_content(content)


class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


class MakeASandwich(BaseModel):
    "Make a sandwich given a list of ingredients."

    bread_type: str
    cheese_type: str
    condiments: list[str]
    vegetables: list[str]


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
    schema: Union[type, dict[str, Any], None],
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
    expected = 431  # Updated to match token count with mocked 100x100 image

    # Mock _url_to_size to avoid PIL dependency in unit tests
    with patch("langchain_openai.chat_models.base._url_to_size") as mock_url_to_size:
        mock_url_to_size.return_value = (100, 100)  # 100x100 pixel image
        actual = llm.get_num_tokens_from_messages(messages)

    assert expected == actual

    # Test file inputs
    messages = [
        HumanMessage(
            [
                "Summarize this document.",
                {
                    "type": "file",
                    "file": {
                        "filename": "my file",
                        "file_data": "data:application/pdf;base64,<data>",
                    },
                },
            ]
        )
    ]
    with pytest.warns(match="file inputs are not supported"):
        actual = llm.get_num_tokens_from_messages(messages)
        assert actual == 13


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
def test_schema_from_with_structured_output(schema: type) -> None:
    """Test schema from with_structured_output."""

    llm = ChatOpenAI(model="gpt-4o")

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


def test__create_usage_metadata_responses() -> None:
    response_usage_metadata = {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 50},
        "output_tokens": 50,
        "output_tokens_details": {"reasoning_tokens": 10},
        "total_tokens": 150,
    }
    result = _create_usage_metadata_responses(response_usage_metadata)

    assert result == UsageMetadata(
        output_tokens=50,
        input_tokens=100,
        total_tokens=150,
        input_token_details={"cache_read": 50},
        output_token_details={"reasoning": 10},
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
        _evaluation: SelfEvaluation

    llm.with_structured_output(JokeWithEvaluation, method="json_schema")


def test__get_request_payload() -> None:
    llm = ChatOpenAI(model="gpt-4o-2024-08-06")
    messages: list = [
        SystemMessage("hello"),
        SystemMessage("bye", additional_kwargs={"__openai_role__": "developer"}),
        SystemMessage(content=[{"type": "text", "text": "hello!"}]),
        {"role": "human", "content": "how are you"},
        {"role": "user", "content": [{"type": "text", "text": "feeling today"}]},
    ]
    expected = {
        "messages": [
            {"role": "system", "content": "hello"},
            {"role": "developer", "content": "bye"},
            {"role": "system", "content": [{"type": "text", "text": "hello!"}]},
            {"role": "user", "content": "how are you"},
            {"role": "user", "content": [{"type": "text", "text": "feeling today"}]},
        ],
        "model": "gpt-4o-2024-08-06",
        "stream": False,
    }
    payload = llm._get_request_payload(messages)
    assert payload == expected

    # Test we coerce to developer role for o-series models
    llm = ChatOpenAI(model="o3-mini")
    payload = llm._get_request_payload(messages)
    expected = {
        "messages": [
            {"role": "developer", "content": "hello"},
            {"role": "developer", "content": "bye"},
            {"role": "developer", "content": [{"type": "text", "text": "hello!"}]},
            {"role": "user", "content": "how are you"},
            {"role": "user", "content": [{"type": "text", "text": "feeling today"}]},
        ],
        "model": "o3-mini",
        "stream": False,
    }
    assert payload == expected

    # Test we ignore reasoning blocks from other providers
    reasoning_messages: list = [
        {
            "role": "user",
            "content": [
                {"type": "reasoning_content", "reasoning_content": "reasoning..."},
                {"type": "text", "text": "reasoned response"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "thinking", "thinking": "thinking..."},
                {"type": "text", "text": "thoughtful response"},
            ],
        },
    ]
    expected = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "reasoned response"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "thoughtful response"}],
            },
        ],
        "model": "o3-mini",
        "stream": False,
    }
    payload = llm._get_request_payload(reasoning_messages)
    assert payload == expected


def test_init_o1() -> None:
    with pytest.warns(None) as record:  # type: ignore[call-overload]
        ChatOpenAI(model="o1", reasoning_effort="medium")
    assert len(record) == 0


def test_init_minimal_reasoning_effort() -> None:
    with pytest.warns(None) as record:  # type: ignore[call-overload]
        ChatOpenAI(model="gpt-5", reasoning_effort="minimal")
    assert len(record) == 0


@pytest.mark.parametrize("use_responses_api", [False, True])
@pytest.mark.parametrize("use_max_completion_tokens", [True, False])
def test_minimal_reasoning_effort_payload(
    use_max_completion_tokens: bool, use_responses_api: bool
) -> None:
    """Test that minimal reasoning effort is included in request payload."""
    if use_max_completion_tokens:
        kwargs = {"max_completion_tokens": 100}
    else:
        kwargs = {"max_tokens": 100}

    init_kwargs: dict[str, Any] = {
        "model": "gpt-5",
        "reasoning_effort": "minimal",
        "use_responses_api": use_responses_api,
        **kwargs,
    }

    if use_responses_api:
        init_kwargs["output_version"] = "responses/v1"

    llm = ChatOpenAI(**init_kwargs)

    messages = [
        {"role": "developer", "content": "respond with just 'test'"},
        {"role": "user", "content": "hello"},
    ]

    payload = llm._get_request_payload(messages, stop=None)

    # When using responses API, reasoning_effort becomes reasoning.effort
    if use_responses_api:
        assert "reasoning" in payload
        assert payload["reasoning"]["effort"] == "minimal"
        # For responses API, tokens param becomes max_output_tokens
        assert payload["max_output_tokens"] == 100
    else:
        # For non-responses API, reasoning_effort remains as is
        assert payload["reasoning_effort"] == "minimal"
        if use_max_completion_tokens:
            assert payload["max_completion_tokens"] == 100
        else:
            # max_tokens gets converted to max_completion_tokens in non-responses API
            assert payload["max_completion_tokens"] == 100


def test_verbosity_parameter_payload() -> None:
    """Test verbosity parameter is included in request payload for Responses API."""
    llm = ChatOpenAI(
        model="gpt-5",
        verbosity="high",
        use_responses_api=True,
        output_version="responses/v1",
    )

    messages = [{"role": "user", "content": "hello"}]
    payload = llm._get_request_payload(messages, stop=None)

    assert payload["text"]["verbosity"] == "high"
    assert payload["text"]["format"]["type"] == "text"


def test_structured_output_old_model() -> None:
    class Output(TypedDict):
        """output."""

        foo: str

    with pytest.warns(match="Cannot use method='json_schema'"):
        llm = ChatOpenAI(model="gpt-4").with_structured_output(Output)
    # assert tool calling was used instead of json_schema
    assert "tools" in llm.steps[0].kwargs  # type: ignore
    assert "response_format" not in llm.steps[0].kwargs  # type: ignore


def test_structured_outputs_parser() -> None:
    parsed_response = GenerateUsername(name="alice", hair_color="black")
    llm_output = ChatGeneration(
        message=AIMessage(
            content='{"name": "alice", "hair_color": "black"}',
            additional_kwargs={"parsed": parsed_response},
        )
    )
    output_parser = RunnableLambda(
        partial(_oai_structured_outputs_parser, schema=GenerateUsername)
    )
    serialized = dumps(llm_output)
    deserialized = loads(serialized)
    assert isinstance(deserialized, ChatGeneration)
    result = output_parser.invoke(deserialized.message)
    assert result == parsed_response


def test__construct_lc_result_from_responses_api_error_handling() -> None:
    """Test that errors in the response are properly raised."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        error=ResponseError(message="Test error", code="server_error"),
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[],
    )

    with pytest.raises(ValueError) as excinfo:
        _construct_lc_result_from_responses_api(response)

    assert "Test error" in str(excinfo.value)


def test__construct_lc_result_from_responses_api_basic_text_response() -> None:
    """Test a basic text response with no tools or special features."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Hello, world!", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
        usage=ResponseUsage(
            input_tokens=10,
            output_tokens=3,
            total_tokens=13,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    assert isinstance(result, ChatResult)
    assert len(result.generations) == 1
    assert isinstance(result.generations[0], ChatGeneration)
    assert isinstance(result.generations[0].message, AIMessage)
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Hello, world!", "annotations": []}
    ]
    assert result.generations[0].message.id == "msg_123"
    assert result.generations[0].message.usage_metadata
    assert result.generations[0].message.usage_metadata["input_tokens"] == 10
    assert result.generations[0].message.usage_metadata["output_tokens"] == 3
    assert result.generations[0].message.usage_metadata["total_tokens"] == 13
    assert result.generations[0].message.response_metadata["id"] == "resp_123"
    assert result.generations[0].message.response_metadata["model_name"] == "gpt-4o"

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Hello, world!", "annotations": [], "id": "msg_123"}
    ]
    assert result.generations[0].message.id == "resp_123"
    assert result.generations[0].message.response_metadata["id"] == "resp_123"


def test__construct_lc_result_from_responses_api_multiple_text_blocks() -> None:
    """Test a response with multiple text blocks."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="First part", annotations=[]
                    ),
                    ResponseOutputText(
                        type="output_text", text="Second part", annotations=[]
                    ),
                ],
                role="assistant",
                status="completed",
            )
        ],
    )

    result = _construct_lc_result_from_responses_api(response)

    assert len(result.generations[0].message.content) == 2
    assert result.generations[0].message.content == [
        {"type": "text", "text": "First part", "annotations": []},
        {"type": "text", "text": "Second part", "annotations": []},
    ]


def test__construct_lc_result_from_responses_api_multiple_messages() -> None:
    """Test a response with multiple text blocks."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(type="output_text", text="foo", annotations=[])
                ],
                role="assistant",
                status="completed",
            ),
            ResponseReasoningItem(
                type="reasoning",
                id="rs_123",
                summary=[Summary(type="summary_text", text="reasoning foo")],
            ),
            ResponseOutputMessage(
                type="message",
                id="msg_234",
                content=[
                    ResponseOutputText(type="output_text", text="bar", annotations=[])
                ],
                role="assistant",
                status="completed",
            ),
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    assert result.generations[0].message.content == [
        {"type": "text", "text": "foo", "annotations": []},
        {"type": "text", "text": "bar", "annotations": []},
    ]
    assert result.generations[0].message.additional_kwargs == {
        "reasoning": {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "reasoning foo"}],
            "id": "rs_123",
        }
    }
    assert result.generations[0].message.id == "msg_234"

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )

    assert result.generations[0].message.content == [
        {"type": "text", "text": "foo", "annotations": [], "id": "msg_123"},
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "reasoning foo"}],
            "id": "rs_123",
        },
        {"type": "text", "text": "bar", "annotations": [], "id": "msg_234"},
    ]
    assert result.generations[0].message.id == "resp_123"


def test__construct_lc_result_from_responses_api_refusal_response() -> None:
    """Test a response with a refusal."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputRefusal(
                        type="refusal", refusal="I cannot assist with that request."
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    assert result.generations[0].message.additional_kwargs["refusal"] == (
        "I cannot assist with that request."
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    assert result.generations[0].message.content == [
        {
            "type": "refusal",
            "refusal": "I cannot assist with that request.",
            "id": "msg_123",
        }
    ]


def test__construct_lc_result_from_responses_api_function_call_valid_json() -> None:
    """Test a response with a valid function call."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                id="func_123",
                call_id="call_123",
                name="get_weather",
                arguments='{"location": "New York", "unit": "celsius"}',
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    msg: AIMessage = cast(AIMessage, result.generations[0].message)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "tool_call"
    assert msg.tool_calls[0]["name"] == "get_weather"
    assert msg.tool_calls[0]["id"] == "call_123"
    assert msg.tool_calls[0]["args"] == {"location": "New York", "unit": "celsius"}
    assert _FUNCTION_CALL_IDS_MAP_KEY in result.generations[0].message.additional_kwargs
    assert (
        result.generations[0].message.additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][
            "call_123"
        ]
        == "func_123"
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    msg = cast(AIMessage, result.generations[0].message)
    assert msg.tool_calls
    assert msg.content == [
        {
            "type": "function_call",
            "id": "func_123",
            "name": "get_weather",
            "arguments": '{"location": "New York", "unit": "celsius"}',
            "call_id": "call_123",
        }
    ]


def test__construct_lc_result_from_responses_api_function_call_invalid_json() -> None:
    """Test a response with an invalid JSON function call."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                id="func_123",
                call_id="call_123",
                name="get_weather",
                arguments='{"location": "New York", "unit": "celsius"',
                # Missing closing brace
            )
        ],
    )

    result = _construct_lc_result_from_responses_api(response)

    msg: AIMessage = cast(AIMessage, result.generations[0].message)
    assert len(msg.invalid_tool_calls) == 1
    assert msg.invalid_tool_calls[0]["type"] == "invalid_tool_call"
    assert msg.invalid_tool_calls[0]["name"] == "get_weather"
    assert msg.invalid_tool_calls[0]["id"] == "call_123"
    assert (
        msg.invalid_tool_calls[0]["args"]
        == '{"location": "New York", "unit": "celsius"'
    )
    assert "error" in msg.invalid_tool_calls[0]
    assert _FUNCTION_CALL_IDS_MAP_KEY in result.generations[0].message.additional_kwargs


def test__construct_lc_result_from_responses_api_complex_response() -> None:
    """Test a complex response with multiple output types."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="Here's the information you requested:",
                        annotations=[],
                    )
                ],
                role="assistant",
                status="completed",
            ),
            ResponseFunctionToolCall(
                type="function_call",
                id="func_123",
                call_id="call_123",
                name="get_weather",
                arguments='{"location": "New York"}',
            ),
        ],
        metadata=dict(key1="value1", key2="value2"),
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        status="completed",
        user="user_123",
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    # Check message content
    assert result.generations[0].message.content == [
        {
            "type": "text",
            "text": "Here's the information you requested:",
            "annotations": [],
        }
    ]

    # Check tool calls
    msg: AIMessage = cast(AIMessage, result.generations[0].message)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "get_weather"

    # Check metadata
    assert result.generations[0].message.response_metadata["id"] == "resp_123"
    assert result.generations[0].message.response_metadata["metadata"] == {
        "key1": "value1",
        "key2": "value2",
    }
    assert result.generations[0].message.response_metadata["incomplete_details"] == {
        "reason": "max_output_tokens"
    }
    assert result.generations[0].message.response_metadata["status"] == "completed"
    assert result.generations[0].message.response_metadata["user"] == "user_123"

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    msg = cast(AIMessage, result.generations[0].message)
    assert msg.response_metadata["metadata"] == {"key1": "value1", "key2": "value2"}
    assert msg.content == [
        {
            "type": "text",
            "text": "Here's the information you requested:",
            "annotations": [],
            "id": "msg_123",
        },
        {
            "type": "function_call",
            "id": "func_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"location": "New York"}',
        },
    ]


def test__construct_lc_result_from_responses_api_no_usage_metadata() -> None:
    """Test a response without usage metadata."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Hello, world!", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            )
        ],
        # No usage field
    )

    result = _construct_lc_result_from_responses_api(response)

    assert cast(AIMessage, result.generations[0].message).usage_metadata is None


def test__construct_lc_result_from_responses_api_web_search_response() -> None:
    """Test a response with web search output."""
    from openai.types.responses.response_function_web_search import (
        ResponseFunctionWebSearch,
    )

    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFunctionWebSearch(
                id="websearch_123",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="search query"),
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 1
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["type"]
        == "web_search_call"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["id"]
        == "websearch_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["status"]
        == "completed"
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    assert result.generations[0].message.content == [
        {
            "type": "web_search_call",
            "id": "websearch_123",
            "status": "completed",
            "action": {"query": "search query", "type": "search"},
        }
    ]


def test__construct_lc_result_from_responses_api_file_search_response() -> None:
    """Test a response with file search output."""
    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseFileSearchToolCall(
                id="filesearch_123",
                type="file_search_call",
                status="completed",
                queries=["python code", "langchain"],
                results=[
                    Result(
                        file_id="file_123",
                        filename="example.py",
                        score=0.95,
                        text="def hello_world() -> None:\n    print('Hello, world!')",
                        attributes={"language": "python", "size": 42},
                    )
                ],
            )
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 1
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["type"]
        == "file_search_call"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["id"]
        == "filesearch_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["status"]
        == "completed"
    )
    assert result.generations[0].message.additional_kwargs["tool_outputs"][0][
        "queries"
    ] == ["python code", "langchain"]
    assert (
        len(
            result.generations[0].message.additional_kwargs["tool_outputs"][0][
                "results"
            ]
        )
        == 1
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["results"][
            0
        ]["file_id"]
        == "file_123"
    )
    assert (
        result.generations[0].message.additional_kwargs["tool_outputs"][0]["results"][
            0
        ]["score"]
        == 0.95
    )

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    assert result.generations[0].message.content == [
        {
            "type": "file_search_call",
            "id": "filesearch_123",
            "status": "completed",
            "queries": ["python code", "langchain"],
            "results": [
                {
                    "file_id": "file_123",
                    "filename": "example.py",
                    "score": 0.95,
                    "text": "def hello_world() -> None:\n    print('Hello, world!')",
                    "attributes": {"language": "python", "size": 42},
                }
            ],
        }
    ]


def test__construct_lc_result_from_responses_api_mixed_search_responses() -> None:
    """Test a response with both web search and file search outputs."""

    response = Response(
        id="resp_123",
        created_at=1234567890,
        model="gpt-4o",
        object="response",
        parallel_tool_calls=True,
        tools=[],
        tool_choice="auto",
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_123",
                content=[
                    ResponseOutputText(
                        type="output_text", text="Here's what I found:", annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
            ),
            ResponseFunctionWebSearch(
                id="websearch_123",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="search query"),
            ),
            ResponseFileSearchToolCall(
                id="filesearch_123",
                type="file_search_call",
                status="completed",
                queries=["python code"],
                results=[
                    Result(
                        file_id="file_123",
                        filename="example.py",
                        score=0.95,
                        text="def hello_world() -> None:\n    print('Hello, world!')",
                    )
                ],
            ),
        ],
    )

    # v0
    result = _construct_lc_result_from_responses_api(response)

    # Check message content
    assert result.generations[0].message.content == [
        {"type": "text", "text": "Here's what I found:", "annotations": []}
    ]

    # Check tool outputs
    assert "tool_outputs" in result.generations[0].message.additional_kwargs
    assert len(result.generations[0].message.additional_kwargs["tool_outputs"]) == 2

    # Check web search output
    web_search = next(
        output
        for output in result.generations[0].message.additional_kwargs["tool_outputs"]
        if output["type"] == "web_search_call"
    )
    assert web_search["id"] == "websearch_123"
    assert web_search["status"] == "completed"

    # Check file search output
    file_search = next(
        output
        for output in result.generations[0].message.additional_kwargs["tool_outputs"]
        if output["type"] == "file_search_call"
    )
    assert file_search["id"] == "filesearch_123"
    assert file_search["queries"] == ["python code"]
    assert file_search["results"][0]["filename"] == "example.py"

    # responses/v1
    result = _construct_lc_result_from_responses_api(
        response, output_version="responses/v1"
    )
    assert result.generations[0].message.content == [
        {
            "type": "text",
            "text": "Here's what I found:",
            "annotations": [],
            "id": "msg_123",
        },
        {
            "type": "web_search_call",
            "id": "websearch_123",
            "status": "completed",
            "action": {"type": "search", "query": "search query"},
        },
        {
            "type": "file_search_call",
            "id": "filesearch_123",
            "queries": ["python code"],
            "results": [
                {
                    "file_id": "file_123",
                    "filename": "example.py",
                    "score": 0.95,
                    "text": "def hello_world() -> None:\n    print('Hello, world!')",
                }
            ],
            "status": "completed",
        },
    ]


def test__construct_responses_api_input_human_message_with_text_blocks_conversion() -> (
    None
):
    """Test that human messages with text blocks are properly converted."""
    messages: list = [
        HumanMessage(content=[{"type": "text", "text": "What's in this image?"}])
    ]
    result = _construct_responses_api_input(messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 1
    assert result[0]["content"][0]["type"] == "input_text"
    assert result[0]["content"][0]["text"] == "What's in this image?"


def test__construct_responses_api_input_multiple_message_components() -> None:
    """Test that human messages with text blocks are properly converted."""
    # v0
    messages = [
        AIMessage(
            content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}],
            id="msg_123",
            response_metadata={"id": "resp_123"},
        )
    ]
    result = _construct_responses_api_input(messages)
    assert result == [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "foo", "annotations": []},
                {"type": "input_text", "text": "bar", "annotations": []},
            ],
            "id": "msg_123",
        }
    ]

    # responses/v1
    messages = [
        AIMessage(
            content=[
                {"type": "text", "text": "foo", "id": "msg_123"},
                {"type": "text", "text": "bar", "id": "msg_123"},
                {"type": "refusal", "refusal": "I refuse.", "id": "msg_123"},
                {"type": "text", "text": "baz", "id": "msg_234"},
            ]
        )
    ]
    result = _construct_responses_api_input(messages)

    assert result == [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "foo", "annotations": []},
                {"type": "input_text", "text": "bar", "annotations": []},
                {"type": "refusal", "refusal": "I refuse."},
            ],
            "id": "msg_123",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "baz", "annotations": []}],
            "id": "msg_234",
        },
    ]


def test__construct_responses_api_input_human_message_with_image_url_conversion() -> (
    None
):
    """Test that human messages with image_url blocks are properly converted."""
    messages: list = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                },
            ]
        )
    ]
    result = _construct_responses_api_input(messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text block conversion
    assert result[0]["content"][0]["type"] == "input_text"
    assert result[0]["content"][0]["text"] == "What's in this image?"

    # Check image block conversion
    assert result[0]["content"][1]["type"] == "input_image"
    assert result[0]["content"][1]["image_url"] == "https://example.com/image.jpg"
    assert result[0]["content"][1]["detail"] == "high"


def test__construct_responses_api_input_ai_message_with_tool_calls() -> None:
    """Test that AI messages with tool calls are properly converted."""
    tool_calls = [
        {
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "San Francisco"},
            "type": "tool_call",
        }
    ]

    ai_message = AIMessage(
        content=[
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
                "call_id": "call_123",
                "id": "fc_456",
            }
        ],
        tool_calls=tool_calls,
    )

    result = _construct_responses_api_input([ai_message])

    assert len(result) == 1
    assert result[0]["type"] == "function_call"
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == '{"location": "San Francisco"}'
    assert result[0]["call_id"] == "call_123"
    assert result[0]["id"] == "fc_456"

    # Message with only tool calls attribute provided
    ai_message = AIMessage(content="", tool_calls=tool_calls)

    result = _construct_responses_api_input([ai_message])

    assert len(result) == 1
    assert result[0]["type"] == "function_call"
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == '{"location": "San Francisco"}'
    assert result[0]["call_id"] == "call_123"
    assert "id" not in result[0]


def test__construct_responses_api_input_ai_message_with_tool_calls_and_content() -> (
    None
):
    """Test that AI messages with both tool calls and content are properly converted."""
    tool_calls = [
        {
            "id": "call_123",
            "name": "get_weather",
            "args": {"location": "San Francisco"},
            "type": "tool_call",
        }
    ]

    # Content blocks
    ai_message = AIMessage(
        content=[
            {"type": "text", "text": "I'll check the weather for you."},
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
                "call_id": "call_123",
                "id": "fc_456",
            },
        ],
        tool_calls=tool_calls,
    )

    result = _construct_responses_api_input([ai_message])

    assert len(result) == 2

    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == [
        {
            "type": "input_text",
            "text": "I'll check the weather for you.",
            "annotations": [],
        }
    ]

    assert result[1]["type"] == "function_call"
    assert result[1]["name"] == "get_weather"
    assert result[1]["arguments"] == '{"location": "San Francisco"}'
    assert result[1]["call_id"] == "call_123"
    assert result[1]["id"] == "fc_456"

    # String content
    ai_message = AIMessage(
        content="I'll check the weather for you.", tool_calls=tool_calls
    )

    result = _construct_responses_api_input([ai_message])

    assert len(result) == 2

    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == [
        {
            "type": "input_text",
            "text": "I'll check the weather for you.",
        }
    ]

    assert result[1]["type"] == "function_call"
    assert result[1]["name"] == "get_weather"
    assert result[1]["arguments"] == '{"location": "San Francisco"}'
    assert result[1]["call_id"] == "call_123"
    assert "id" not in result[1]


def test__construct_responses_api_input_tool_message_conversion() -> None:
    """Test that tool messages are properly converted to function_call_output."""
    messages = [
        ToolMessage(
            content='{"temperature": 72, "conditions": "sunny"}',
            tool_call_id="call_123",
        )
    ]

    result = _construct_responses_api_input(messages)

    assert len(result) == 1
    assert result[0]["type"] == "function_call_output"
    assert result[0]["output"] == '{"temperature": 72, "conditions": "sunny"}'
    assert result[0]["call_id"] == "call_123"


def test__construct_responses_api_input_multiple_message_types() -> None:
    """Test conversion of a conversation with multiple message types."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        SystemMessage(
            content=[{"type": "text", "text": "You are a very helpful assistant!"}]
        ),
        HumanMessage(content="What's the weather in San Francisco?"),
        HumanMessage(
            content=[{"type": "text", "text": "What's the weather in San Francisco?"}]
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "type": "tool_call",
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"location": "San Francisco"},
                }
            ],
        ),
        ToolMessage(
            content='{"temperature": 72, "conditions": "sunny"}',
            tool_call_id="call_123",
        ),
        AIMessage(content="The weather in San Francisco is 72°F and sunny."),
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "The weather in San Francisco is 72°F and sunny.",
                }
            ]
        ),
    ]
    messages_copy = [m.model_copy(deep=True) for m in messages]

    result = _construct_responses_api_input(messages)

    assert len(result) == len(messages)

    # Check system message
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."

    assert result[1]["role"] == "system"
    assert result[1]["content"] == [
        {"type": "input_text", "text": "You are a very helpful assistant!"}
    ]

    # Check human message
    assert result[2]["role"] == "user"
    assert result[2]["content"] == "What's the weather in San Francisco?"
    assert result[3]["role"] == "user"
    assert result[3]["content"] == [
        {"type": "input_text", "text": "What's the weather in San Francisco?"}
    ]

    # Check function call
    assert result[4]["type"] == "function_call"
    assert result[4]["name"] == "get_weather"
    assert result[4]["arguments"] == '{"location": "San Francisco"}'
    assert result[4]["call_id"] == "call_123"

    # Check function call output
    assert result[5]["type"] == "function_call_output"
    assert result[5]["output"] == '{"temperature": 72, "conditions": "sunny"}'
    assert result[5]["call_id"] == "call_123"

    assert result[6]["role"] == "assistant"
    assert result[6]["content"] == [
        {
            "type": "input_text",
            "text": "The weather in San Francisco is 72°F and sunny.",
        }
    ]

    assert result[7]["role"] == "assistant"
    assert result[7]["content"] == [
        {
            "type": "input_text",
            "text": "The weather in San Francisco is 72°F and sunny.",
            "annotations": [],
        }
    ]

    # assert no mutation has occurred
    assert messages_copy == messages

    # Test dict messages
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version="responses/v1"
    )
    message_dicts: list = [
        {"role": "developer", "content": "This is a developer message."},
        {
            "role": "developer",
            "content": [{"type": "text", "text": "This is a developer message!"}],
        },
    ]
    payload = llm._get_request_payload(message_dicts)
    result = payload["input"]
    assert len(result) == 2
    assert result[0]["role"] == "developer"
    assert result[0]["content"] == "This is a developer message."
    assert result[1]["role"] == "developer"
    assert result[1]["content"] == [
        {"type": "input_text", "text": "This is a developer message!"}
    ]


def test_service_tier() -> None:
    llm = ChatOpenAI(model="o4-mini", service_tier="flex")
    payload = llm._get_request_payload([HumanMessage("Hello")])
    assert payload["service_tier"] == "flex"


class FakeTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self.chat_model_start_inputs: list = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        pass

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        self.chat_model_start_inputs.append({"args": args, "kwargs": kwargs})
        return super().on_chat_model_start(*args, **kwargs)


def test_mcp_tracing() -> None:
    # Test we exclude sensitive information from traces
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version="responses/v1"
    )

    tracer = FakeTracer()
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> Response:
        return Response(
            id="resp_123",
            created_at=1234567890,
            model="o4-mini",
            object="response",
            parallel_tool_calls=True,
            tools=[],
            tool_choice="auto",
            output=[
                ResponseOutputMessage(
                    type="message",
                    id="msg_123",
                    content=[
                        ResponseOutputText(
                            type="output_text", text="Test response", annotations=[]
                        )
                    ],
                    role="assistant",
                    status="completed",
                )
            ],
        )

    mock_client.responses.create = mock_create
    input_message = HumanMessage("Test query")
    tools = [
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "always",
            "headers": {"Authorization": "Bearer PLACEHOLDER"},
        }
    ]
    with patch.object(llm, "root_client", mock_client):
        llm_with_tools = llm.bind_tools(tools)
        _ = llm_with_tools.invoke([input_message], config={"callbacks": [tracer]})

    # Test headers are not traced
    assert len(tracer.chat_model_start_inputs) == 1
    invocation_params = tracer.chat_model_start_inputs[0]["kwargs"]["invocation_params"]
    for tool in invocation_params["tools"]:
        if "headers" in tool:
            assert tool["headers"] == "**REDACTED**"
    for substring in ["Authorization", "Bearer", "PLACEHOLDER"]:
        assert substring not in str(tracer.chat_model_start_inputs)

    # Test headers are correctly propagated to request
    payload = llm_with_tools._get_request_payload([input_message], tools=tools)  # type: ignore[attr-defined]
    assert payload["tools"][0]["headers"]["Authorization"] == "Bearer PLACEHOLDER"


def test_compat() -> None:
    # Check compatibility with v0.3 message format
    message_v03 = AIMessage(
        content=[
            {"type": "text", "text": "Hello, world!", "annotations": [{"type": "foo"}]}
        ],
        additional_kwargs={
            "reasoning": {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            },
            "tool_outputs": [
                {
                    "type": "web_search_call",
                    "id": "websearch_123",
                    "status": "completed",
                }
            ],
            "refusal": "I cannot assist with that.",
        },
        response_metadata={"id": "resp_123"},
        id="msg_123",
    )

    message = _convert_from_v03_ai_message(message_v03)
    expected = AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
                "id": "rs_123",
            },
            {
                "type": "text",
                "text": "Hello, world!",
                "annotations": [{"type": "foo"}],
                "id": "msg_123",
            },
            {"type": "refusal", "refusal": "I cannot assist with that."},
            {"type": "web_search_call", "id": "websearch_123", "status": "completed"},
        ],
        response_metadata={"id": "resp_123"},
        id="resp_123",
    )
    assert message == expected

    ## Check no mutation
    assert message != message_v03
    assert len(message_v03.content) == 1
    assert all(
        item in message_v03.additional_kwargs
        for item in ["reasoning", "tool_outputs", "refusal"]
    )

    # Convert back
    message_v03_output = _convert_to_v03_ai_message(message)
    assert message_v03_output == message_v03
    assert message_v03_output is not message_v03


def test_get_last_messages() -> None:
    messages: list[BaseMessage] = [HumanMessage("Hello")]
    last_messages, previous_response_id = _get_last_messages(messages)
    assert last_messages == [HumanMessage("Hello")]
    assert previous_response_id is None

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("How are you?"),
    ]

    last_messages, previous_response_id = _get_last_messages(messages)
    assert last_messages == [HumanMessage("How are you?")]
    assert previous_response_id == "resp_123"

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("How are you?"),
        AIMessage("Well thanks.", response_metadata={"id": "resp_456"}),
        HumanMessage("Great."),
    ]
    last_messages, previous_response_id = _get_last_messages(messages)
    assert last_messages == [HumanMessage("Great.")]
    assert previous_response_id == "resp_456"

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("What's the weather?"),
        AIMessage(
            "",
            response_metadata={"id": "resp_456"},
            tool_calls=[
                {
                    "type": "tool_call",
                    "name": "get_weather",
                    "id": "call_123",
                    "args": {"location": "San Francisco"},
                }
            ],
        ),
        ToolMessage("It's sunny.", tool_call_id="call_123"),
    ]
    last_messages, previous_response_id = _get_last_messages(messages)
    assert last_messages == [ToolMessage("It's sunny.", tool_call_id="call_123")]
    assert previous_response_id == "resp_456"

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("How are you?"),
        AIMessage("Well thanks.", response_metadata={"id": "resp_456"}),
        HumanMessage("Good."),
        HumanMessage("Great."),
    ]
    last_messages, previous_response_id = _get_last_messages(messages)
    assert last_messages == [HumanMessage("Good."), HumanMessage("Great.")]
    assert previous_response_id == "resp_456"

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
    ]
    last_messages, response_id = _get_last_messages(messages)
    assert last_messages == []
    assert response_id == "resp_123"


def test_get_request_payload_use_previous_response_id() -> None:
    # Default - don't use previous_response ID
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version="responses/v1"
    )
    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("How are you?"),
    ]
    payload = llm._get_request_payload(messages)
    assert "previous_response_id" not in payload
    assert len(payload["input"]) == 3

    # Use previous response ID
    llm = ChatOpenAI(
        model="o4-mini",
        # Specifying use_previous_response_id automatically engages Responses API
        use_previous_response_id=True,
    )
    payload = llm._get_request_payload(messages)
    assert payload["previous_response_id"] == "resp_123"
    assert len(payload["input"]) == 1

    # Check single message
    messages = [HumanMessage("Hello")]
    payload = llm._get_request_payload(messages)
    assert "previous_response_id" not in payload
    assert len(payload["input"]) == 1


def test_make_computer_call_output_from_message() -> None:
    # List content
    tool_message = ToolMessage(
        content=[
            {"type": "input_image", "image_url": "data:image/png;base64,<image_data>"}
        ],
        tool_call_id="call_abc123",
        additional_kwargs={"type": "computer_call_output"},
    )
    result = _make_computer_call_output_from_message(tool_message)

    assert result == {
        "type": "computer_call_output",
        "call_id": "call_abc123",
        "output": {
            "type": "input_image",
            "image_url": "data:image/png;base64,<image_data>",
        },
    }

    # String content
    tool_message = ToolMessage(
        content="data:image/png;base64,<image_data>",
        tool_call_id="call_abc123",
        additional_kwargs={"type": "computer_call_output"},
    )
    result = _make_computer_call_output_from_message(tool_message)

    assert result == {
        "type": "computer_call_output",
        "call_id": "call_abc123",
        "output": {
            "type": "input_image",
            "image_url": "data:image/png;base64,<image_data>",
        },
    }

    # Safety checks
    tool_message = ToolMessage(
        content=[
            {"type": "input_image", "image_url": "data:image/png;base64,<image_data>"}
        ],
        tool_call_id="call_abc123",
        additional_kwargs={
            "type": "computer_call_output",
            "acknowledged_safety_checks": [
                {
                    "id": "cu_sc_abc234",
                    "code": "malicious_instructions",
                    "message": "Malicious instructions detected.",
                }
            ],
        },
    )
    result = _make_computer_call_output_from_message(tool_message)

    assert result == {
        "type": "computer_call_output",
        "call_id": "call_abc123",
        "output": {
            "type": "input_image",
            "image_url": "data:image/png;base64,<image_data>",
        },
        "acknowledged_safety_checks": [
            {
                "id": "cu_sc_abc234",
                "code": "malicious_instructions",
                "message": "Malicious instructions detected.",
            }
        ],
    }


def test_lc_tool_call_to_openai_tool_call_unicode() -> None:
    """Test that Unicode characters in tool call args are preserved correctly."""
    from langchain_openai.chat_models.base import _lc_tool_call_to_openai_tool_call

    tool_call = ToolCall(
        id="call_123",
        name="create_customer",
        args={"customer_name": "你好啊集团"},
        type="tool_call",
    )

    result = _lc_tool_call_to_openai_tool_call(tool_call)

    assert result["type"] == "function"
    assert result["id"] == "call_123"
    assert result["function"]["name"] == "create_customer"

    # Ensure Unicode characters are preserved, not escaped as \\uXXXX
    arguments_str = result["function"]["arguments"]
    parsed_args = json.loads(arguments_str)
    assert parsed_args["customer_name"] == "你好啊集团"
    # Also ensure the raw JSON string contains Unicode, not escaped sequences
    assert "你好啊集团" in arguments_str
    assert "\\u4f60" not in arguments_str  # Should not contain escaped Unicode


def test_extra_body_parameter() -> None:
    """Test that extra_body parameter is properly included in request payload."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr(
            "test-api-key"
        ),  # Set a fake API key to avoid validation error
        extra_body={"ttl": 300, "custom_param": "test_value"},
    )

    messages = [HumanMessage(content="Hello")]
    payload = llm._get_request_payload(messages)

    # Verify extra_body is included in the payload
    assert "extra_body" in payload
    assert payload["extra_body"]["ttl"] == 300
    assert payload["extra_body"]["custom_param"] == "test_value"


def test_extra_body_with_model_kwargs() -> None:
    """Test that extra_body and model_kwargs work together correctly."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr(
            "test-api-key"
        ),  # Set a fake API key to avoid validation error
        temperature=0.5,
        extra_body={"ttl": 600},
        model_kwargs={"custom_non_openai_param": "test_value"},
    )

    messages = [HumanMessage(content="Hello")]
    payload = llm._get_request_payload(messages)

    # Verify both extra_body and model_kwargs are in payload
    assert payload["extra_body"]["ttl"] == 600
    assert payload["custom_non_openai_param"] == "test_value"
    assert payload["temperature"] == 0.5


@pytest.mark.parametrize("use_responses_api", [False, True])
def test_gpt_5_temperature(use_responses_api: bool) -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano", temperature=0.5, use_responses_api=use_responses_api
    )

    messages = [HumanMessage(content="Hello")]
    payload = llm._get_request_payload(messages)
    assert "temperature" not in payload  # not supported for gpt-5 family models

    llm = ChatOpenAI(
        model="gpt-5-chat", temperature=0.5, use_responses_api=use_responses_api
    )
    messages = [HumanMessage(content="Hello")]
    payload = llm._get_request_payload(messages)
    assert payload["temperature"] == 0.5  # gpt-5-chat is exception
