"""Test Groq Chat API wrapper."""

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import langchain_core.load as lc_load
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)

from langchain_groq.chat_models import (
    ChatGroq,
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _create_usage_metadata,
)

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "fake-key"


def test_groq_model_param() -> None:
    llm = ChatGroq(model="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"
    llm = ChatGroq(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


def test_function_message_dict_to_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test__convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(
        content="foo", response_metadata={"model_provider": "groq"}
    )
    assert result == expected_output


def test__convert_dict_to_message_tool_call() -> None:
    raw_tool_call = {
        "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
        "function": {
            "arguments": '{"name":"Sally","hair_color":"green"}',
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
        response_metadata={"model_provider": "groq"},
    )
    assert result == expected_output

    # Test malformed tool call
    raw_tool_calls = [
        {
            "id": "call_wm0JY6CdwOMZ4eTxHWUThDNz",
            "function": {
                "arguments": "oops",
                "name": "GenerateUsername",
            },
            "type": "function",
        },
        {
            "id": "call_abc123",
            "function": {
                "arguments": '{"name":"Sally","hair_color":"green"}',
                "name": "GenerateUsername",
            },
            "type": "function",
        },
    ]
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
                error="Function GenerateUsername arguments:\n\noops\n\nare not valid JSON. Received JSONDecodeError Expecting value: line 1 column 1 (char 0)\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE ",  # noqa: E501
                type="invalid_tool_call",
            ),
        ],
        tool_calls=[
            ToolCall(
                name="GenerateUsername",
                args={"name": "Sally", "hair_color": "green"},
                id="call_abc123",
                type="tool_call",
            ),
        ],
        response_metadata={"model_provider": "groq"},
    )
    assert result == expected_output


def test__convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bar Baz",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_groq_invoke(mock_completion: dict) -> None:
    llm = ChatGroq(model="foo")
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


async def test_groq_ainvoke(mock_completion: dict) -> None:
    llm = ChatGroq(model="foo")
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("bar")
        assert res.content == "Bar Baz"
        assert type(res) is AIMessage
    assert completed


def test_chat_groq_extra_kwargs() -> None:
    """Test extra kwargs to chat groq."""
    # Check that foo is saved in extra_kwargs.
    with pytest.warns(UserWarning) as record:
        llm = ChatGroq(model="foo", foo=3, max_tokens=10)  # type: ignore[call-arg]
        assert llm.max_tokens == 10
        assert llm.model_kwargs == {"foo": 3}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if extra_kwargs are provided, they are added to it.
    with pytest.warns(UserWarning) as record:
        llm = ChatGroq(model="foo", foo=3, model_kwargs={"bar": 2})  # type: ignore[call-arg]
        assert llm.model_kwargs == {"foo": 3, "bar": 2}
    assert len(record) == 1
    assert type(record[0].message) is UserWarning
    assert "foo is not default parameter" in record[0].message.args[0]

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatGroq(model="foo", foo=3, model_kwargs={"foo": 2})  # type: ignore[call-arg]

    # Test that if explicit param is specified in kwargs it errors
    with pytest.raises(ValueError):
        ChatGroq(model="foo", model_kwargs={"temperature": 0.2})

    # Test that "model" cannot be specified in kwargs
    with pytest.raises(ValueError):
        ChatGroq(model="foo", model_kwargs={"model": "test-model"})


def test_chat_groq_invalid_streaming_params() -> None:
    """Test that an error is raised if streaming is invoked with n>1."""
    with pytest.raises(ValueError):
        ChatGroq(
            model="foo",
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


def test_chat_groq_secret() -> None:
    """Test that secret is not printed."""
    secret = "secretKey"  # noqa: S105
    not_secret = "safe"  # noqa: S105
    llm = ChatGroq(model="foo", api_key=secret, model_kwargs={"not_secret": not_secret})  # type: ignore[call-arg, arg-type]
    stringified = str(llm)
    assert not_secret in stringified
    assert secret not in stringified


@pytest.mark.filterwarnings("ignore:The function `loads` is in beta")
def test_groq_serialization() -> None:
    """Test that ChatGroq can be successfully serialized and deserialized."""
    api_key1 = "top secret"
    api_key2 = "topest secret"
    llm = ChatGroq(model="foo", api_key=api_key1, temperature=0.5)  # type: ignore[call-arg, arg-type]
    dump = lc_load.dumps(llm)
    llm2 = lc_load.loads(
        dump,
        valid_namespaces=["langchain_groq"],
        secrets_map={"GROQ_API_KEY": api_key2},
    )

    assert type(llm2) is ChatGroq

    # Ensure api key wasn't dumped and instead was read from secret map.
    assert llm.groq_api_key is not None
    assert llm.groq_api_key.get_secret_value() not in dump
    assert llm2.groq_api_key is not None
    assert llm2.groq_api_key.get_secret_value() == api_key2

    # Ensure a non-secret field was preserved
    assert llm.temperature == llm2.temperature

    # Ensure a None was preserved
    assert llm.groq_api_base == llm2.groq_api_base


def test_create_usage_metadata_basic() -> None:
    """Test basic usage metadata creation without details."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    result = _create_usage_metadata(token_usage)

    assert isinstance(result, dict)
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150
    assert "input_token_details" not in result
    assert "output_token_details" not in result


def test_create_usage_metadata_with_cached_tokens() -> None:
    """Test usage metadata with prompt caching."""
    token_usage = {
        "prompt_tokens": 2006,
        "completion_tokens": 300,
        "total_tokens": 2306,
        "prompt_tokens_details": {"cached_tokens": 1920},
    }

    result = _create_usage_metadata(token_usage)

    assert isinstance(result, dict)
    assert result["input_tokens"] == 2006
    assert result["output_tokens"] == 300
    assert result["total_tokens"] == 2306
    assert "input_token_details" in result
    assert isinstance(result["input_token_details"], dict)
    assert result["input_token_details"]["cache_read"] == 1920
    assert "output_token_details" not in result


def test_create_usage_metadata_with_all_details() -> None:
    """Test usage metadata with all available details."""
    token_usage = {
        "prompt_tokens": 2006,
        "completion_tokens": 450,
        "total_tokens": 2456,
        "prompt_tokens_details": {"cached_tokens": 1920},
        "completion_tokens_details": {"reasoning_tokens": 200},
    }

    result = _create_usage_metadata(token_usage)

    assert isinstance(result, dict)
    assert result["input_tokens"] == 2006
    assert result["output_tokens"] == 450
    assert result["total_tokens"] == 2456

    assert "input_token_details" in result
    assert isinstance(result["input_token_details"], dict)
    assert result["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" in result
    assert isinstance(result["output_token_details"], dict)
    assert result["output_token_details"]["reasoning"] == 200


def test_create_usage_metadata_missing_total_tokens() -> None:
    """Test that total_tokens is calculated when missing."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }

    result = _create_usage_metadata(token_usage)

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150


def test_create_usage_metadata_empty_details() -> None:
    """Test that empty detail dicts don't create token detail objects."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {},
    }

    result = _create_usage_metadata(token_usage)

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150
    assert "input_token_details" not in result
    assert "output_token_details" not in result


def test_create_usage_metadata_zero_cached_tokens() -> None:
    """Test that zero cached tokens are not included (falsy)."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 0},
    }

    result = _create_usage_metadata(token_usage)

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150
    assert "input_token_details" not in result


def test_create_usage_metadata_with_reasoning_tokens() -> None:
    """Test usage metadata with reasoning tokens."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 450,
        "total_tokens": 550,
        "completion_tokens_details": {"reasoning_tokens": 200},
    }

    result = _create_usage_metadata(token_usage)

    assert isinstance(result, dict)
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 450
    assert result["total_tokens"] == 550
    assert "output_token_details" in result
    assert isinstance(result["output_token_details"], dict)
    assert result["output_token_details"]["reasoning"] == 200
    assert "input_token_details" not in result


def test_create_usage_metadata_with_cached_and_reasoning_tokens() -> None:
    """Test usage metadata with both cached and reasoning tokens."""
    token_usage = {
        "prompt_tokens": 2006,
        "completion_tokens": 450,
        "total_tokens": 2456,
        "prompt_tokens_details": {"cached_tokens": 1920},
        "completion_tokens_details": {"reasoning_tokens": 200},
    }

    result = _create_usage_metadata(token_usage)

    assert isinstance(result, dict)
    assert result["input_tokens"] == 2006
    assert result["output_tokens"] == 450
    assert result["total_tokens"] == 2456

    assert "input_token_details" in result
    assert isinstance(result["input_token_details"], dict)
    assert result["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" in result
    assert isinstance(result["output_token_details"], dict)
    assert result["output_token_details"]["reasoning"] == 200


def test_create_usage_metadata_zero_reasoning_tokens() -> None:
    """Test that zero reasoning tokens are not included (falsy)."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }

    result = _create_usage_metadata(token_usage)

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150
    assert "output_token_details" not in result


def test_create_usage_metadata_empty_completion_details() -> None:
    """Test that empty completion_tokens_details don't create output_token_details."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": {},
    }

    result = _create_usage_metadata(token_usage)

    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["total_tokens"] == 150
    assert "output_token_details" not in result


def test_chat_result_with_usage_metadata() -> None:
    """Test that _create_chat_result properly includes usage metadata."""
    llm = ChatGroq(model="test-model")

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2006,
            "completion_tokens": 300,
            "total_tokens": 2306,
            "prompt_tokens_details": {"cached_tokens": 1920},
        },
    }

    result = llm._create_chat_result(mock_response, {})

    assert len(result.generations) == 1
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.content == "Test response"

    assert message.usage_metadata is not None
    assert isinstance(message.usage_metadata, dict)
    assert message.usage_metadata["input_tokens"] == 2006
    assert message.usage_metadata["output_tokens"] == 300
    assert message.usage_metadata["total_tokens"] == 2306

    assert "input_token_details" in message.usage_metadata
    assert message.usage_metadata["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" not in message.usage_metadata


def test_chat_result_with_reasoning_tokens() -> None:
    """Test that _create_chat_result properly includes reasoning tokens."""
    llm = ChatGroq(model="test-model")

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test reasoning response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 450,
            "total_tokens": 550,
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
    }

    result = llm._create_chat_result(mock_response, {})

    assert len(result.generations) == 1
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.content == "Test reasoning response"

    assert message.usage_metadata is not None
    assert isinstance(message.usage_metadata, dict)
    assert message.usage_metadata["input_tokens"] == 100
    assert message.usage_metadata["output_tokens"] == 450
    assert message.usage_metadata["total_tokens"] == 550

    assert "output_token_details" in message.usage_metadata
    assert message.usage_metadata["output_token_details"]["reasoning"] == 200

    assert "input_token_details" not in message.usage_metadata


def test_chat_result_with_cached_and_reasoning_tokens() -> None:
    """Test that _create_chat_result includes both cached and reasoning tokens."""
    llm = ChatGroq(model="test-model")

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response with both",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2006,
            "completion_tokens": 450,
            "total_tokens": 2456,
            "prompt_tokens_details": {"cached_tokens": 1920},
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
    }

    result = llm._create_chat_result(mock_response, {})

    assert len(result.generations) == 1
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.content == "Test response with both"

    assert message.usage_metadata is not None
    assert isinstance(message.usage_metadata, dict)
    assert message.usage_metadata["input_tokens"] == 2006
    assert message.usage_metadata["output_tokens"] == 450
    assert message.usage_metadata["total_tokens"] == 2456

    assert "input_token_details" in message.usage_metadata
    assert message.usage_metadata["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" in message.usage_metadata
    assert message.usage_metadata["output_token_details"]["reasoning"] == 200


def test_chat_result_backward_compatibility() -> None:
    """Test that responses without new fields still work."""
    llm = ChatGroq(model="test-model")

    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }

    result = llm._create_chat_result(mock_response, {})

    assert len(result.generations) == 1
    message = result.generations[0].message
    assert isinstance(message, AIMessage)

    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 100
    assert message.usage_metadata["output_tokens"] == 50
    assert message.usage_metadata["total_tokens"] == 150

    assert "input_token_details" not in message.usage_metadata
    assert "output_token_details" not in message.usage_metadata


def test_streaming_with_usage_metadata() -> None:
    """Test that streaming properly includes usage metadata."""
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
        "x_groq": {
            "usage": {
                "prompt_tokens": 2006,
                "completion_tokens": 300,
                "total_tokens": 2306,
                "prompt_tokens_details": {"cached_tokens": 1920},
            }
        },
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"

    assert result.usage_metadata is not None
    assert isinstance(result.usage_metadata, dict)
    assert result.usage_metadata["input_tokens"] == 2006
    assert result.usage_metadata["output_tokens"] == 300
    assert result.usage_metadata["total_tokens"] == 2306

    assert "input_token_details" in result.usage_metadata
    assert result.usage_metadata["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" not in result.usage_metadata


def test_streaming_with_reasoning_tokens() -> None:
    """Test that streaming properly includes reasoning tokens in usage metadata."""
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
        "x_groq": {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 450,
                "total_tokens": 550,
                "completion_tokens_details": {"reasoning_tokens": 200},
            }
        },
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"

    assert result.usage_metadata is not None
    assert isinstance(result.usage_metadata, dict)
    assert result.usage_metadata["input_tokens"] == 100
    assert result.usage_metadata["output_tokens"] == 450
    assert result.usage_metadata["total_tokens"] == 550

    assert "output_token_details" in result.usage_metadata
    assert result.usage_metadata["output_token_details"]["reasoning"] == 200

    assert "input_token_details" not in result.usage_metadata


def test_streaming_with_cached_and_reasoning_tokens() -> None:
    """Test that streaming includes both cached and reasoning tokens."""
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
        "x_groq": {
            "usage": {
                "prompt_tokens": 2006,
                "completion_tokens": 450,
                "total_tokens": 2456,
                "prompt_tokens_details": {"cached_tokens": 1920},
                "completion_tokens_details": {"reasoning_tokens": 200},
            }
        },
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"

    assert result.usage_metadata is not None
    assert isinstance(result.usage_metadata, dict)
    assert result.usage_metadata["input_tokens"] == 2006
    assert result.usage_metadata["output_tokens"] == 450
    assert result.usage_metadata["total_tokens"] == 2456

    assert "input_token_details" in result.usage_metadata
    assert result.usage_metadata["input_token_details"]["cache_read"] == 1920

    assert "output_token_details" in result.usage_metadata
    assert result.usage_metadata["output_token_details"]["reasoning"] == 200


def test_streaming_without_usage_metadata() -> None:
    """Test that streaming works without usage metadata (backward compatibility)."""
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert isinstance(result, AIMessageChunk)
    assert result.content == "Hello"
    assert result.usage_metadata is None


def test_combine_llm_outputs_with_token_details() -> None:
    """Test that _combine_llm_outputs properly combines nested token details."""
    llm = ChatGroq(model="test-model")

    llm_outputs: list[dict[str, Any] | None] = [
        {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
            "model_name": "test-model",
            "system_fingerprint": "fp_123",
        },
        {
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 150},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            "model_name": "test-model",
            "system_fingerprint": "fp_123",
        },
    ]

    result = llm._combine_llm_outputs(llm_outputs)

    assert result["token_usage"]["prompt_tokens"] == 300
    assert result["token_usage"]["completion_tokens"] == 150
    assert result["token_usage"]["total_tokens"] == 450
    assert result["token_usage"]["prompt_tokens_details"]["cached_tokens"] == 230
    assert result["token_usage"]["completion_tokens_details"]["reasoning_tokens"] == 60
    assert result["model_name"] == "test-model"
    assert result["system_fingerprint"] == "fp_123"


def test_combine_llm_outputs_with_missing_details() -> None:
    """Test _combine_llm_outputs when some outputs have details and others don't."""
    llm = ChatGroq(model="test-model")

    llm_outputs: list[dict[str, Any] | None] = [
        {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "model_name": "test-model",
        },
        {
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            "model_name": "test-model",
        },
    ]

    result = llm._combine_llm_outputs(llm_outputs)

    assert result["token_usage"]["prompt_tokens"] == 300
    assert result["token_usage"]["completion_tokens"] == 150
    assert result["token_usage"]["total_tokens"] == 450
    assert result["token_usage"]["completion_tokens_details"]["reasoning_tokens"] == 40
    assert "prompt_tokens_details" not in result["token_usage"]
