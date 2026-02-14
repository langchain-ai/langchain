"""Unit tests for `ChatOpenRouter` chat model."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.load import dumpd, dumps, load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding
from pydantic import BaseModel, Field, SecretStr

from langchain_openrouter.chat_models import (
    ChatOpenRouter,
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
    _convert_video_block_to_openrouter,
    _create_usage_metadata,
    _format_message_content,
)

MODEL_NAME = "openai/gpt-4o-mini"


def _make_model(**kwargs: Any) -> ChatOpenRouter:
    """Create a `ChatOpenRouter` with sane defaults for unit tests."""
    defaults: dict[str, Any] = {"model": MODEL_NAME, "api_key": SecretStr("test-key")}
    defaults.update(kwargs)
    return ChatOpenRouter(**defaults)


# ---------------------------------------------------------------------------
# Pydantic schemas used across multiple test classes
# ---------------------------------------------------------------------------


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(description="The city and state")


class GenerateUsername(BaseModel):
    """Generate a username from a full name."""

    name: str = Field(description="The full name")
    hair_color: str = Field(description="The hair color")


# ---------------------------------------------------------------------------
# Mock helpers for SDK responses
# ---------------------------------------------------------------------------

_SIMPLE_RESPONSE_DICT: dict[str, Any] = {
    "id": "gen-abc123",
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
            "index": 0,
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
    "model": MODEL_NAME,
    "object": "chat.completion",
    "created": 1700000000.0,
}

_TOOL_RESPONSE_DICT: dict[str, Any] = {
    "id": "gen-tool123",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "GetWeather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "index": 0,
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    "model": MODEL_NAME,
    "object": "chat.completion",
    "created": 1700000000.0,
}

_STREAM_CHUNKS: list[dict[str, Any]] = [
    {
        "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}],
        "model": MODEL_NAME,
        "object": "chat.completion.chunk",
        "created": 1700000000.0,
        "id": "gen-stream1",
    },
    {
        "choices": [{"delta": {"content": "Hello"}, "index": 0}],
        "model": MODEL_NAME,
        "object": "chat.completion.chunk",
        "created": 1700000000.0,
        "id": "gen-stream1",
    },
    {
        "choices": [{"delta": {"content": " world"}, "index": 0}],
        "model": MODEL_NAME,
        "object": "chat.completion.chunk",
        "created": 1700000000.0,
        "id": "gen-stream1",
    },
    {
        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "model": MODEL_NAME,
        "object": "chat.completion.chunk",
        "created": 1700000000.0,
        "id": "gen-stream1",
    },
]


def _make_sdk_response(response_dict: dict[str, Any]) -> MagicMock:
    """Build a MagicMock that behaves like an SDK ChatResponse."""
    mock = MagicMock()
    mock.model_dump.return_value = response_dict
    return mock


class _MockSyncStream:
    """Synchronous iterator that mimics the SDK EventStream."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    def __iter__(self) -> _MockSyncStream:
        return self

    def __next__(self) -> MagicMock:
        if not self._chunks:
            raise StopIteration
        chunk = self._chunks.pop(0)
        mock = MagicMock()
        mock.model_dump.return_value = chunk
        return mock


class _MockAsyncStream:
    """Async iterator that mimics the SDK EventStreamAsync."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> _MockAsyncStream:
        return self

    async def __anext__(self) -> MagicMock:
        if not self._chunks:
            raise StopAsyncIteration
        chunk = self._chunks.pop(0)
        mock = MagicMock()
        mock.model_dump.return_value = chunk
        return mock


# ===========================================================================
# Instantiation tests
# ===========================================================================


class TestChatOpenRouterInstantiation:
    """Tests for `ChatOpenRouter` instantiation."""

    def test_basic_instantiation(self) -> None:
        """Test basic model instantiation with required params."""
        model = _make_model()
        assert model.model_name == MODEL_NAME
        assert model.openrouter_api_base is None

    def test_api_key_from_field(self) -> None:
        """Test that API key is properly set."""
        model = _make_model()
        assert model.openrouter_api_key is not None
        assert model.openrouter_api_key.get_secret_value() == "test-key"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API key is read from OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-123")
        model = ChatOpenRouter(model=MODEL_NAME)
        assert model.openrouter_api_key is not None
        assert model.openrouter_api_key.get_secret_value() == "env-key-123"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY must be set"):
            ChatOpenRouter(model=MODEL_NAME)

    def test_model_required(self) -> None:
        """Test that model name is required."""
        with pytest.raises((ValueError, TypeError)):
            ChatOpenRouter(api_key=SecretStr("test-key"))  # type: ignore[call-arg]

    def test_secret_masking(self) -> None:
        """Test that API key is not exposed in string representation."""
        model = _make_model(api_key=SecretStr("super-secret"))
        model_str = str(model)
        assert "super-secret" not in model_str

    def test_secret_masking_repr(self) -> None:
        """Test that API key is masked in repr too."""
        model = _make_model(api_key=SecretStr("super-secret"))
        assert "super-secret" not in repr(model)

    def test_api_key_is_secret_str(self) -> None:
        """Test that openrouter_api_key is a SecretStr instance."""
        model = _make_model()
        assert isinstance(model.openrouter_api_key, SecretStr)

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        model = _make_model()
        assert model._llm_type == "openrouter-chat"

    def test_ls_params(self) -> None:
        """Test LangSmith params include openrouter provider."""
        model = _make_model()
        ls_params = model._get_ls_params()
        assert ls_params["ls_provider"] == "openrouter"

    def test_client_created(self) -> None:
        """Test that OpenRouter SDK client is created."""
        model = _make_model()
        assert model.client is not None

    def test_client_reused_for_same_params(self) -> None:
        """Test that the SDK client is reused when model is re-validated."""
        model = _make_model()
        client_1 = model.client
        # Re-validate does not replace the existing client
        model.validate_environment()  # type: ignore[operator]
        assert model.client is client_1

    def test_app_url_passed_to_client(self) -> None:
        """Test that app_url is passed as http_referer to the SDK client."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                app_url="https://myapp.com",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["http_referer"] == "https://myapp.com"

    def test_app_title_passed_to_client(self) -> None:
        """Test that app_title is passed as x_title to the SDK client."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                app_title="My App",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["x_title"] == "My App"

    def test_reasoning_in_params(self) -> None:
        """Test that `reasoning` is included in default params."""
        model = _make_model(reasoning={"effort": "high"})
        params = model._default_params
        assert params["reasoning"] == {"effort": "high"}

    def test_openrouter_provider_in_params(self) -> None:
        """Test that `openrouter_provider` is included in default params."""
        model = _make_model(openrouter_provider={"order": ["Anthropic"]})
        params = model._default_params
        assert params["provider"] == {"order": ["Anthropic"]}

    def test_route_in_params(self) -> None:
        """Test that `route` is included in default params."""
        model = _make_model(route="fallback")
        params = model._default_params
        assert params["route"] == "fallback"

    def test_optional_params_excluded_when_none(self) -> None:
        """Test that None optional params are not in default params."""
        model = _make_model()
        params = model._default_params
        assert "temperature" not in params
        assert "max_tokens" not in params
        assert "top_p" not in params
        assert "reasoning" not in params

    def test_temperature_included_when_set(self) -> None:
        """Test that temperature is included when explicitly set."""
        model = _make_model(temperature=0.5)
        params = model._default_params
        assert params["temperature"] == 0.5


# ===========================================================================
# Serialization tests
# ===========================================================================


class TestSerialization:
    """Tests for serialization round-trips."""

    def test_is_lc_serializable(self) -> None:
        """Test that ChatOpenRouter declares itself as serializable."""
        assert ChatOpenRouter.is_lc_serializable() is True

    def test_dumpd_load_roundtrip(self) -> None:
        """Test that dumpd/load round-trip preserves model config."""
        model = _make_model(temperature=0.7, max_tokens=100)
        serialized = dumpd(model)
        deserialized = load(
            serialized,
            valid_namespaces=["langchain_openrouter"],
            allowed_objects="all",
            secrets_from_env=False,
            secrets_map={"OPENROUTER_API_KEY": "test-key"},
        )
        assert isinstance(deserialized, ChatOpenRouter)
        assert deserialized.model_name == MODEL_NAME
        assert deserialized.temperature == 0.7
        assert deserialized.max_tokens == 100

    def test_dumps_does_not_leak_secrets(self) -> None:
        """Test that dumps output does not contain the raw API key."""
        model = _make_model(api_key=SecretStr("super-secret-key"))
        serialized = dumps(model)
        assert "super-secret-key" not in serialized


# ===========================================================================
# Mocked generate / stream tests
# ===========================================================================


class TestMockedGenerate:
    """Tests for _generate / _agenerate with a mocked SDK client."""

    def test_invoke_basic(self) -> None:
        """Test basic invoke returns an AIMessage via mocked SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        result = model.invoke("Hello")
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"
        model.client.chat.send.assert_called_once()

    def test_invoke_with_tool_response(self) -> None:
        """Test invoke that returns tool calls."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_TOOL_RESPONSE_DICT)

        result = model.invoke("What's the weather?")
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "GetWeather"

    def test_invoke_passes_correct_messages(self) -> None:
        """Test that invoke converts messages and passes them to the SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke([HumanMessage(content="Hi")])
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    def test_invoke_strips_internal_kwargs(self) -> None:
        """Test that LangChain-internal kwargs are stripped before SDK call."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model._generate(
            [HumanMessage(content="Hi")],
            ls_structured_output_format={"kwargs": {"method": "function_calling"}},
        )
        call_kwargs = model.client.chat.send.call_args[1]
        assert "ls_structured_output_format" not in call_kwargs

    def test_invoke_usage_metadata(self) -> None:
        """Test that usage metadata is populated on the response."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        result = model.invoke("Hello")
        assert isinstance(result, AIMessage)
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 10
        assert result.usage_metadata["output_tokens"] == 5
        assert result.usage_metadata["total_tokens"] == 15

    def test_stream_basic(self) -> None:
        """Test streaming returns AIMessageChunks via mocked SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _MockSyncStream(
            [dict(c) for c in _STREAM_CHUNKS]
        )

        chunks = list(model.stream("Hello"))
        assert len(chunks) > 0
        assert all(isinstance(c, AIMessageChunk) for c in chunks)
        # Concatenated content should be "Hello world"
        full_content = "".join(c.content for c in chunks if isinstance(c.content, str))
        assert "Hello" in full_content
        assert "world" in full_content

    def test_stream_passes_stream_true(self) -> None:
        """Test that stream sends stream=True to the SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _MockSyncStream(
            [dict(c) for c in _STREAM_CHUNKS]
        )

        list(model.stream("Hello"))
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["stream"] is True

    def test_invoke_with_streaming_flag(self) -> None:
        """Test that invoke delegates to stream when streaming=True."""
        model = _make_model(streaming=True)
        model.client = MagicMock()
        model.client.chat.send.return_value = _MockSyncStream(
            [dict(c) for c in _STREAM_CHUNKS]
        )

        result = model.invoke("Hello")
        assert isinstance(result, AIMessage)
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["stream"] is True

    async def test_ainvoke_basic(self) -> None:
        """Test async invoke returns an AIMessage via mocked SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send_async = AsyncMock(
            return_value=_make_sdk_response(_SIMPLE_RESPONSE_DICT)
        )

        result = await model.ainvoke("Hello")
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"
        model.client.chat.send_async.assert_awaited_once()

    async def test_astream_basic(self) -> None:
        """Test async streaming returns AIMessageChunks via mocked SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send_async = AsyncMock(
            return_value=_MockAsyncStream(_STREAM_CHUNKS)
        )

        chunks = [c async for c in model.astream("Hello")]
        assert len(chunks) > 0
        assert all(isinstance(c, AIMessageChunk) for c in chunks)

    def test_stream_response_metadata_fields(self) -> None:
        """Test response-level metadata in streaming response_metadata."""
        model = _make_model()
        model.client = MagicMock()
        stream_chunks: list[dict[str, Any]] = [
            {
                "choices": [
                    {"delta": {"role": "assistant", "content": "Hi"}, "index": 0}
                ],
                "model": "anthropic/claude-sonnet-4-5",
                "system_fingerprint": "fp_stream123",
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-stream-meta",
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "native_finish_reason": "end_turn",
                        "index": 0,
                    }
                ],
                "model": "anthropic/claude-sonnet-4-5",
                "system_fingerprint": "fp_stream123",
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-stream-meta",
            },
        ]
        model.client.chat.send.return_value = _MockSyncStream(stream_chunks)

        chunks = list(model.stream("Hello"))
        assert len(chunks) >= 2

        # Find the chunk with finish_reason (final metadata chunk)
        final = [
            c for c in chunks if c.response_metadata.get("finish_reason") == "stop"
        ]
        assert len(final) == 1
        meta = final[0].response_metadata
        assert meta["model"] == "anthropic/claude-sonnet-4-5"
        assert meta["system_fingerprint"] == "fp_stream123"
        assert meta["native_finish_reason"] == "end_turn"
        assert meta["finish_reason"] == "stop"
        assert meta["id"] == "gen-stream-meta"
        assert meta["created"] == 1700000000.0
        assert meta["object"] == "chat.completion.chunk"

    async def test_astream_response_metadata_fields(self) -> None:
        """Test response-level metadata in async streaming response_metadata."""
        model = _make_model()
        model.client = MagicMock()
        stream_chunks: list[dict[str, Any]] = [
            {
                "choices": [
                    {"delta": {"role": "assistant", "content": "Hi"}, "index": 0}
                ],
                "model": "anthropic/claude-sonnet-4-5",
                "system_fingerprint": "fp_async123",
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-astream-meta",
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "native_finish_reason": "end_turn",
                        "index": 0,
                    }
                ],
                "model": "anthropic/claude-sonnet-4-5",
                "system_fingerprint": "fp_async123",
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-astream-meta",
            },
        ]
        model.client.chat.send_async = AsyncMock(
            return_value=_MockAsyncStream(stream_chunks)
        )

        chunks = [c async for c in model.astream("Hello")]
        assert len(chunks) >= 2

        # Find the chunk with finish_reason (final metadata chunk)
        final = [
            c for c in chunks if c.response_metadata.get("finish_reason") == "stop"
        ]
        assert len(final) == 1
        meta = final[0].response_metadata
        assert meta["model"] == "anthropic/claude-sonnet-4-5"
        assert meta["system_fingerprint"] == "fp_async123"
        assert meta["native_finish_reason"] == "end_turn"
        assert meta["id"] == "gen-astream-meta"
        assert meta["created"] == 1700000000.0
        assert meta["object"] == "chat.completion.chunk"


# ===========================================================================
# Request payload verification
# ===========================================================================


class TestRequestPayload:
    """Tests verifying the exact dict sent to the SDK."""

    def test_message_format_in_payload(self) -> None:
        """Test that messages are formatted correctly in the SDK call."""
        model = _make_model(temperature=0)
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke(
            [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Hi"),
            ]
        )
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

    def test_model_kwargs_forwarded(self) -> None:
        """Test that extra model_kwargs are included in the SDK call."""
        model = _make_model(model_kwargs={"top_k": 50})
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke("Hi")
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["top_k"] == 50

    def test_stop_sequences_in_payload(self) -> None:
        """Test that stop sequences are passed to the SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke("Hi", stop=["END"])
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["stop"] == ["END"]

    def test_tool_format_in_payload(self) -> None:
        """Test that tools are formatted in OpenAI-compatible structure."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_TOOL_RESPONSE_DICT)

        bound = model.bind_tools([GetWeather])
        bound.invoke("What's the weather?")
        call_kwargs = model.client.chat.send.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "GetWeather"
        assert "parameters" in tools[0]["function"]

    def test_openrouter_params_in_payload(self) -> None:
        """Test that OpenRouter-specific params appear in the SDK call."""
        model = _make_model(
            reasoning={"effort": "high"},
            openrouter_provider={"order": ["Anthropic"]},
            route="fallback",
        )
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke("Hi")
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high"}
        assert call_kwargs["provider"] == {"order": ["Anthropic"]}
        assert call_kwargs["route"] == "fallback"


# ===========================================================================
# bind_tools tests
# ===========================================================================


class TestBindTools:
    """Tests for the bind_tools public method."""

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "auto",
            "none",
            "required",
            "GetWeather",
            {"type": "function", "function": {"name": "GetWeather"}},
            None,
        ],
    )
    def test_bind_tools_tool_choice(self, tool_choice: Any) -> None:
        """Test bind_tools accepts various tool_choice values."""
        model = _make_model()
        bound = model.bind_tools(
            [GetWeather, GenerateUsername], tool_choice=tool_choice
        )
        assert isinstance(bound, RunnableBinding)

    def test_bind_tools_bool_true_single_tool(self) -> None:
        """Test bind_tools with tool_choice=True and a single tool."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], tool_choice=True)
        assert isinstance(bound, RunnableBinding)
        kwargs = bound.kwargs
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "GetWeather"},
        }

    def test_bind_tools_bool_true_multiple_tools_raises(self) -> None:
        """Test bind_tools with tool_choice=True and multiple tools raises."""
        model = _make_model()
        with pytest.raises(ValueError, match="tool_choice can only be True"):
            model.bind_tools([GetWeather, GenerateUsername], tool_choice=True)

    def test_bind_tools_any_maps_to_required(self) -> None:
        """Test that tool_choice='any' is mapped to 'required'."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], tool_choice="any")
        assert isinstance(bound, RunnableBinding)
        assert bound.kwargs["tool_choice"] == "required"

    def test_bind_tools_string_name_becomes_dict(self) -> None:
        """Test that a specific tool name string is converted to a dict."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], tool_choice="GetWeather")
        assert isinstance(bound, RunnableBinding)
        assert bound.kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "GetWeather"},
        }

    def test_bind_tools_formats_tools_correctly(self) -> None:
        """Test that tools are converted to OpenAI format."""
        model = _make_model()
        bound = model.bind_tools([GetWeather])
        assert isinstance(bound, RunnableBinding)
        tools = bound.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "GetWeather"

    def test_bind_tools_no_choice_omits_key(self) -> None:
        """Test that tool_choice=None does not set tool_choice in kwargs."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], tool_choice=None)
        assert isinstance(bound, RunnableBinding)
        assert "tool_choice" not in bound.kwargs

    def test_bind_tools_strict_forwarded(self) -> None:
        """Test that strict param is forwarded to tool definitions."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], strict=True)
        assert isinstance(bound, RunnableBinding)
        tools = bound.kwargs["tools"]
        assert tools[0]["function"]["strict"] is True

    def test_bind_tools_strict_none_by_default(self) -> None:
        """Test that strict is not set when not provided."""
        model = _make_model()
        bound = model.bind_tools([GetWeather])
        assert isinstance(bound, RunnableBinding)
        tools = bound.kwargs["tools"]
        assert "strict" not in tools[0]["function"]

    def test_bind_tools_parallel_tool_calls_false(self) -> None:
        """Test that parallel_tool_calls=False is forwarded."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], parallel_tool_calls=False)
        assert isinstance(bound, RunnableBinding)
        assert bound.kwargs["parallel_tool_calls"] is False

    def test_bind_tools_parallel_tool_calls_not_set_by_default(self) -> None:
        """Test that parallel_tool_calls is not set when not provided."""
        model = _make_model()
        bound = model.bind_tools([GetWeather])
        assert isinstance(bound, RunnableBinding)
        assert "parallel_tool_calls" not in bound.kwargs


# ===========================================================================
# with_structured_output tests
# ===========================================================================


class TestWithStructuredOutput:
    """Tests for the with_structured_output public method."""

    @pytest.mark.parametrize("method", ["function_calling", "json_schema"])
    @pytest.mark.parametrize("include_raw", ["yes", "no"])
    def test_with_structured_output_pydantic(
        self,
        method: Literal["function_calling", "json_schema"],
        include_raw: str,
    ) -> None:
        """Test with_structured_output using a Pydantic schema."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method=method, include_raw=(include_raw == "yes")
        )
        assert structured is not None

    @pytest.mark.parametrize("method", ["function_calling", "json_schema"])
    def test_with_structured_output_dict_schema(
        self,
        method: Literal["function_calling", "json_schema"],
    ) -> None:
        """Test with_structured_output using a JSON schema dict."""
        schema = GenerateUsername.model_json_schema()
        model = _make_model()
        structured = model.with_structured_output(schema, method=method)
        assert structured is not None

    def test_with_structured_output_none_schema_function_calling_raises(self) -> None:
        """Test that schema=None with function_calling raises ValueError."""
        model = _make_model()
        with pytest.raises(ValueError, match="schema must be specified"):
            model.with_structured_output(None, method="function_calling")

    def test_with_structured_output_none_schema_json_schema_raises(self) -> None:
        """Test that schema=None with json_schema raises ValueError."""
        model = _make_model()
        with pytest.raises(ValueError, match="schema must be specified"):
            model.with_structured_output(None, method="json_schema")

    def test_with_structured_output_invalid_method_raises(self) -> None:
        """Test that an unrecognized method raises ValueError."""
        model = _make_model()
        with pytest.raises(ValueError, match="Unrecognized method"):
            model.with_structured_output(
                GenerateUsername,
                method="invalid",  # type: ignore[arg-type]
            )

    def test_with_structured_output_json_schema_sets_response_format(self) -> None:
        """Test that json_schema method sets response_format correctly."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method="json_schema"
        )
        # The first step in the chain should be the bound model
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "GenerateUsername"

    def test_with_structured_output_json_mode_warns_and_falls_back(self) -> None:
        """Test that json_mode warns and falls back to json_schema."""
        model = _make_model()
        with pytest.warns(match="Defaulting to 'json_schema'"):
            structured = model.with_structured_output(
                GenerateUsername, method="json_mode"  # type: ignore[arg-type]
            )
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["type"] == "json_schema"

    def test_with_structured_output_strict_function_calling(self) -> None:
        """Test that strict is forwarded for function_calling method."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method="function_calling", strict=True
        )
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        tools = bound.kwargs["tools"]
        assert tools[0]["function"]["strict"] is True

    def test_with_structured_output_strict_json_schema(self) -> None:
        """Test that strict is forwarded for json_schema method."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method="json_schema", strict=True
        )
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["json_schema"]["strict"] is True

    def test_with_structured_output_json_mode_with_strict_warns_and_forwards(
        self,
    ) -> None:
        """Test that json_mode with strict warns, falls back to json_schema, and
        forwards strict."""
        model = _make_model()
        with pytest.warns(match="Defaulting to 'json_schema'"):
            structured = model.with_structured_output(
                GenerateUsername,
                method="json_mode",  # type: ignore[arg-type]
                strict=True,
            )
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True


# ===========================================================================
# Message conversion tests
# ===========================================================================


class TestMessageConversion:
    """Tests for message conversion functions."""

    def test_human_message_to_dict(self) -> None:
        """Test converting HumanMessage to dict."""
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_system_message_to_dict(self) -> None:
        """Test converting SystemMessage to dict."""
        msg = SystemMessage(content="You are helpful.")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "system", "content": "You are helpful."}

    def test_ai_message_to_dict(self) -> None:
        """Test converting AIMessage to dict."""
        msg = AIMessage(content="Hi there!")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "assistant", "content": "Hi there!"}

    def test_ai_message_with_reasoning_content_to_dict(self) -> None:
        """Test that reasoning_content is preserved when converting back to dict."""
        msg = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Let me think about this..."},
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "The answer is 42."
        assert result["reasoning"] == "Let me think about this..."

    def test_ai_message_with_reasoning_details_to_dict(self) -> None:
        """Test that reasoning_details is preserved when converting back to dict."""
        details = [
            {"type": "reasoning.text", "text": "Step 1: analyze"},
            {"type": "reasoning.text", "text": "Step 2: solve"},
        ]
        msg = AIMessage(
            content="Answer",
            additional_kwargs={"reasoning_details": details},
        )
        result = _convert_message_to_dict(msg)
        assert result["reasoning_details"] == details
        assert "reasoning" not in result

    def test_ai_message_with_both_reasoning_fields_to_dict(self) -> None:
        """Test that both reasoning_content and reasoning_details are preserved."""
        details = [{"type": "reasoning.text", "text": "detailed thinking"}]
        msg = AIMessage(
            content="Answer",
            additional_kwargs={
                "reasoning_content": "I thought about it",
                "reasoning_details": details,
            },
        )
        result = _convert_message_to_dict(msg)
        assert result["reasoning"] == "I thought about it"
        assert result["reasoning_details"] == details

    def test_reasoning_roundtrip_through_dict(self) -> None:
        """Test that reasoning survives dict -> message -> dict roundtrip."""
        original_dict = {
            "role": "assistant",
            "content": "The answer",
            "reasoning": "My thinking process",
            "reasoning_details": [{"type": "reasoning.text", "text": "step-by-step"}],
        }
        msg = _convert_dict_to_message(original_dict)
        result = _convert_message_to_dict(msg)
        assert result["reasoning"] == "My thinking process"
        assert result["reasoning_details"] == original_dict["reasoning_details"]

    def test_tool_message_to_dict(self) -> None:
        """Test converting ToolMessage to dict."""
        msg = ToolMessage(content="result", tool_call_id="call_123")
        result = _convert_message_to_dict(msg)
        assert result == {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_123",
        }

    def test_chat_message_to_dict(self) -> None:
        """Test converting ChatMessage to dict."""
        msg = ChatMessage(content="Hello", role="developer")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "developer", "content": "Hello"}

    def test_ai_message_with_tool_calls_to_dict(self) -> None:
        """Test converting AIMessage with tool calls to dict."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "SF"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_dict_to_ai_message(self) -> None:
        """Test converting dict to AIMessage."""
        d = {"role": "assistant", "content": "Hello!"}
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello!"

    def test_dict_to_ai_message_with_reasoning(self) -> None:
        """Test that reasoning is extracted from response dict."""
        d = {
            "role": "assistant",
            "content": "Answer",
            "reasoning": "Let me think...",
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert msg.additional_kwargs["reasoning_content"] == "Let me think..."

    def test_dict_to_ai_message_with_tool_calls(self) -> None:
        """Test converting dict with tool calls to AIMessage."""
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                }
            ],
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    def test_dict_to_ai_message_with_invalid_tool_calls(self) -> None:
        """Test that malformed tool calls produce invalid_tool_calls."""
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "not-valid-json{{{",
                    },
                }
            ],
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert len(msg.invalid_tool_calls) == 1
        assert len(msg.tool_calls) == 0
        assert msg.invalid_tool_calls[0]["name"] == "get_weather"

    def test_dict_to_human_message(self) -> None:
        """Test converting dict to HumanMessage."""
        d = {"role": "user", "content": "Hi"}
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, HumanMessage)

    def test_dict_to_system_message(self) -> None:
        """Test converting dict to SystemMessage."""
        d = {"role": "system", "content": "Be helpful"}
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, SystemMessage)

    def test_dict_to_tool_message(self) -> None:
        """Test converting dict with role=tool to ToolMessage."""
        d = {
            "role": "tool",
            "content": "result data",
            "tool_call_id": "call_42",
            "name": "get_weather",
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, ToolMessage)
        assert msg.content == "result data"
        assert msg.tool_call_id == "call_42"
        assert msg.additional_kwargs["name"] == "get_weather"

    def test_dict_to_chat_message_unknown_role(self) -> None:
        """Test that unrecognized roles fall back to ChatMessage."""
        d = {"role": "developer", "content": "Some content"}
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, ChatMessage)
        assert msg.role == "developer"
        assert msg.content == "Some content"

    def test_ai_message_with_list_content_filters_non_text(self) -> None:
        """Test that non-text blocks are filtered from AIMessage list content."""
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ]
        )
        result = _convert_message_to_dict(msg)
        assert result["content"] == [{"type": "text", "text": "Hello"}]


# ===========================================================================
# _create_chat_result tests
# ===========================================================================


class TestCreateChatResult:
    """Tests for _create_chat_result."""

    def test_model_provider_in_response_metadata(self) -> None:
        """Test that model_provider is set in response metadata."""
        model = _make_model()
        result = model._create_chat_result(_SIMPLE_RESPONSE_DICT)
        assert (
            result.generations[0].message.response_metadata.get("model_provider")
            == "openrouter"
        )

    def test_reasoning_from_response(self) -> None:
        """Test that reasoning content is extracted from response."""
        model = _make_model()
        response_dict: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning": "Let me think...",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        result = model._create_chat_result(response_dict)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "Let me think..."
        )

    def test_usage_metadata_created(self) -> None:
        """Test that usage metadata is created from token usage."""
        model = _make_model()
        result = model._create_chat_result(_SIMPLE_RESPONSE_DICT)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        usage = msg.usage_metadata
        assert usage is not None
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_tool_calls_in_response(self) -> None:
        """Test that tool calls are extracted from response."""
        model = _make_model()
        result = model._create_chat_result(_TOOL_RESPONSE_DICT)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "GetWeather"

    def test_response_model_in_metadata(self) -> None:
        """Test that the response model is included in response_metadata."""
        model = _make_model()
        result = model._create_chat_result(_SIMPLE_RESPONSE_DICT)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.response_metadata["model"] == MODEL_NAME

    def test_response_model_propagated_to_llm_output(self) -> None:
        """Test that llm_output uses response model when available."""
        model = _make_model()
        response = {
            **_SIMPLE_RESPONSE_DICT,
            "model": "openai/gpt-4o",
        }
        result = model._create_chat_result(response)
        assert result.llm_output is not None
        assert result.llm_output["model_name"] == "openai/gpt-4o"

    def test_system_fingerprint_in_metadata(self) -> None:
        """Test that system_fingerprint is included in response_metadata."""
        model = _make_model()
        response = {
            **_SIMPLE_RESPONSE_DICT,
            "system_fingerprint": "fp_abc123",
        }
        result = model._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.response_metadata["system_fingerprint"] == "fp_abc123"

    def test_native_finish_reason_in_metadata(self) -> None:
        """Test that native_finish_reason is included in response_metadata."""
        model = _make_model()
        response: dict[str, Any] = {
            **_SIMPLE_RESPONSE_DICT,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                    "native_finish_reason": "end_turn",
                    "index": 0,
                }
            ],
        }
        result = model._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.response_metadata["native_finish_reason"] == "end_turn"

    def test_missing_optional_metadata_excluded(self) -> None:
        """Test that absent optional fields are not added to response_metadata."""
        model = _make_model()
        response: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }
        result = model._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert "system_fingerprint" not in msg.response_metadata
        assert "native_finish_reason" not in msg.response_metadata
        assert "model" not in msg.response_metadata
        assert result.llm_output is not None
        assert "id" not in result.llm_output
        assert "created" not in result.llm_output
        assert "object" not in result.llm_output

    def test_id_created_object_in_llm_output(self) -> None:
        """Test that id, created, and object are included in llm_output."""
        model = _make_model()
        result = model._create_chat_result(_SIMPLE_RESPONSE_DICT)
        assert result.llm_output is not None
        assert result.llm_output["id"] == "gen-abc123"
        assert result.llm_output["created"] == 1700000000.0
        assert result.llm_output["object"] == "chat.completion"

    def test_float_token_usage_normalized_to_int_in_usage_metadata(self) -> None:
        """Test that float token counts are cast to int in usage_metadata."""
        model = _make_model()
        response: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 585.0,
                "completion_tokens": 56.0,
                "total_tokens": 641.0,
                "completion_tokens_details": {"reasoning_tokens": 10.0},
                "prompt_tokens_details": {"cached_tokens": 20.0},
            },
            "model": MODEL_NAME,
        }
        result = model._create_chat_result(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        usage = msg.usage_metadata
        assert usage is not None
        assert usage["input_tokens"] == 585
        assert isinstance(usage["input_tokens"], int)
        assert usage["output_tokens"] == 56
        assert isinstance(usage["output_tokens"], int)
        assert usage["total_tokens"] == 641
        assert isinstance(usage["total_tokens"], int)
        assert usage["input_token_details"]["cache_read"] == 20
        assert isinstance(usage["input_token_details"]["cache_read"], int)
        assert usage["output_token_details"]["reasoning"] == 10
        assert isinstance(usage["output_token_details"]["reasoning"], int)


# ===========================================================================
# Streaming chunk tests
# ===========================================================================


class TestStreamingChunks:
    """Tests for streaming chunk conversion."""

    def test_reasoning_in_streaming_chunk(self) -> None:
        """Test that reasoning is extracted from streaming delta."""
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning": "Streaming reasoning",
                    },
                },
            ],
        }
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert (
            message_chunk.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning"
        )

    def test_model_provider_in_streaming_chunk(self) -> None:
        """Test that model_provider is set in streaming chunk metadata."""
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                },
            ],
        }
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.response_metadata.get("model_provider") == "openrouter"

    def test_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chunk: dict[str, Any] = {"choices": [{"delta": {"content": "Hello"}}]}
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.additional_kwargs.get("reasoning_content") is None

    def test_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chunk: dict[str, Any] = {"choices": [{"delta": {}}]}
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.additional_kwargs.get("reasoning_content") is None

    def test_chunk_with_tool_calls(self) -> None:
        """Test that tool calls are extracted from streaming delta."""
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"loc',
                                },
                            }
                        ],
                    },
                },
            ],
        }
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert len(message_chunk.tool_call_chunks) == 1
        assert message_chunk.tool_call_chunks[0]["name"] == "get_weather"
        assert message_chunk.tool_call_chunks[0]["args"] == '{"loc'
        assert message_chunk.tool_call_chunks[0]["id"] == "call_1"
        assert message_chunk.tool_call_chunks[0]["index"] == 0

    def test_chunk_with_user_role(self) -> None:
        """Test that a chunk with role=user produces HumanMessageChunk."""
        chunk: dict[str, Any] = {
            "choices": [{"delta": {"role": "user", "content": "test"}}]
        }
        msg = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(msg, HumanMessageChunk)

    def test_chunk_with_system_role(self) -> None:
        """Test that a chunk with role=system produces SystemMessageChunk."""
        chunk: dict[str, Any] = {
            "choices": [{"delta": {"role": "system", "content": "test"}}]
        }
        # Use ChatMessageChunk default so role dispatch isn't short-circuited
        msg = _convert_chunk_to_message_chunk(chunk, ChatMessageChunk)
        assert isinstance(msg, SystemMessageChunk)

    def test_chunk_with_unknown_role(self) -> None:
        """Test that an unknown role falls back to ChatMessageChunk."""
        chunk: dict[str, Any] = {
            "choices": [{"delta": {"role": "developer", "content": "test"}}]
        }
        msg = _convert_chunk_to_message_chunk(chunk, ChatMessageChunk)
        assert isinstance(msg, ChatMessageChunk)

    def test_chunk_with_usage(self) -> None:
        """Test that usage metadata is extracted from streaming chunk."""
        chunk: dict[str, Any] = {
            "choices": [{"delta": {"content": ""}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.usage_metadata is not None
        assert message_chunk.usage_metadata["input_tokens"] == 10


# ===========================================================================
# Usage metadata tests
# ===========================================================================


class TestUsageMetadata:
    """Tests for _create_usage_metadata."""

    def test_basic_usage(self) -> None:
        """Test basic usage metadata creation."""
        usage = _create_usage_metadata(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_float_tokens_cast_to_int(self) -> None:
        """Test that float token counts are cast to int."""
        usage = _create_usage_metadata(
            {"prompt_tokens": 10.0, "completion_tokens": 5.0, "total_tokens": 15.0}
        )
        assert usage["input_tokens"] == 10
        assert isinstance(usage["input_tokens"], int)

    def test_missing_tokens_default_to_zero(self) -> None:
        """Test that missing token fields default to zero."""
        usage = _create_usage_metadata({})
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_total_tokens_computed_if_missing(self) -> None:
        """Test that total_tokens is computed if not provided."""
        usage = _create_usage_metadata({"prompt_tokens": 10, "completion_tokens": 5})
        assert usage["total_tokens"] == 15

    def test_token_details(self) -> None:
        """Test that token details are extracted."""
        usage = _create_usage_metadata(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 20},
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        )
        assert "input_token_details" in usage
        assert usage["input_token_details"]["cache_read"] == 20
        assert "output_token_details" in usage
        assert usage["output_token_details"]["reasoning"] == 10

    def test_cache_creation_details(self) -> None:
        """Test that cache_write_tokens maps to cache_creation."""
        usage = _create_usage_metadata(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 80,
                },
            }
        )
        assert "input_token_details" in usage
        assert usage["input_token_details"]["cache_creation"] == 80

    def test_alternative_token_key_names(self) -> None:
        """Test fallback to input_tokens/output_tokens key names."""
        usage = _create_usage_metadata(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15


# ===========================================================================
# Error-path tests
# ===========================================================================


class TestErrorPaths:
    """Tests for error handling in various code paths."""

    def test_n_less_than_1_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            _make_model(n=0)

    def test_n_greater_than_1_with_streaming_raises(self) -> None:
        """Test that n > 1 with streaming raises ValueError."""
        with pytest.raises(ValueError, match="n must be 1 when streaming"):
            _make_model(n=2, streaming=True)

    def test_n_forwarded_in_params(self) -> None:
        """Test that n > 1 is included in _default_params."""
        model = _make_model(n=3)
        assert model._default_params["n"] == 3

    def test_n_default_excluded_from_params(self) -> None:
        """Test that n=1 (default) is not in _default_params."""
        model = _make_model()
        assert "n" not in model._default_params

    def test_unknown_message_type_raises(self) -> None:
        """Test that unknown message types raise TypeError."""
        from langchain_core.messages import FunctionMessage  # noqa: PLC0415

        msg = FunctionMessage(content="result", name="fn")
        with pytest.raises(TypeError, match="Got unknown type"):
            _convert_message_to_dict(msg)

    def test_duplicate_model_kwargs_raises(self) -> None:
        """Test that passing a param in both field and model_kwargs raises."""
        with pytest.raises(ValueError, match="supplied twice"):
            _make_model(temperature=0.5, model_kwargs={"temperature": 0.7})

    def test_known_field_in_model_kwargs_raises(self) -> None:
        """Test that a known field passed in model_kwargs raises."""
        with pytest.raises(ValueError, match="should be specified explicitly"):
            _make_model(model_kwargs={"model_name": "some-model"})

    def test_max_retries_zero_disables_retries(self) -> None:
        """Test that max_retries=0 does not configure retry."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                max_retries=0,
            )
            call_kwargs = mock_cls.call_args[1]
            assert "retry_config" not in call_kwargs

    def test_max_retries_scales_elapsed_time(self) -> None:
        """Test that max_retries value scales max_elapsed_time."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                max_retries=4,
            )
            call_kwargs = mock_cls.call_args[1]
            retry_config = call_kwargs["retry_config"]
            assert retry_config.backoff.max_elapsed_time == 4 * 150_000


# ===========================================================================
# Reasoning details tests
# ===========================================================================


class TestReasoningDetails:
    """Tests for reasoning_details extraction.

    OpenRouter returns reasoning metadata via `reasoning_details` for models
    like OpenAI o-series and Gemini (thought signatures). This verifies the
    field is preserved in both streaming and non-streaming paths.
    """

    def test_reasoning_details_in_non_streaming_response(self) -> None:
        """Test that reasoning_details are extracted from a non-streaming response."""
        details = [
            {"type": "reasoning.text", "text": "Step 1: analyze the problem"},
            {"type": "reasoning.text", "text": "Step 2: solve it"},
        ]
        d = {
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning_details": details,
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert msg.additional_kwargs["reasoning_details"] == details

    def test_reasoning_details_in_streaming_chunk(self) -> None:
        """Test that reasoning_details are extracted from a streaming chunk."""
        details = [{"type": "reasoning.text", "text": "thinking..."}]
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Answer",
                        "reasoning_details": details,
                    },
                }
            ],
        }
        message_chunk = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)
        assert isinstance(message_chunk, AIMessageChunk)
        assert message_chunk.additional_kwargs["reasoning_details"] == details

    def test_reasoning_and_reasoning_details_coexist(self) -> None:
        """Test that both reasoning and reasoning_details can be present."""
        d = {
            "role": "assistant",
            "content": "Answer",
            "reasoning": "I thought about it",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "detailed thinking"},
            ],
        }
        msg = _convert_dict_to_message(d)
        assert isinstance(msg, AIMessage)
        assert msg.additional_kwargs["reasoning_content"] == "I thought about it"
        assert len(msg.additional_kwargs["reasoning_details"]) == 1

    def test_reasoning_in_full_invoke_flow(self) -> None:
        """Test reasoning extraction through the full invoke path."""
        model = _make_model()
        model.client = MagicMock()
        response_dict: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "9.9 is larger than 9.11",
                        "reasoning": "Comparing decimals: 9.9 = 9.90 > 9.11",
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "text": "Let me compare these numbers...",
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        model.client.chat.send.return_value = _make_sdk_response(response_dict)

        result = model.invoke("Which is larger: 9.11 or 9.9?")
        assert isinstance(result, AIMessage)
        assert result.content == "9.9 is larger than 9.11"
        assert result.additional_kwargs["reasoning_content"] == (
            "Comparing decimals: 9.9 = 9.90 > 9.11"
        )
        assert len(result.additional_kwargs["reasoning_details"]) == 1

    def test_reasoning_in_streaming_flow(self) -> None:
        """Test reasoning extraction through the full streaming path."""
        model = _make_model()
        model.client = MagicMock()
        reasoning_chunks = [
            {
                "choices": [
                    {"delta": {"role": "assistant", "content": ""}, "index": 0}
                ],
                "model": MODEL_NAME,
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-reason",
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning": "Thinking step 1...",
                        },
                        "index": 0,
                    }
                ],
                "model": MODEL_NAME,
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-reason",
            },
            {
                "choices": [
                    {
                        "delta": {"content": "The answer"},
                        "index": 0,
                    }
                ],
                "model": MODEL_NAME,
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-reason",
            },
            {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "model": MODEL_NAME,
                "object": "chat.completion.chunk",
                "created": 1700000000.0,
                "id": "gen-reason",
            },
        ]
        model.client.chat.send.return_value = _MockSyncStream(
            [dict(c) for c in reasoning_chunks]
        )

        chunks = list(model.stream("Think about this"))
        reasoning_found = any(
            c.additional_kwargs.get("reasoning_content") for c in chunks
        )
        assert reasoning_found, "Expected reasoning_content in at least one chunk"


# ===========================================================================
# OpenRouter-specific params tests (issues #34797, #34962)
# ===========================================================================


class TestOpenRouterSpecificParams:
    """Tests for OpenRouter-specific parameter handling."""

    def test_plugins_in_params(self) -> None:
        """Test that `plugins` is included in default params."""
        plugins = [{"id": "web", "max_results": 3}]
        model = _make_model(plugins=plugins)
        params = model._default_params
        assert params["plugins"] == plugins

    def test_plugins_excluded_when_none(self) -> None:
        """Test that `plugins` key is absent when not set."""
        model = _make_model()
        params = model._default_params
        assert "plugins" not in params

    def test_plugins_in_payload(self) -> None:
        """Test that `plugins` appear in the actual SDK call."""
        plugins = [{"id": "web", "max_results": 5}]
        model = _make_model(plugins=plugins)
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke("Search the web for LangChain")
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["plugins"] == plugins

    def test_max_completion_tokens_in_params(self) -> None:
        """Test that max_completion_tokens is included when set."""
        model = _make_model(max_completion_tokens=1024)
        params = model._default_params
        assert params["max_completion_tokens"] == 1024

    def test_max_completion_tokens_excluded_when_none(self) -> None:
        """Test that max_completion_tokens is absent when not set."""
        model = _make_model()
        params = model._default_params
        assert "max_completion_tokens" not in params

    def test_base_url_passed_to_client(self) -> None:
        """Test that base_url is passed as server_url to the SDK client."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                base_url="https://custom.openrouter.ai/api/v1",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["server_url"] == "https://custom.openrouter.ai/api/v1"

    def test_timeout_passed_to_client(self) -> None:
        """Test that timeout is passed as timeout_ms to the SDK client."""
        with patch("openrouter.OpenRouter") as mock_cls:
            mock_cls.return_value = MagicMock()
            ChatOpenRouter(
                model=MODEL_NAME,
                api_key=SecretStr("test-key"),
                timeout=30000,
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["timeout_ms"] == 30000

    def test_all_openrouter_params_in_single_payload(self) -> None:
        """Test that all OpenRouter-specific params coexist in a payload."""
        model = _make_model(
            reasoning={"effort": "high"},
            openrouter_provider={"order": ["Anthropic"], "allow_fallbacks": True},
            route="fallback",
            plugins=[{"id": "web"}],
        )
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_SIMPLE_RESPONSE_DICT)

        model.invoke("Hi")
        call_kwargs = model.client.chat.send.call_args[1]
        assert call_kwargs["reasoning"] == {"effort": "high"}
        assert call_kwargs["provider"] == {
            "order": ["Anthropic"],
            "allow_fallbacks": True,
        }
        assert call_kwargs["route"] == "fallback"
        assert call_kwargs["plugins"] == [{"id": "web"}]


# ===========================================================================
# Multimodal content formatting tests
# ===========================================================================


class TestFormatMessageContent:
    """Tests for `_format_message_content` handling of data blocks."""

    def test_string_content_passthrough(self) -> None:
        """Test that plain string content passes through unchanged."""
        assert _format_message_content("Hello") == "Hello"

    def test_empty_string_passthrough(self) -> None:
        """Test that empty string passes through unchanged."""
        assert _format_message_content("") == ""

    def test_none_passthrough(self) -> None:
        """Test that None passes through unchanged."""
        assert _format_message_content(None) is None

    def test_text_block_passthrough(self) -> None:
        """Test that standard text content blocks pass through."""
        content = [{"type": "text", "text": "Hello"}]
        result = _format_message_content(content)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_image_url_block_passthrough(self) -> None:
        """Test that image_url content blocks pass through."""
        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png"},
            },
        ]
        result = _format_message_content(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"

    def test_image_base64_block(self) -> None:
        """Test that base64 image blocks are converted to image_url format."""
        content = [
            {
                "type": "image",
                "base64": "iVBORw0KGgo=",
                "mime_type": "image/png",
            },
        ]
        result = _format_message_content(content)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_audio_base64_block(self) -> None:
        """Test that base64 audio blocks are converted to input_audio format."""
        content = [
            {"type": "text", "text": "Transcribe this audio."},
            {
                "type": "audio",
                "base64": "UklGR...",
                "mime_type": "audio/wav",
            },
        ]
        result = _format_message_content(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "input_audio"
        assert result[1]["input_audio"]["data"] == "UklGR..."
        assert result[1]["input_audio"]["format"] == "wav"

    def test_video_url_block(self) -> None:
        """Test that video URL blocks are converted to video_url format."""
        content = [
            {"type": "text", "text": "Describe this video."},
            {
                "type": "video",
                "url": "https://example.com/video.mp4",
            },
        ]
        result = _format_message_content(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1] == {
            "type": "video_url",
            "video_url": {"url": "https://example.com/video.mp4"},
        }

    def test_video_base64_block(self) -> None:
        """Test that base64 video blocks are converted to video_url data URI."""
        content = [
            {
                "type": "video",
                "base64": "AAAAIGZ0...",
                "mime_type": "video/mp4",
            },
        ]
        result = _format_message_content(content)
        assert len(result) == 1
        assert result[0]["type"] == "video_url"
        assert result[0]["video_url"]["url"] == ("data:video/mp4;base64,AAAAIGZ0...")

    def test_video_base64_default_mime_type(self) -> None:
        """Test that video base64 defaults to video/mp4 when mime_type is missing."""
        content = [
            {
                "type": "video",
                "base64": "AAAAIGZ0...",
            },
        ]
        result = _format_message_content(content)
        assert result[0]["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_video_block_missing_source_raises(self) -> None:
        """Test that video blocks without url or base64 raise ValueError."""
        block: dict[str, Any] = {"type": "video", "mime_type": "video/mp4"}
        with pytest.raises(ValueError, match=r"url.*base64"):
            _convert_video_block_to_openrouter(block)

    def test_mixed_multimodal_content(self) -> None:
        """Test formatting a message with text, image, audio, and video blocks."""
        content = [
            {"type": "text", "text": "Analyze these inputs."},
            {"type": "image", "url": "https://example.com/img.png"},
            {"type": "audio", "base64": "audio_data", "mime_type": "audio/mp3"},
            {"type": "video", "url": "https://example.com/clip.mp4"},
        ]
        result = _format_message_content(content)
        assert len(result) == 4
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[2]["type"] == "input_audio"
        assert result[3]["type"] == "video_url"


# ===========================================================================
# Structured output tests
# ===========================================================================


class TestStructuredOutputIntegration:
    """Tests for structured output covering issue-specific scenarios."""

    def test_structured_output_function_calling_invokes_with_tools(self) -> None:
        """Test that `function_calling` structured output sends tools in payload."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_TOOL_RESPONSE_DICT)

        structured = model.with_structured_output(GetWeather, method="function_calling")
        # The first step in the chain is the bound model
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        assert "tools" in bound.kwargs
        assert bound.kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "GetWeather"},
        }

    def test_structured_output_json_schema_no_beta_parse(self) -> None:
        """Test that `json_schema` method uses `response_format`, not `beta.parse`."""
        model = _make_model()
        structured = model.with_structured_output(GetWeather, method="json_schema")
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert "schema" in rf["json_schema"]

    def test_response_format_json_schema_reaches_sdk(self) -> None:
        """Test that `response_format` from json_schema method is sent to the SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(
            {
                **_SIMPLE_RESPONSE_DICT,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"location": "SF"}',
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }
        )

        structured = model.with_structured_output(GetWeather, method="json_schema")
        structured.invoke("weather in SF")
        call_kwargs = model.client.chat.send.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    def test_response_format_json_mode_falls_back_to_json_schema_in_sdk(self) -> None:
        """Test that json_mode warns, falls back to json_schema, and reaches SDK."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(
            {
                **_SIMPLE_RESPONSE_DICT,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"location": "SF"}',
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            }
        )

        with pytest.warns(match="Defaulting to 'json_schema'"):
            structured = model.with_structured_output(
                GetWeather, method="json_mode"  # type: ignore[arg-type]
            )
        structured.invoke("weather in SF")
        call_kwargs = model.client.chat.send.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    def test_include_raw_returns_raw_and_parsed_on_success(self) -> None:
        """Test that `include_raw=True` returns raw message, parsed output, no error."""
        model = _make_model()
        model.client = MagicMock()
        model.client.chat.send.return_value = _make_sdk_response(_TOOL_RESPONSE_DICT)

        structured = model.with_structured_output(
            GetWeather, method="function_calling", include_raw=True
        )
        result = structured.invoke("weather in SF")
        assert isinstance(result, dict)
        assert "raw" in result
        assert "parsed" in result
        assert "parsing_error" in result
        assert isinstance(result["raw"], AIMessage)
        assert result["parsing_error"] is None
        # PydanticToolsParser returns a Pydantic instance, not a dict
        assert isinstance(result["parsed"], GetWeather)
        assert result["parsed"].location == "San Francisco"

    def test_include_raw_preserves_raw_on_parse_failure(self) -> None:
        """Test that `include_raw=True` still returns the raw message on parse error."""
        model = _make_model()
        model.client = MagicMock()
        # Return a tool call whose arguments fail Pydantic validation
        # (missing required field "location")
        bad_tool_response: dict[str, Any] = {
            **_SIMPLE_RESPONSE_DICT,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "GetWeather",
                                    "arguments": '{"wrong_field": "oops"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            ],
        }
        model.client.chat.send.return_value = _make_sdk_response(bad_tool_response)

        structured = model.with_structured_output(
            GetWeather, method="function_calling", include_raw=True
        )
        result = structured.invoke("weather in SF")
        assert isinstance(result, dict)
        assert "raw" in result
        assert isinstance(result["raw"], AIMessage)
        # Raw response should have the tool call even though parsing failed
        assert len(result["raw"].tool_calls) == 1
        # Parsed should be None since Pydantic validation failed
        assert result["parsed"] is None
        # parsing_error should capture the validation exception
        assert result["parsing_error"] is not None


# ===========================================================================
# Multiple choices (n > 1) response tests
# ===========================================================================


class TestMultipleChoices:
    """Tests for handling responses with `n > 1`."""

    def test_multiple_choices_in_response(self) -> None:
        """Test that multiple choices in a response produce multiple generations."""
        model = _make_model(n=2)
        response_dict: dict[str, Any] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Answer A"},
                    "finish_reason": "stop",
                    "index": 0,
                },
                {
                    "message": {"role": "assistant", "content": "Answer B"},
                    "finish_reason": "stop",
                    "index": 1,
                },
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        result = model._create_chat_result(response_dict)
        assert len(result.generations) == 2
        assert result.generations[0].message.content == "Answer A"
        assert result.generations[1].message.content == "Answer B"


# ===========================================================================
# Environment variable configuration tests
# ===========================================================================


class TestEnvironmentConfiguration:
    """Tests for environment variable based configuration."""

    def test_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENROUTER_API_BASE env var sets the base URL."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://custom.example.com")
        model = ChatOpenRouter(model=MODEL_NAME)
        assert model.openrouter_api_base == "https://custom.example.com"

    def test_app_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENROUTER_APP_URL env var sets the app URL."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        monkeypatch.setenv("OPENROUTER_APP_URL", "https://myapp.com")
        model = ChatOpenRouter(model=MODEL_NAME)
        assert model.app_url == "https://myapp.com"

    def test_app_title_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENROUTER_APP_TITLE env var sets the app title."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        monkeypatch.setenv("OPENROUTER_APP_TITLE", "My LangChain App")
        model = ChatOpenRouter(model=MODEL_NAME)
        assert model.app_title == "My LangChain App"
