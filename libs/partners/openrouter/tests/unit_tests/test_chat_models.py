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
    _create_usage_metadata,
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

    def test_openrouter_reasoning_in_params(self) -> None:
        """Test that openrouter_reasoning is included in default params."""
        model = _make_model(openrouter_reasoning={"effort": "high"})
        params = model._default_params
        assert params["reasoning"] == {"effort": "high"}

    def test_openrouter_provider_in_params(self) -> None:
        """Test that openrouter_provider is included in default params."""
        model = _make_model(openrouter_provider={"order": ["Anthropic"]})
        params = model._default_params
        assert params["provider"] == {"order": ["Anthropic"]}

    def test_openrouter_route_in_params(self) -> None:
        """Test that openrouter_route is included in default params."""
        model = _make_model(openrouter_route="fallback")
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
            openrouter_reasoning={"effort": "high"},
            openrouter_provider={"order": ["Anthropic"]},
            openrouter_route="fallback",
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

    def test_bind_tools_strict_ignored(self) -> None:
        """Test that strict param is accepted but ignored."""
        model = _make_model()
        bound = model.bind_tools([GetWeather], strict=True)
        assert isinstance(bound, RunnableBinding)


# ===========================================================================
# with_structured_output tests
# ===========================================================================


class TestWithStructuredOutput:
    """Tests for the with_structured_output public method."""

    @pytest.mark.parametrize("method", ["function_calling", "json_schema", "json_mode"])
    @pytest.mark.parametrize("include_raw", ["yes", "no"])
    def test_with_structured_output_pydantic(
        self,
        method: Literal["function_calling", "json_mode", "json_schema"],
        include_raw: str,
    ) -> None:
        """Test with_structured_output using a Pydantic schema."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method=method, include_raw=(include_raw == "yes")
        )
        assert structured is not None

    @pytest.mark.parametrize("method", ["function_calling", "json_schema", "json_mode"])
    def test_with_structured_output_dict_schema(
        self,
        method: Literal["function_calling", "json_mode", "json_schema"],
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

    def test_with_structured_output_json_mode_sets_response_format(self) -> None:
        """Test that json_mode method sets response_format correctly."""
        model = _make_model()
        structured = model.with_structured_output(GenerateUsername, method="json_mode")
        bound = structured.first  # type: ignore[attr-defined]
        assert isinstance(bound, RunnableBinding)
        rf = bound.kwargs["response_format"]
        assert rf["type"] == "json_object"

    def test_with_structured_output_strict_ignored(self) -> None:
        """Test that strict param is accepted but ignored."""
        model = _make_model()
        structured = model.with_structured_output(
            GenerateUsername, method="function_calling", strict=True
        )
        assert structured is not None


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
        assert "tool_calls" in message_chunk.additional_kwargs

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
        assert "output_token_details" in usage

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
# _combine_llm_outputs tests
# ===========================================================================


class TestCombineLLMOutputs:
    """Tests for _combine_llm_outputs."""

    def test_single_output(self) -> None:
        """Test combining a single output."""
        model = _make_model()
        result = model._combine_llm_outputs(
            [{"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
        )
        assert result["token_usage"]["prompt_tokens"] == 10
        assert result["token_usage"]["completion_tokens"] == 5

    def test_multiple_outputs_accumulated(self) -> None:
        """Test that token counts from multiple outputs are accumulated."""
        model = _make_model()
        result = model._combine_llm_outputs(
            [
                {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    }
                },
                {
                    "token_usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    }
                },
            ]
        )
        assert result["token_usage"]["prompt_tokens"] == 30
        assert result["token_usage"]["completion_tokens"] == 15
        assert result["token_usage"]["total_tokens"] == 45

    def test_none_outputs_skipped(self) -> None:
        """Test that None outputs are skipped."""
        model = _make_model()
        result = model._combine_llm_outputs(
            [None, {"token_usage": {"prompt_tokens": 10}}, None]
        )
        assert result["token_usage"]["prompt_tokens"] == 10

    def test_empty_list(self) -> None:
        """Test combining an empty list."""
        model = _make_model()
        result = model._combine_llm_outputs([])
        assert result["token_usage"] == {}
        assert result["model_name"] == MODEL_NAME

    def test_nested_dict_accumulation(self) -> None:
        """Test that nested dicts (e.g. token details) are accumulated."""
        model = _make_model()
        result = model._combine_llm_outputs(
            [
                {
                    "token_usage": {
                        "prompt_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 5},
                    }
                },
                {
                    "token_usage": {
                        "prompt_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    }
                },
            ]
        )
        assert result["token_usage"]["prompt_tokens"] == 30
        assert result["token_usage"]["prompt_tokens_details"]["cached_tokens"] == 8

    def test_none_token_usage_skipped(self) -> None:
        """Test that outputs with token_usage=None are handled."""
        model = _make_model()
        result = model._combine_llm_outputs(
            [{"token_usage": None}, {"token_usage": {"prompt_tokens": 5}}]
        )
        assert result["token_usage"]["prompt_tokens"] == 5


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
