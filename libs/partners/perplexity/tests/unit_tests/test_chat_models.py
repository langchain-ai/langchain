import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding
from pytest_mock import MockerFixture

from langchain_perplexity import ChatPerplexity, MediaResponse, WebSearchOptions
from langchain_perplexity.chat_models import (
    _content_to_text,
    _convert_responses_stream_event_to_chunk,
    _create_usage_metadata,
    _flatten_responses_tool,
    _translate_responses_input,
)


def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model="foo")
    assert llm.model == "foo"


def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model="test", model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_perplexity_initialization() -> None:
    """Test perplexity initialization."""
    # Verify that chat perplexity can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatPerplexity(
            model="test", timeout=1, api_key="test", temperature=0.7, verbose=True
        ),
        ChatPerplexity(
            model="test",
            request_timeout=1,
            pplx_api_key="test",
            temperature=0.7,
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert (
            model.pplx_api_key is not None
            and model.pplx_api_key.get_secret_value() == "test"
        )


def test_perplexity_new_params() -> None:
    """Test new Perplexity-specific parameters."""
    web_search_options = WebSearchOptions(search_type="pro", search_context_size="high")
    media_response = MediaResponse(overrides={"return_videos": True})

    llm = ChatPerplexity(
        model="sonar-pro",
        search_mode="academic",
        web_search_options=web_search_options,
        media_response=media_response,
        return_images=True,
    )

    params = llm._default_params
    assert params["search_mode"] == "academic"
    assert params["web_search_options"] == {
        "search_type": "pro",
        "search_context_size": "high",
    }

    assert params["extra_body"]["media_response"] == {
        "overrides": {"return_videos": True}
    }
    assert params["return_images"] is True


def test_perplexity_stream_includes_citations(mocker: MockerFixture) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)
    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "Perplexity"}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: BaseMessage | None = None
    chunks_list = list(stream)
    # BaseChatModel.stream() adds an extra chunk after the final chunk from _stream
    assert len(chunks_list) == 4
    for i, chunk in enumerate(
        chunks_list[:3]
    ):  # Only check first 3 chunks against mock
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"].get("content", "")
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
    # Process the 4th chunk
    assert full is not None
    full = cast(BaseMessage, full + chunks_list[3])
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {"citations": ["example.com", "example2.com"]}

    patcher.assert_called_once()


def test_perplexity_stream_includes_videos_and_reasoning(mocker: MockerFixture) -> None:
    """Test that stream extracts videos and reasoning_steps."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Thinking... "}, "finish_reason": None}],
        "videos": [{"url": "http://video.com", "thumbnail_url": "http://thumb.com"}],
        "reasoning_steps": [{"thought": "I should search", "type": "web_search"}],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }

    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    mocker.patch.object(llm.client.chat.completions, "create", return_value=mock_stream)

    stream = list(llm.stream("test"))
    first_chunk = stream[0]

    assert "videos" in first_chunk.additional_kwargs
    assert first_chunk.additional_kwargs["videos"][0]["url"] == "http://video.com"
    assert "reasoning_steps" in first_chunk.additional_kwargs
    assert (
        first_chunk.additional_kwargs["reasoning_steps"][0]["thought"]
        == "I should search"
    )


def test_create_usage_metadata_basic() -> None:
    """Test _create_usage_metadata with basic token counts."""
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": 0,
        "citation_tokens": 0,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 20
    assert usage_metadata["total_tokens"] == 30
    assert usage_metadata["output_token_details"]["reasoning"] == 0
    assert usage_metadata["output_token_details"]["citation_tokens"] == 0  # type: ignore[typeddict-item]


def test_perplexity_invoke_includes_num_search_queries(mocker: MockerFixture) -> None:
    """Test that invoke includes num_search_queries in response_metadata."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "num_search_queries": 3,
        "search_context_size": "high",
    }

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Test response",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.model = "test-model"
    mock_response.usage = mock_usage
    # Mock optional fields as empty/None
    mock_response.videos = None
    mock_response.reasoning_steps = None
    mock_response.citations = None
    mock_response.search_results = None
    mock_response.images = None
    mock_response.related_questions = None

    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("Test query")

    assert result.response_metadata["num_search_queries"] == 3
    assert result.response_metadata["search_context_size"] == "high"
    assert result.response_metadata["model_name"] == "test-model"
    patcher.assert_called_once()


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    from langchain_perplexity._version import __version__

    llm = ChatPerplexity(model="test")
    assert llm.metadata is not None
    versions = llm.metadata["versions"]
    assert "langchain-core" in versions
    assert "langchain-perplexity" in versions
    assert versions["langchain-perplexity"] == __version__


def test_profile() -> None:
    model = ChatPerplexity(model="sonar")
    assert model.profile


def test_convert_tool_message_to_dict() -> None:
    """A ToolMessage serializes to a ``tool``-role dict so tool results can be
    fed back to the model in a client-side tool-calling loop."""
    llm = ChatPerplexity(model="test", api_key="test")
    message = ToolMessage(content="result text", tool_call_id="call_123")
    assert llm._convert_message_to_dict(message) == {
        "role": "tool",
        "content": "result text",
        "tool_call_id": "call_123",
    }


def test_convert_ai_message_with_tool_calls_to_dict() -> None:
    """``AIMessage.tool_calls`` are serialized rather than dropped."""
    llm = ChatPerplexity(model="test", api_key="test")
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "search",
                "args": {"query": "langchain"},
                "type": "tool_call",
            }
        ],
    )
    result = llm._convert_message_to_dict(message)
    assert result["role"] == "assistant"
    # Empty content alongside tool_calls must be sent as null, not "".
    assert result["content"] is None
    assert result["tool_calls"] == [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps({"query": "langchain"}),
            },
        }
    ]


def test_convert_ai_message_with_invalid_tool_calls_to_dict() -> None:
    """Invalid tool calls are serialized with their raw (unparsed) argument string."""
    llm = ChatPerplexity(model="test", api_key="test")
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "id": "call_bad",
                "name": "search",
                "args": "{not valid json",
                "error": "could not parse args",
                "type": "invalid_tool_call",
            }
        ],
    )
    result = llm._convert_message_to_dict(message)
    assert result["tool_calls"] == [
        {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "search", "arguments": "{not valid json"},
        }
    ]


def test_convert_ai_message_preserves_content_alongside_tool_calls() -> None:
    """Non-empty content is preserved (not nulled) when tool_calls are present."""
    llm = ChatPerplexity(model="test", api_key="test")
    message = AIMessage(
        content="Let me look that up.",
        tool_calls=[
            {
                "id": "call_123",
                "name": "search",
                "args": {"query": "weather"},
                "type": "tool_call",
            }
        ],
    )
    result = llm._convert_message_to_dict(message)
    assert result["content"] == "Let me look that up."


def test_convert_ai_message_with_valid_and_invalid_tool_calls_to_dict() -> None:
    """Valid and invalid tool calls serialize together, valid ones first."""
    llm = ChatPerplexity(model="test", api_key="test")
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_ok",
                "name": "search",
                "args": {"query": "weather"},
                "type": "tool_call",
            }
        ],
        invalid_tool_calls=[
            {
                "id": "call_bad",
                "name": "search",
                "args": "{not valid json",
                "error": "could not parse args",
                "type": "invalid_tool_call",
            }
        ],
    )
    result = llm._convert_message_to_dict(message)
    assert result["tool_calls"] == [
        {
            "id": "call_ok",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps({"query": "weather"}),
            },
        },
        {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "search", "arguments": "{not valid json"},
        },
    ]


def _weather_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }


def _bound_kwargs(bound: Any) -> dict[str, Any]:
    """Return the kwargs from the `RunnableBinding` that `bind_tools` produces."""
    assert isinstance(bound, RunnableBinding)
    return dict(bound.kwargs)


def test_bind_tools_is_overridden() -> None:
    """`bind_tools` must be overridden so `langchain-tests` detects tool support.

    The standard suite derives `has_tool_calling` from
    `bind_tools is not BaseChatModel.bind_tools`; if this regresses, the entire
    tool-calling test family is silently skipped.
    """
    assert ChatPerplexity.bind_tools is not BaseChatModel.bind_tools


def test_bind_tools_formats_function_tool() -> None:
    """A callable is converted to the OpenAI (Chat Completions) function shape."""
    llm = ChatPerplexity(model="test", api_key="test")

    def get_weather(location: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    bound = llm.bind_tools([get_weather])
    tools = _bound_kwargs(bound)["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"


def test_bind_tools_passes_through_builtin_tool() -> None:
    """Perplexity built-in tools are bound unchanged, not run through conversion."""
    llm = ChatPerplexity(model="test", api_key="test")
    bound = llm.bind_tools([{"type": "web_search"}])
    assert _bound_kwargs(bound)["tools"] == [{"type": "web_search"}]


def test_bind_tools_tool_choice_by_name() -> None:
    llm = ChatPerplexity(model="test", api_key="test")
    bound = llm.bind_tools([_weather_tool()], tool_choice="get_weather")
    assert _bound_kwargs(bound)["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_bind_tools_tool_choice_any_and_bool() -> None:
    llm = ChatPerplexity(model="test", api_key="test")
    any_bound = llm.bind_tools([_weather_tool()], tool_choice="any")
    assert _bound_kwargs(any_bound)["tool_choice"] == "required"
    true_bound = llm.bind_tools([_weather_tool()], tool_choice=True)
    assert _bound_kwargs(true_bound)["tool_choice"] == "required"


def test_bind_tools_tool_choice_invalid_raises() -> None:
    llm = ChatPerplexity(model="test", api_key="test")
    with pytest.raises(ValueError, match="Unrecognized tool_choice"):
        llm.bind_tools([_weather_tool()], tool_choice=123)  # type: ignore[arg-type]


def test_content_to_text() -> None:
    """List content is reduced to text; tool_use and other blocks are dropped."""
    assert _content_to_text("hello") == "hello"
    assert (
        _content_to_text(
            [
                {"type": "text", "text": "some text"},
                {"type": "tool_use", "id": "1", "name": "f", "input": {}},
            ]
        )
        == "some text"
    )
    assert _content_to_text([]) == ""
    assert _content_to_text(None) == ""


def test_flatten_responses_tool() -> None:
    """Function tools are flattened for the Responses API; built-ins pass through."""
    assert _flatten_responses_tool(_weather_tool()) == {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
    assert _flatten_responses_tool({"type": "web_search"}) == {"type": "web_search"}


def test_translate_responses_input_tool_roundtrip() -> None:
    """Tool turns become typed Responses input items (no `tool` role exists)."""
    message_dicts: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "18C cloudy", "tool_call_id": "call_1"},
    ]
    translated = _translate_responses_input(message_dicts)
    assert translated[0] == {"role": "user", "content": "hi"}
    # Empty/None assistant content is dropped; only the function_call item remains.
    assert translated[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": '{"location": "Paris"}',
    }
    assert translated[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "18C cloudy",
    }


def test_translate_responses_input_keeps_assistant_text_with_tool_calls() -> None:
    """An assistant turn with both text and tool_calls emits the text first."""
    translated = _translate_responses_input(
        [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            }
        ]
    )
    assert translated[0] == {"role": "assistant", "content": "Let me check."}
    assert translated[1]["type"] == "function_call"
    assert translated[1]["call_id"] == "call_1"


def test_to_responses_payload_flattens_tools_and_translates_messages() -> None:
    """End-to-end: `_to_responses_payload` flattens tools and translates tool turns."""
    llm = ChatPerplexity(model="openai/gpt-5", api_key="test", use_responses_api=True)
    message_dicts: list[dict[str, Any]] = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "18C cloudy", "tool_call_id": "call_1"},
    ]
    payload = llm._to_responses_payload(message_dicts, {"tools": [_weather_tool()]})
    # tools flattened to the Responses shape
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]
    # tool turns translated into typed input items, with call_id pairing preserved
    fc = [i for i in payload["input"] if i.get("type") == "function_call"]
    fco = [i for i in payload["input"] if i.get("type") == "function_call_output"]
    assert len(fc) == 1
    assert len(fco) == 1
    assert fc[0]["call_id"] == fco[0]["call_id"] == "call_1"
    assert fc[0]["name"] == "get_weather"
    assert fc[0]["arguments"] == '{"location": "Paris"}'


def test_convert_responses_stream_event_emits_tool_call_chunk() -> None:
    """A streamed `function_call` output item becomes a tool-call chunk."""
    event = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "call_id": "call_1",
            "id": "item_1",
            "name": "get_weather",
            "arguments": '{"location": "Paris"}',
        },
    }
    chunk = _convert_responses_stream_event_to_chunk(event)
    assert chunk is not None
    message = chunk.message
    assert isinstance(message, AIMessageChunk)
    tcc = message.tool_call_chunks
    assert len(tcc) == 1
    assert tcc[0]["name"] == "get_weather"
    assert tcc[0]["args"] == '{"location": "Paris"}'
    assert tcc[0]["id"] == "call_1"
    assert tcc[0]["index"] == 0


def test_convert_responses_stream_event_aggregates_multiple_tool_calls() -> None:
    """Distinct Responses output items aggregate as distinct tool calls.

    `call_id`/`id` are intentionally omitted so that `index` (derived from each
    event's `output_index`) is the *only* thing separating the two calls. This
    keeps the test sensitive to the indexing logic: with a hardcoded
    ``index=0`` the chunks would merge into one corrupted call. Real streams
    always carry a unique `call_id`, which would keep the calls distinct on its
    own, so this payload isolates the mechanism rather than mirroring the wire
    format.
    """
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "name": "get_population",
                "arguments": '{"location": "Paris"}',
            },
        },
    ]
    chunks = [
        chunk
        for event in events
        if (chunk := _convert_responses_stream_event_to_chunk(event)) is not None
    ]

    message = chunks[0].message + chunks[1].message

    assert isinstance(message, AIMessageChunk)
    assert message.tool_calls == [
        {
            "name": "get_weather",
            "args": {"location": "Paris"},
            "id": None,
            "type": "tool_call",
        },
        {
            "name": "get_population",
            "args": {"location": "Paris"},
            "id": None,
            "type": "tool_call",
        },
    ]


def test_convert_responses_stream_event_ignores_non_function_items() -> None:
    """Non-function output items (e.g. messages) do not yield tool-call chunks."""
    event = {
        "type": "response.output_item.done",
        "item": {"type": "message", "content": "hi"},
    }
    assert _convert_responses_stream_event_to_chunk(event) is None
