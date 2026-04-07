"""Unit tests for ChatOllama."""

import json
import logging
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_ollama.chat_models import (
    ChatOllama,
    _parse_arguments_from_tool_call,
    _parse_json_string,
)

MODEL_NAME = "llama3.1"


dummy_raw_tool_call = {
    "function": {"name": "test_func", "arguments": ""},
}


class TestChatOllama(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}


def test__parse_arguments_from_tool_call() -> None:
    """Test that string arguments are preserved as strings in tool call parsing.

    PR #30154
    String-typed tool arguments (like IDs or long strings) were being incorrectly
    processed. The parser should preserve string values as strings rather than
    attempting to parse them as JSON when they're already valid string arguments.

    Use a long string ID to ensure string arguments maintain their original type after
    parsing, which is critical for tools expecting string inputs.
    """
    raw_response = (
        '{"model":"sample-model","message":{"role":"assistant","content":"",'
        '"tool_calls":[{"function":{"name":"get_profile_details",'
        '"arguments":{"arg_1":"12345678901234567890123456"}}}]},"done":false}'
    )
    raw_tool_calls = json.loads(raw_response)["message"]["tool_calls"]
    response = _parse_arguments_from_tool_call(raw_tool_calls[0])
    assert response is not None
    assert isinstance(response["arg_1"], str)
    assert response["arg_1"] == "12345678901234567890123456"


def test__parse_arguments_from_tool_call_with_function_name_metadata() -> None:
    """Test that functionName metadata is filtered out from tool arguments.

    Some models may include metadata like `functionName` in the arguments
    that just echoes the function name. This should be filtered out for
    no-argument tools to return an empty dictionary.
    """
    raw_tool_call_with_metadata = {
        "function": {
            "name": "magic_function_no_args",
            "arguments": {"functionName": "magic_function_no_args"},
        }
    }
    response = _parse_arguments_from_tool_call(raw_tool_call_with_metadata)
    assert response == {}

    # Arguments contain both real args and metadata
    raw_tool_call_mixed = {
        "function": {
            "name": "some_function",
            "arguments": {"functionName": "some_function", "real_arg": "value"},
        }
    }
    response_mixed = _parse_arguments_from_tool_call(raw_tool_call_mixed)
    assert response_mixed == {"real_arg": "value"}

    # functionName has different value (should be preserved)
    raw_tool_call_different = {
        "function": {"name": "function_a", "arguments": {"functionName": "function_b"}}
    }
    response_different = _parse_arguments_from_tool_call(raw_tool_call_different)
    assert response_different == {"functionName": "function_b"}


def test_arbitrary_roles_accepted_in_chatmessages() -> None:
    """Test that `ChatOllama` accepts arbitrary roles in `ChatMessage`."""
    response = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "The meaning of life..."},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        llm = ChatOllama(
            model=MODEL_NAME,
            verbose=True,
            format=None,
        )
        messages = [
            ChatMessage(
                role="somerandomrole",
                content="I'm ok with you adding any role message now!",
            ),
            ChatMessage(role="control", content="thinking"),
            ChatMessage(role="user", content="What is the meaning of life?"),
        ]
        llm.invoke(messages)


@patch("langchain_ollama.chat_models.validate_model")
def test_validate_model_on_init(mock_validate_model: Any) -> None:
    """Test that the model is validated on initialization when requested."""
    ChatOllama(model=MODEL_NAME, validate_model_on_init=True)
    mock_validate_model.assert_called_once()
    mock_validate_model.reset_mock()

    ChatOllama(model=MODEL_NAME, validate_model_on_init=False)
    mock_validate_model.assert_not_called()
    ChatOllama(model=MODEL_NAME)
    mock_validate_model.assert_not_called()


@pytest.mark.parametrize(
    ("input_string", "expected_output"),
    [
        # Case 1: Standard double-quoted JSON
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        # Case 2: Single-quoted string (the original bug)
        ("{'key': 'value', 'number': 123}", {"key": "value", "number": 123}),
        # Case 3: String with an internal apostrophe
        ('{"text": "It\'s a great test!"}', {"text": "It's a great test!"}),
        # Case 4: Mixed quotes that ast can handle
        ("{'text': \"It's a great test!\"}", {"text": "It's a great test!"}),
    ],
)
def test_parse_json_string_success_cases(
    input_string: str, expected_output: Any
) -> None:
    """Tests that `_parse_json_string` correctly parses valid and fixable strings."""
    raw_tool_call = {"function": {"name": "test_func", "arguments": input_string}}
    result = _parse_json_string(input_string, raw_tool_call=raw_tool_call, skip=False)
    assert result == expected_output


def test_parse_json_string_failure_case_raises_exception() -> None:
    """Tests that `_parse_json_string` raises an exception for malformed strings."""
    malformed_string = "{'key': 'value',,}"  # Double comma is invalid
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    with pytest.raises(OutputParserException):
        _parse_json_string(
            malformed_string,
            raw_tool_call=raw_tool_call,
            skip=False,
        )


def test_parse_json_string_skip_returns_input_on_failure() -> None:
    """Tests that `skip=True` returns the original string on parse failure."""
    malformed_string = "{'not': valid,,,}"
    raw_tool_call = {"function": {"name": "test_func", "arguments": malformed_string}}
    result = _parse_json_string(
        malformed_string,
        raw_tool_call=raw_tool_call,
        skip=True,  # We want the original invalid string back
    )
    assert result == malformed_string


def test_load_response_with_empty_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that load responses with empty content log a warning and are skipped."""
    load_only_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_only_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_response_with_whitespace_content_is_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses w/ only whitespace content log a warning and are skipped."""
    load_whitespace_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "   \n  \t  "},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_whitespace_response

        llm = ChatOllama(model="test-model")

        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ValueError, match="No data received from Ollama stream"),
        ):
            llm.invoke([HumanMessage("Hello")])
        assert "Ollama returned empty response with done_reason='load'" in caplog.text


def test_load_followed_by_content_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses log a warning and are skipped when followed by content."""
    load_then_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": ""},
        },
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:01.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_then_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert "Ollama returned empty response with done_reason='load'" in caplog.text
        assert result.content == "Hello! How can I help you today?"
        assert result.response_metadata.get("done_reason") == "stop"


def test_load_response_with_actual_content_is_not_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load responses with actual content are NOT skipped and log no warning."""
    load_with_content_response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "load",
            "message": {"role": "assistant", "content": "This is actual content"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_with_content_response

        llm = ChatOllama(model="test-model")

        with caplog.at_level(logging.WARNING):
            result = llm.invoke([HumanMessage("Hello")])

        assert result.content == "This is actual content"
        assert result.response_metadata.get("done_reason") == "load"
        assert not caplog.text


def test_none_parameters_excluded_from_options() -> None:
    """Test that None parameters are excluded from the options dict sent to Ollama."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        # Create ChatOllama with only num_ctx set
        llm = ChatOllama(model="test-model", num_ctx=4096)
        llm.invoke([HumanMessage("Hello")])

        # Verify that chat was called
        assert mock_client.chat.called

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Only num_ctx should be in options, not None parameters
        assert "num_ctx" in options
        assert options["num_ctx"] == 4096

        # These parameters should NOT be in options since they were None
        assert "mirostat" not in options
        assert "mirostat_eta" not in options
        assert "mirostat_tau" not in options
        assert "tfs_z" not in options


def test_all_none_parameters_results_in_empty_options() -> None:
    """Test that when all parameters are None, options dict is empty."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        # Create ChatOllama with no parameters set
        llm = ChatOllama(model="test-model")
        llm.invoke([HumanMessage("Hello")])

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Options should be empty when no parameters are set
        assert options == {}


def test_explicit_options_dict_preserved() -> None:
    """Test that explicitly provided options dict is preserved and not filtered."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        llm = ChatOllama(model="test-model")
        # Pass explicit options dict, including None values
        llm.invoke(
            [HumanMessage("Hello")],
            options={"temperature": 0.5, "custom_param": None},
        )

        # Get the options dict that was passed to chat
        call_kwargs = mock_client.chat.call_args[1]
        options = call_kwargs.get("options", {})

        # Explicit options should be preserved as-is
        assert options == {"temperature": 0.5, "custom_param": None}


def test_reasoning_param_passed_to_client() -> None:
    """Test that the reasoning parameter is correctly passed to the Ollama client."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "deepseek-r1",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "message": {"role": "assistant", "content": "I am thinking..."},
                "done": True,
                "done_reason": "stop",
            }
        ]

        # Case 1: reasoning=True in init
        llm = ChatOllama(model="deepseek-r1", reasoning=True)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True

        # Case 2: reasoning=False in init
        llm = ChatOllama(model="deepseek-r1", reasoning=False)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is False

        # Case 3: reasoning passed in invoke
        llm = ChatOllama(model="deepseek-r1")
        llm.invoke([HumanMessage("Hello")], reasoning=True)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True


def test_logprobs_params_passed_to_client() -> None:
    """Test that logprobs parameters are correctly passed to the Ollama client."""
    response = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "done_reason": "stop",
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        # Case 1: logprobs=True, top_logprobs=5 in init
        llm = ChatOllama(model=MODEL_NAME, logprobs=True, top_logprobs=5)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 5

        # Case 2: override via invoke kwargs
        llm = ChatOllama(model=MODEL_NAME)
        llm.invoke([HumanMessage("Hello")], logprobs=True, top_logprobs=3)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 3

        # Case 3: auto-enabled logprobs propagates to client
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            llm = ChatOllama(model=MODEL_NAME, top_logprobs=3)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 3

        # Case 4: defaults are None when not set
        llm = ChatOllama(model=MODEL_NAME)
        llm.invoke([HumanMessage("Hello")])

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["logprobs"] is None
        assert call_kwargs["top_logprobs"] is None


def test_top_logprobs_validation() -> None:
    """Test that top_logprobs must be a positive integer."""
    with patch("langchain_ollama.chat_models.Client"):
        with pytest.raises(ValueError, match="`top_logprobs` must be a positive"):
            ChatOllama(model=MODEL_NAME, top_logprobs=0)

        with pytest.raises(ValueError, match="`top_logprobs` must be a positive"):
            ChatOllama(model=MODEL_NAME, top_logprobs=-1)

        # Valid values should not raise
        llm = ChatOllama(model=MODEL_NAME, logprobs=True, top_logprobs=1)
        assert llm.top_logprobs == 1


def test_top_logprobs_without_logprobs_auto_enables() -> None:
    """Test that setting top_logprobs without logprobs auto-enables logprobs."""
    with patch("langchain_ollama.chat_models.Client"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            llm = ChatOllama(model=MODEL_NAME, top_logprobs=5)
            assert llm.logprobs is True
            assert len(w) == 1
            assert "Setting `logprobs=True` automatically" in str(w[0].message)

        # No warning when logprobs=True explicitly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChatOllama(model=MODEL_NAME, logprobs=True, top_logprobs=5)
            logprobs_warnings = [x for x in w if "top_logprobs" in str(x.message)]
            assert len(logprobs_warnings) == 0


def test_top_logprobs_with_logprobs_false_raises() -> None:
    """Setting top_logprobs with logprobs=False is a contradictory config."""
    with (
        patch("langchain_ollama.chat_models.Client"),
        pytest.raises(ValueError, match=r"logprobs.*explicitly.*False"),
    ):
        ChatOllama(model=MODEL_NAME, logprobs=False, top_logprobs=5)


def test_logprobs_accumulated_from_stream_into_response_metadata() -> None:
    """Logprobs from intermediate streaming chunks are accumulated into the
    final response_metadata when using invoke()."""
    stream_responses = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "The"},
            "done": False,
            "logprobs": [
                {"token": "The", "logprob": -0.5, "bytes": [84, 104, 101]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": " sky"},
            "done": False,
            "logprobs": [
                {"token": " sky", "logprob": -0.1, "bytes": [32, 115, 107, 121]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(stream_responses)

        llm = ChatOllama(model=MODEL_NAME, logprobs=True)
        result = llm.invoke([HumanMessage("What color is the sky?")])

        logprobs = result.response_metadata["logprobs"]
        assert len(logprobs) == 2
        assert logprobs[0]["token"] == "The"
        assert logprobs[0]["logprob"] == -0.5
        assert logprobs[1]["token"] == " sky"
        assert logprobs[1]["logprob"] == -0.1


def test_logprobs_on_individual_streaming_chunks() -> None:
    """Each streaming chunk should carry its own per-token logprobs in
    response_metadata when logprobs are enabled."""
    stream_responses = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "Hi"},
            "done": False,
            "logprobs": [
                {"token": "Hi", "logprob": -0.3, "bytes": [72, 105]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "!"},
            "done": False,
            "logprobs": [
                {"token": "!", "logprob": -0.01, "bytes": [33]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(stream_responses)

        llm = ChatOllama(model=MODEL_NAME, logprobs=True)
        chunks = list(llm.stream([HumanMessage("Hello")]))

        assert chunks[0].response_metadata["logprobs"] == [
            {"token": "Hi", "logprob": -0.3, "bytes": [72, 105]},
        ]

        assert chunks[1].response_metadata["logprobs"] == [
            {"token": "!", "logprob": -0.01, "bytes": [33]},
        ]

        assert "logprobs" not in chunks[2].response_metadata


async def test_logprobs_on_individual_async_streaming_chunks() -> None:
    """Async streaming chunks should carry per-token logprobs in
    response_metadata when logprobs are enabled."""
    stream_responses = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "Hi"},
            "done": False,
            "logprobs": [
                {"token": "Hi", "logprob": -0.3, "bytes": [72, 105]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "!"},
            "done": False,
            "logprobs": [
                {"token": "!", "logprob": -0.01, "bytes": [33]},
            ],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    async def async_stream_responses() -> Any:
        for resp in stream_responses:
            yield resp

    with patch("langchain_ollama.chat_models.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = async_stream_responses()

        llm = ChatOllama(model=MODEL_NAME, logprobs=True)
        chunks = [chunk async for chunk in llm.astream([HumanMessage("Hello")])]

        assert chunks[0].response_metadata["logprobs"] == [
            {"token": "Hi", "logprob": -0.3, "bytes": [72, 105]},
        ]

        assert chunks[1].response_metadata["logprobs"] == [
            {"token": "!", "logprob": -0.01, "bytes": [33]},
        ]

        assert "logprobs" not in chunks[2].response_metadata


def test_logprobs_empty_list_preserved() -> None:
    """An empty logprobs list `[]` should be preserved, not treated as absent."""
    stream_responses = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "Hi"},
            "done": False,
            "logprobs": [],
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(stream_responses)

        llm = ChatOllama(model=MODEL_NAME, logprobs=True)
        chunks = list(llm.stream([HumanMessage("Hello")]))

        assert chunks[0].response_metadata["logprobs"] == []


def test_logprobs_none_when_not_requested() -> None:
    """When logprobs are not requested, response_metadata should not contain
    logprobs (or it should be None)."""
    stream_responses = [
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": False,
        },
        {
            "model": MODEL_NAME,
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
        },
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = iter(stream_responses)

        llm = ChatOllama(model=MODEL_NAME)
        result = llm.invoke([HumanMessage("Hello")])

        assert result.response_metadata.get("logprobs") is None


def test_create_chat_stream_raises_when_client_none() -> None:
    """Test that _create_chat_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = ChatOllama(model="test-model")
        # Force _client to None to simulate uninitialized state
        llm._client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="sync client is not initialized"):
            list(llm._create_chat_stream([HumanMessage("Hello")]))


async def test_acreate_chat_stream_raises_when_client_none() -> None:
    """Test that _acreate_chat_stream raises RuntimeError when client is None."""
    with patch("langchain_ollama.chat_models.AsyncClient") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = ChatOllama(model="test-model")
        # Force _async_client to None to simulate uninitialized state
        llm._async_client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="async client is not initialized"):
            async for _ in llm._acreate_chat_stream([HumanMessage("Hello")]):
                pass


def test_invoke_raises_when_client_none() -> None:
    """Test that RuntimeError propagates through the public invoke() API."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        llm = ChatOllama(model="test-model")
        llm._client = None  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="sync client is not initialized"):
            llm.invoke([HumanMessage("Hello")])


def test_chat_ollama_ignores_strict_arg() -> None:
    """Test that ChatOllama ignores the 'strict' argument."""
    response = [
        {
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00.000000000Z",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ]

    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = response

        llm = ChatOllama(model="test-model")
        # Invoke with strict=True
        llm.invoke([HumanMessage("Hello")], strict=True)

        # Check that 'strict' was NOT passed to the client
        call_kwargs = mock_client.chat.call_args[1]
        assert "strict" not in call_kwargs


def test_chat_ollama_supports_response_format_json_schema() -> None:
    """Test that ChatOllama correctly maps json_schema response_format to format."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": schema, "strict": True},
        }

        llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") == schema


def test_chat_ollama_supports_response_format_json_object() -> None:
    """Test ChatOllama maps json_object response_format to format='json'."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        response_format = {"type": "json_object"}

        llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") == "json"


def test_chat_ollama_prioritizes_explicit_format() -> None:
    """Test explicit 'format' arg takes precedence over 'response_format'."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        response_format = {"type": "json_object"}

        # User passes BOTH format param and response_format
        # Should warn about ignored response_format
        with pytest.warns(UserWarning, match="Both 'format' and 'response_format'"):
            llm.invoke(
                [HumanMessage("Hello")],
                format="some_custom_format",
                response_format=response_format,
            )

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        # Should keep the explicit format
        assert call_kwargs.get("format") == "some_custom_format"


def test_chat_ollama_warns_invalid_response_format_type() -> None:
    """Test ChatOllama warns on non-dict response_format."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        # Pass a list (invalid type) instead of a dict
        response_format = ["invalid_type"]

        with pytest.warns(UserWarning, match="Ignored invalid 'response_format' type"):
            llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") is None


def test_chat_ollama_warns_unrecognized_response_format_type() -> None:
    """Test ChatOllama warns on unrecognized response_format type (e.g. 'text')."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")
        response_format = {"type": "text"}  # Not json_object or json_schema

        with pytest.warns(UserWarning, match="Ignored unrecognized 'response_format'"):
            llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") is None


def test_chat_ollama_warns_json_schema_missing_schema_key() -> None:
    """Test ChatOllama warns when json_schema block has no 'schema' key."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")

        # json_schema present but no schema key
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test"},
        }
        with pytest.warns(UserWarning, match="no 'schema' key was found"):
            llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") is None


def test_chat_ollama_warns_json_schema_missing_json_schema_key() -> None:
    """Test ChatOllama warns when json_schema type has no 'json_schema' block."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")

        # type is json_schema but json_schema key is missing entirely
        response_format = {"type": "json_schema"}
        with pytest.warns(UserWarning, match="'json_schema' value is"):
            llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") is None


def test_chat_ollama_warns_json_schema_block_not_dict() -> None:
    """Test ChatOllama warns when json_schema value is not a dict."""
    with patch("langchain_ollama.chat_models.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = [
            {
                "model": "gpt-oss:20b",
                "created_at": "2025-01-01T00:00:00.000000000Z",
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "{}"},
            }
        ]

        llm = ChatOllama(model="gpt-oss:20b")

        # json_schema is a string instead of a dict
        response_format = {"type": "json_schema", "json_schema": "not_a_dict"}
        with pytest.warns(UserWarning, match="'json_schema' value is"):
            llm.invoke([HumanMessage("Hello")], response_format=response_format)

        call_kwargs = mock_client.chat.call_args[1]
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("format") is None


def test_reasoning_content_serialized_as_thinking() -> None:
    """Test that `reasoning_content` in `AIMessage` is serialized as `'thinking'`.

    When an AIMessage has `reasoning_content` in `additional_kwargs` (set during
    deserialization of Ollama thinking responses), it should be written back as
    the 'thinking' field in the outgoing Ollama message dict so the model can
    see its prior chain-of-thought in multi-turn conversations.

    Reproduces https://github.com/langchain-ai/langchain/issues/36177.
    """
    with patch("langchain_ollama.chat_models.Client"):
        llm = ChatOllama(model="deepseek-r1")

    messages: list[BaseMessage] = [
        HumanMessage(content="Solve 2+2"),
        AIMessage(
            content="4",
            additional_kwargs={"reasoning_content": "2+2 equals 4"},
        ),
        HumanMessage(content="Now solve 3+3"),
    ]
    ollama_messages = llm._convert_messages_to_ollama_messages(messages)

    assistant_msg = ollama_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg.get("thinking") == "2+2 equals 4", (
        "reasoning_content should be serialized as 'thinking' in the Ollama message"
    )


def test_convert_messages_does_not_mutate_input_list() -> None:
    """Test that `_convert_messages_to_ollama_messages` does not mutate the input list.

    Previously, the v1 content conversion replaced elements in the input list
    via `messages[idx] = ...`, which mutated the caller's list in-place.

    Regression test for https://github.com/langchain-ai/langchain/issues/36564.
    """
    with patch("langchain_ollama.chat_models.Client"):
        llm = ChatOllama(model="test-model")

    v1_ai_message = AIMessage(
        content=[{"type": "text", "text": "Hello from v1"}],
        response_metadata={"output_version": "v1"},
    )
    messages: list = [
        HumanMessage(content="Hi"),
        v1_ai_message,
    ]

    # Keep a reference to the original second element
    original_message = messages[1]

    llm._convert_messages_to_ollama_messages(messages)

    assert messages[1] is original_message, (
        "_convert_messages_to_ollama_messages should not mutate the caller's list"
    )


def test_reasoning_content_absent_no_thinking_key() -> None:
    """AIMessage without `reasoning_content` should not produce a `thinking` key."""
    with patch("langchain_ollama.chat_models.Client"):
        llm = ChatOllama(model="test-model")

    messages: list[BaseMessage] = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello"),
    ]
    ollama_messages = llm._convert_messages_to_ollama_messages(messages)
    assert "thinking" not in ollama_messages[1]


def test_reasoning_content_empty_string_preserved() -> None:
    """An explicitly set empty-string `reasoning_content` should still round-trip."""
    with patch("langchain_ollama.chat_models.Client"):
        llm = ChatOllama(model="test-model")

    messages: list[BaseMessage] = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello", additional_kwargs={"reasoning_content": ""}),
    ]
    ollama_messages = llm._convert_messages_to_ollama_messages(messages)
    assert ollama_messages[1].get("thinking") == ""


def test_non_ai_message_reasoning_content_ignored() -> None:
    """Non-AIMessage types with `reasoning_content` should not produce `thinking`."""
    with patch("langchain_ollama.chat_models.Client"):
        llm = ChatOllama(model="test-model")

    messages: list[BaseMessage] = [
        HumanMessage(
            content="Hi",
            additional_kwargs={"reasoning_content": "should be ignored"},
        ),
    ]
    ollama_messages = llm._convert_messages_to_ollama_messages(messages)
    assert "thinking" not in ollama_messages[0]
