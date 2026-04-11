from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAI

# ---------------------------------------------------------------------------
# Existing tests — fixed to pass api_key so they don't need env var
# ---------------------------------------------------------------------------

_EXPECTED_NUM_TOKENS = {
    "ada": 17,
    "babbage": 17,
    "curie": 17,
    "davinci": 17,
    "gpt-4": 12,
    "gpt-4-32k": 12,
    "gpt-3.5-turbo": 12,
    "o1": 11,
    "o3": 11,
    "gpt-4o": 11,
}

_MODELS = models = ["ada", "babbage", "curie", "davinci"]
_CHAT_MODELS = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "o1", "o3", "gpt-4o"]


@pytest.mark.xfail(reason="Old models require different tiktoken cached file")
@pytest.mark.parametrize("model", _MODELS)
def test_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = OpenAI(model=model)
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


@pytest.mark.parametrize("model", _CHAT_MODELS)
def test_chat_openai_get_num_tokens(model: str) -> None:
    """Test get_tokens."""
    llm = ChatOpenAI(model=model, openai_api_key="fake-key")  # type: ignore[call-arg]
    assert llm.get_num_tokens("表情符号是\n🦜🔗") == _EXPECTED_NUM_TOKENS[model]


# ---------------------------------------------------------------------------
# New tests — get_num_tokens_from_messages via OpenAI token count API
#
# NOTE: These tests require the new implementation in base.py that calls
# self.root_client.responses.input_tokens.count(...) instead of tiktoken.
# If these fail with "assert X == 17" it means base.py has not been updated.
# ---------------------------------------------------------------------------


def test_get_num_tokens_from_messages_uses_api() -> None:
    """Should call OpenAI token count API and return input_tokens."""
    llm = ChatOpenAI(model="gpt-4o", openai_api_key="fake-key")  # type: ignore[call-arg]
    mock_response = MagicMock()
    mock_response.input_tokens = 17

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        return_value=mock_response,
    ) as mock_count:
        result = llm.get_num_tokens_from_messages([
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello!"),
        ])

    assert result == 17
    mock_count.assert_called_once()


def test_get_num_tokens_from_messages_passes_correct_model() -> None:
    """The model name should be forwarded to the API call."""
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key="fake-key")  # type: ignore[call-arg]
    mock_response = MagicMock()
    mock_response.input_tokens = 10

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        return_value=mock_response,
    ) as mock_count:
        llm.get_num_tokens_from_messages([HumanMessage(content="Hi")])

    assert mock_count.called
    call_kwargs = mock_count.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"


def test_get_num_tokens_from_messages_passes_all_messages() -> None:
    """All messages in the list should be forwarded to the API."""
    llm = ChatOpenAI(model="gpt-4o", openai_api_key="fake-key")  # type: ignore[call-arg]
    mock_response = MagicMock()
    mock_response.input_tokens = 40

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        return_value=mock_response,
    ) as mock_count:
        llm.get_num_tokens_from_messages([
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
            HumanMessage(content="Tell me more."),
        ])

    assert mock_count.called
    call_kwargs = mock_count.call_args.kwargs
    assert len(call_kwargs["input"]) == 4


def test_get_num_tokens_from_messages_passes_tools_to_api() -> None:
    """Tool schemas should be included in the API call when provided."""
    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return "Sunny"

    llm = ChatOpenAI(model="gpt-4o", openai_api_key="fake-key")  # type: ignore[call-arg]
    mock_response = MagicMock()
    mock_response.input_tokens = 42

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        return_value=mock_response,
    ) as mock_count:
        result = llm.get_num_tokens_from_messages(
            [HumanMessage(content="What's the weather in Mumbai?")],
            tools=[get_weather],
        )

    assert result == 42
    assert mock_count.called
    call_kwargs = mock_count.call_args.kwargs
    assert "tools" in call_kwargs


def test_get_num_tokens_from_messages_no_tools_key_when_tools_is_none() -> None:
    """When tools=None, the 'tools' key must NOT be sent to the API."""
    llm = ChatOpenAI(model="gpt-4o", openai_api_key="fake-key")  # type: ignore[call-arg]
    mock_response = MagicMock()
    mock_response.input_tokens = 5

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        return_value=mock_response,
    ) as mock_count:
        llm.get_num_tokens_from_messages(
            [HumanMessage(content="Hi")],
            tools=None,
        )

    assert mock_count.called
    call_kwargs = mock_count.call_args.kwargs
    assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# Fallback tests — tiktoken path when API call fails
# ---------------------------------------------------------------------------


def test_get_num_tokens_from_messages_falls_back_on_api_error() -> None:
    """Should fall back to tiktoken when the API raises an exception."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="fake-key")  # type: ignore[call-arg]

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        side_effect=Exception("API unavailable"),
    ):
        result = llm.get_num_tokens_from_messages([
            HumanMessage(content="Hello")
        ])

    assert isinstance(result, int)
    assert result > 0


def test_get_num_tokens_from_messages_fallback_result_is_reasonable() -> None:
    """Tiktoken fallback count should be in a sensible range."""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="fake-key")  # type: ignore[call-arg]

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        side_effect=Exception("API down"),
    ):
        result = llm.get_num_tokens_from_messages([
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="Hello!"),
        ])

    assert 5 < result < 100


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"])
def test_get_num_tokens_from_messages_fallback_supported_models(
    model: str,
) -> None:
    """All tiktoken-supported models should fall back without raising."""
    llm = ChatOpenAI(model=model, openai_api_key="fake-key")  # type: ignore[call-arg]

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        side_effect=Exception("API unavailable"),
    ):
        result = llm.get_num_tokens_from_messages([
            HumanMessage(content="Hello")
        ])

    assert isinstance(result, int)
    assert result > 0


def test_get_num_tokens_from_messages_unsupported_model_raises() -> None:
    """A model not supported by the API or tiktoken should raise NotImplementedError."""
    llm = ChatOpenAI(model="some-unknown-model-xyz", openai_api_key="fake-key")  # type: ignore[call-arg]

    with patch.object(
        llm.root_client.responses.input_tokens,
        "count",
        side_effect=Exception("API unavailable"),
    ):
        with pytest.raises(NotImplementedError):
            llm.get_num_tokens_from_messages([
                HumanMessage(content="Hello")
            ])