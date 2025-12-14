"""Unit tests for ChatMaritaca.

Author: Anderson Henrique da Silva
Location: Minas Gerais, Brasil
GitHub: https://github.com/anderson-ufrj
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_maritaca import ChatMaritaca
from langchain_maritaca.chat_models import (
    _convert_dict_to_message,
    _convert_message_to_dict,
    _create_usage_metadata,
)

if "MARITACA_API_KEY" not in os.environ:
    os.environ["MARITACA_API_KEY"] = "fake-key"


class TestChatMaritaca:
    """Test suite for ChatMaritaca."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        with patch.dict("os.environ", {"MARITACA_API_KEY": "test-key"}):
            model = ChatMaritaca()
            assert model.model_name == "sabia-3"
            assert model.temperature == 0.7
            assert model.max_retries == 2

    def test_initialization_with_params(self) -> None:
        """Test initialization with custom parameters."""
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabiazinho-3",
            temperature=0.5,
            max_tokens=1000,
            max_retries=3,
        )
        assert model.model_name == "sabiazinho-3"
        assert model.temperature == 0.5
        assert model.max_tokens == 1000
        assert model.max_retries == 3

    def test_initialization_with_alias(self) -> None:
        """Test initialization using parameter aliases."""
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3",
            timeout=30.0,
            base_url="https://custom.api.com",
        )
        assert model.request_timeout == 30.0
        assert model.maritaca_api_base == "https://custom.api.com"

    def test_llm_type(self) -> None:
        """Test _llm_type property."""
        model = ChatMaritaca(api_key="test-key")  # type: ignore[arg-type]
        assert model._llm_type == "maritaca-chat"

    def test_is_lc_serializable(self) -> None:
        """Test is_lc_serializable class method."""
        assert ChatMaritaca.is_lc_serializable() is True

    def test_lc_secrets(self) -> None:
        """Test lc_secrets property."""
        model = ChatMaritaca(api_key="test-key")  # type: ignore[arg-type]
        assert model.lc_secrets == {"maritaca_api_key": "MARITACA_API_KEY"}

    def test_default_params(self) -> None:
        """Test _default_params property."""
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3",
            temperature=0.8,
            max_tokens=500,
            top_p=0.95,
        )
        params = model._default_params
        assert params["model"] == "sabia-3"
        assert params["temperature"] == 0.8
        assert params["max_tokens"] == 500
        assert params["top_p"] == 0.95

    def test_temperature_zero_adjustment(self) -> None:
        """Test that temperature=0 is adjusted to avoid API issues."""
        model = ChatMaritaca(api_key="test-key", temperature=0)  # type: ignore[arg-type]
        assert model.temperature == 1e-8

    def test_n_validation(self) -> None:
        """Test that n must be at least 1."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            ChatMaritaca(api_key="test-key", n=0)  # type: ignore[arg-type]

    def test_streaming_n_validation(self) -> None:
        """Test that n must be 1 when streaming."""
        with pytest.raises(ValueError, match="n must be 1 when streaming"):
            ChatMaritaca(api_key="test-key", n=2, streaming=True)  # type: ignore[arg-type]


class TestMessageConversion:
    """Test message conversion functions."""

    def test_convert_human_message(self) -> None:
        """Test converting HumanMessage."""
        msg = HumanMessage(content="Hello")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_convert_ai_message(self) -> None:
        """Test converting AIMessage."""
        msg = AIMessage(content="Hi there")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "assistant", "content": "Hi there"}

    def test_convert_system_message(self) -> None:
        """Test converting SystemMessage."""
        msg = SystemMessage(content="You are helpful")
        result = _convert_message_to_dict(msg)
        assert result == {"role": "system", "content": "You are helpful"}

    def test_convert_dict_to_human_message(self) -> None:
        """Test converting dict to HumanMessage."""
        msg_dict = {"role": "user", "content": "Hello"}
        result = _convert_dict_to_message(msg_dict)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_convert_dict_to_ai_message(self) -> None:
        """Test converting dict to AIMessage."""
        msg_dict = {"role": "assistant", "content": "Hi there"}
        result = _convert_dict_to_message(msg_dict)
        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    def test_convert_dict_to_system_message(self) -> None:
        """Test converting dict to SystemMessage."""
        msg_dict = {"role": "system", "content": "You are helpful"}
        result = _convert_dict_to_message(msg_dict)
        assert isinstance(result, SystemMessage)
        assert result.content == "You are helpful"


class TestUsageMetadata:
    """Test usage metadata creation."""

    def test_create_usage_metadata(self) -> None:
        """Test creating usage metadata from token usage."""
        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        result = _create_usage_metadata(token_usage)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_create_usage_metadata_missing_total(self) -> None:
        """Test creating usage metadata when total is missing."""
        token_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
        result = _create_usage_metadata(token_usage)
        assert result["total_tokens"] == 30


class TestChatMaritacaIntegration:
    """Integration-style unit tests using mocked HTTP responses."""

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """Create a mock API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "sabia-3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "A capital do Brasil é Brasília.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25,
            },
        }

    def test_invoke_with_mock(self, mock_response: dict[str, Any]) -> None:
        """Test invoke method with mocked HTTP client."""
        model = ChatMaritaca(api_key="test-key")  # type: ignore[arg-type]

        # Mock the HTTP client
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response
        mock_http_response.raise_for_status = MagicMock()

        model.client = MagicMock()
        model.client.post.return_value = mock_http_response

        result = model.invoke([HumanMessage(content="Qual é a capital do Brasil?")])

        assert isinstance(result, AIMessage)
        assert result.content == "A capital do Brasil é Brasília."
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 15
        assert result.usage_metadata["output_tokens"] == 10

    def test_create_chat_result(self, mock_response: dict[str, Any]) -> None:
        """Test _create_chat_result method."""
        model = ChatMaritaca(api_key="test-key")  # type: ignore[arg-type]
        result = model._create_chat_result(mock_response)

        assert len(result.generations) == 1
        assert isinstance(result.generations[0].message, AIMessage)
        expected_content = "A capital do Brasil é Brasília."
        assert result.generations[0].message.content == expected_content
        assert result.llm_output is not None
        assert result.llm_output["model"] == "sabia-3"
        assert result.llm_output["token_usage"]["total_tokens"] == 25

    def test_create_message_dicts(self) -> None:
        """Test _create_message_dicts method."""
        model = ChatMaritaca(api_key="test-key")  # type: ignore[arg-type]
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
        ]

        message_dicts, params = model._create_message_dicts(messages, stop=["END"])

        assert len(message_dicts) == 2
        assert message_dicts[0] == {"role": "system", "content": "You are helpful"}
        assert message_dicts[1] == {"role": "user", "content": "Hello"}
        assert params["stop"] == ["END"]


class TestChatMaritacaLangSmith:
    """Test LangSmith integration."""

    def test_get_ls_params(self) -> None:
        """Test _get_ls_params method."""
        model = ChatMaritaca(
            api_key="test-key",  # type: ignore[arg-type]
            model="sabia-3",
            temperature=0.5,
            max_tokens=100,
        )

        ls_params = model._get_ls_params(stop=["END"])

        assert ls_params["ls_provider"] == "maritaca"
        assert ls_params["ls_model_name"] == "sabia-3"
        assert ls_params["ls_model_type"] == "chat"
        assert ls_params["ls_temperature"] == 0.5
        assert ls_params["ls_max_tokens"] == 100
        assert ls_params["ls_stop"] == ["END"]
