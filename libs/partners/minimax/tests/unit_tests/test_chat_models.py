"""Test chat model integration."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletionMessage
from pydantic import SecretStr

from langchain_minimax.chat_models import DEFAULT_API_BASE, ChatMiniMax

MODEL_NAME = "MiniMax-M2.7"


class MockOpenAIResponse(BaseModel):
    """Mock OpenAI response model."""

    choices: list
    error: None = None

    def model_dump(  # type: ignore[override]
        self,
        *,
        mode: Literal["json", "python"] | str = "python",  # noqa: PYI051
        include: Any = None,
        exclude: Any = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: Literal["none", "warn", "error"] | bool = True,
        context: dict[str, Any] | None = None,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary, ensuring `reasoning_content` is included."""
        choices_list = []
        for choice in self.choices:
            if isinstance(choice.message, ChatCompletionMessage):
                message_dict = choice.message.model_dump()
                if "model_extra" in message_dict:
                    message_dict.update(message_dict["model_extra"])
            else:
                message_dict = {
                    "role": "assistant",
                    "content": choice.message.content,
                }
                if hasattr(choice.message, "reasoning_content"):
                    message_dict["reasoning_content"] = choice.message.reasoning_content
                if hasattr(choice.message, "model_extra"):
                    message_dict.update(choice.message.model_extra)
                    message_dict["model_extra"] = choice.message.model_extra
            choices_list.append({"message": message_dict})

        return {"choices": choices_list, "error": self.error}


class TestChatMiniMaxUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatMiniMax` chat model."""

    @property
    def chat_model_class(self) -> type[ChatMiniMax]:
        """Chat model class being tested."""
        return ChatMiniMax

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "MINIMAX_API_KEY": "api_key",
                "MINIMAX_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatMiniMax:
        """Get a chat model instance for testing."""
        return ChatMiniMax(**self.chat_model_params)


class TestChatMiniMaxCustomUnit:
    """Custom tests specific to MiniMax chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatMiniMax(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_default_api_base(self) -> None:
        """Test that default API base is set correctly."""
        chat_model = ChatMiniMax(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.api_base == DEFAULT_API_BASE

    def test_create_chat_result_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from response."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "This is the reasoning content"
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning content"
        )

    def test_convert_chunk_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from streaming chunk."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "Streaming reasoning content",
                    },
                },
            ],
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning content"
        )

    def test_convert_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {"content": "Main content"}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_convert_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_get_request_payload(self) -> None:
        """Test that tool message content is converted from list to string."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))

        tool_message = ToolMessage(content=[], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "[]"

        tool_message = ToolMessage(content=["item1", "item2"], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == '["item1", "item2"]'

        tool_message = ToolMessage(content="test string", tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "test string"

    def test_temperature_clamping(self) -> None:
        """Test that temperature is clamped to (0.0, 1.0]."""
        chat_model = ChatMiniMax(model=MODEL_NAME, api_key=SecretStr("api_key"))

        # Temperature 0 should be clamped to 0.01
        payload = chat_model._get_request_payload(
            [("user", "test")],
            temperature=0,
        )
        assert payload["temperature"] == 0.01

        # Temperature > 1 should be clamped to 1.0
        payload = chat_model._get_request_payload(
            [("user", "test")],
            temperature=2.0,
        )
        assert payload["temperature"] == 1.0

        # Temperature 0.5 should stay as-is
        payload = chat_model._get_request_payload(
            [("user", "test")],
            temperature=0.5,
        )
        assert payload["temperature"] == 0.5


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatMiniMax(model="MiniMax-M2.7", api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["reasoning_output"]
    assert model.profile["tool_calling"]
    assert model.profile["max_input_tokens"] == 1000000


def test_profile_m25() -> None:
    """Test that M2.5 model profile is loaded correctly."""
    model = ChatMiniMax(model="MiniMax-M2.5", api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["max_input_tokens"] == 204800


def test_profile_highspeed() -> None:
    """Test that highspeed model profile is loaded correctly."""
    model = ChatMiniMax(
        model="MiniMax-M2.7-highspeed", api_key=SecretStr("test_key"),
    )
    assert model.profile is not None
    assert model.profile["reasoning_output"]
