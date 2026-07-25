"""Test chat model integration."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletionMessage
from pydantic import SecretStr

from langchain_moonshot.chat_models import DEFAULT_API_BASE, ChatMoonshot

MODEL_NAME = "moonshot-v1-8k"


class MockMoonshotResponse(BaseModel):
    """Mock OpenAI response model."""

    choices: list
    error: None = None

    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",
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
        """Convert to dictionary."""
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
                if hasattr(choice.message, "model_extra"):
                    message_dict.update(choice.message.model_extra)
                    message_dict["model_extra"] = choice.message.model_extra
            choices_list.append({"message": message_dict})
        return {"choices": choices_list, "error": self.error}


class TestChatMoonshotUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatMoonshot` chat model."""

    @property
    def chat_model_class(self) -> type[ChatMoonshot]:
        """Chat model class being tested."""
        return ChatMoonshot

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "MOONSHOT_API_KEY": "api_key",
                "MOONSHOT_API_BASE": "api_base",
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

    def get_chat_model(self) -> ChatMoonshot:
        """Get a chat model instance for testing."""
        return ChatMoonshot(**self.chat_model_params)


class TestChatMoonshotCustomUnit:
    """Custom tests specific to Moonshot chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatMoonshot(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_create_chat_result(self) -> None:
        """Test that chat result is properly created."""
        chat_model = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_response = MockMoonshotResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert result.generations[0].text == "Main content"
        assert (
            result.generations[0].message.response_metadata.get("model_provider")
            == "moonshot"
        )

    def test_convert_chunk_with_content(self) -> None:
        """Test that chunk content is properly extracted."""
        chat_model = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Hello",
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
        assert chunk_result.text == "Hello"

    def test_convert_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chat_model = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {"choices": [{"delta": {}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk,
            AIMessageChunk,
            None,
        )
        if chunk_result is None:
            msg = "Expected chunk_result not to be None"
            raise AssertionError(msg)
        assert chunk_result.text == ""

    def test_get_request_payload(self) -> None:
        """Test that tool message content is converted from list to string."""
        chat_model = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("api_key"))

        tool_message = ToolMessage(content=[], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "[]"

        tool_message = ToolMessage(content=["item1", "item2"], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == '["item1", "item2"]'

        tool_message = ToolMessage(content="test string", tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "test string"


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("test_key"))
    assert model.profile is not None
    assert model.profile["max_input_tokens"] == 8192


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    llm = ChatMoonshot(model=MODEL_NAME, api_key=SecretStr("test_key"))
    assert llm.metadata is not None
    versions = llm.metadata["lc_versions"]
    assert "langchain-core" in versions
    assert "langchain-moonshot" in versions
    assert "langchain-openai" in versions


def test_env_var_defaults() -> None:
    """Test that env vars are used when no explicit params provided."""
    import os

    os.environ["MOONSHOT_API_KEY"] = "env_key"
    os.environ["MOONSHOT_API_BASE"] = "https://custom.test/v1"
    try:
        model = ChatMoonshot(model=MODEL_NAME)
        assert model.api_key.get_secret_value() == "env_key"
        assert model.api_base == "https://custom.test/v1"
    finally:
        del os.environ["MOONSHOT_API_KEY"]
        del os.environ["MOONSHOT_API_BASE"]
