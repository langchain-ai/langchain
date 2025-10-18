"""Test chat model integration."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletionMessage
from pydantic import SecretStr

from langchain_deepseek.chat_models import ChatDeepSeek

MODEL_NAME = "deepseek-chat"


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
                # Ensure model_extra fields are at top level
                if "model_extra" in message_dict:
                    message_dict.update(message_dict["model_extra"])
            else:
                message_dict = {
                    "role": "assistant",
                    "content": choice.message.content,
                }
                # Add reasoning_content if present
                if hasattr(choice.message, "reasoning_content"):
                    message_dict["reasoning_content"] = choice.message.reasoning_content
                # Add model_extra fields at the top level if present
                if hasattr(choice.message, "model_extra"):
                    message_dict.update(choice.message.model_extra)
                    message_dict["model_extra"] = choice.message.model_extra
            choices_list.append({"message": message_dict})

        return {"choices": choices_list, "error": self.error}


class TestChatDeepSeekUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatDeepSeek` chat model."""

    @property
    def chat_model_class(self) -> type[ChatDeepSeek]:
        """Chat model class being tested."""
        return ChatDeepSeek

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "DEEPSEEK_API_KEY": "api_key",
                "DEEPSEEK_API_BASE": "api_base",
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

    def get_chat_model(self) -> ChatDeepSeek:
        """Get a chat model instance for testing."""
        return ChatDeepSeek(**self.chat_model_params)


class TestChatDeepSeekCustomUnit:
    """Custom tests specific to DeepSeek chat model."""

    def test_create_chat_result_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from response."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
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

    def test_create_chat_result_with_model_extra_reasoning(self) -> None:
        """Test that reasoning is properly extracted from `model_extra`."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_message.model_extra = {"reasoning": "This is the reasoning"}
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Main content",
            "model_extra": {"reasoning": "This is the reasoning"},
        }
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MockOpenAIResponse(choices=[mock_choice], error=None)

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning"
        )

    def test_convert_chunk_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
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

    def test_convert_chunk_with_reasoning(self) -> None:
        """Test that reasoning is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
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
            == "Streaming reasoning"
        )

    def test_convert_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
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
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
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
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))

        tool_message = ToolMessage(content=[], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "[]"

        tool_message = ToolMessage(content=["item1", "item2"], tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == '["item1", "item2"]'

        tool_message = ToolMessage(content="test string", tool_call_id="test_id")
        payload = chat_model._get_request_payload([tool_message])
        assert payload["messages"][0]["content"] == "test string"

    def test_create_chat_result_with_model_provider(self) -> None:
        """Test that `model_provider` is added to `response_metadata`."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.response_metadata.get("model_provider")
            == "deepseek"
        )

    def test_convert_chunk_with_model_provider(self) -> None:
        """Test that `model_provider` is added to `response_metadata` for chunks."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        chunk: dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
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
            chunk_result.message.response_metadata.get("model_provider") == "deepseek"
        )

    def test_create_chat_result_with_model_provider_multiple_generations(
        self,
    ) -> None:
        """Test that `model_provider` is added to all generations when `n > 1`."""
        chat_model = ChatDeepSeek(model=MODEL_NAME, api_key=SecretStr("api_key"))
        mock_message_1 = MagicMock()
        mock_message_1.content = "First response"
        mock_message_1.role = "assistant"
        mock_message_2 = MagicMock()
        mock_message_2.content = "Second response"
        mock_message_2.role = "assistant"

        mock_response = MockOpenAIResponse(
            choices=[
                MagicMock(message=mock_message_1),
                MagicMock(message=mock_message_2),
            ],
            error=None,
        )

        result = chat_model._create_chat_result(mock_response)
        assert len(result.generations) == 2  # noqa: PLR2004
        for generation in result.generations:
            assert (
                generation.message.response_metadata.get("model_provider") == "deepseek"
            )


class TestChatDeepSeekStrictMode:
    """Test strict mode functionality."""

    def test_strict_mode_uses_beta_api(self) -> None:
        """Test that strict mode switches to Beta API endpoint."""
        model = ChatDeepSeek(
            model=MODEL_NAME,
            api_key=SecretStr("test-key"),
            strict=True,
        )

        # Check that the client uses the beta endpoint
        assert str(model.root_client.base_url) == "https://api.deepseek.com/beta/"

    def test_strict_mode_disabled_uses_default_api(self) -> None:
        """Test that without strict mode, default API is used."""
        model = ChatDeepSeek(
            model=MODEL_NAME,
            api_key=SecretStr("test-key"),
            strict=False,
        )

        # Check that the client uses the default endpoint
        assert str(model.root_client.base_url) == "https://api.deepseek.com/v1/"

    def test_strict_mode_none_uses_default_api(self) -> None:
        """Test that strict=None uses default API."""
        model = ChatDeepSeek(
            model=MODEL_NAME,
            api_key=SecretStr("test-key"),
        )

        # Check that the client uses the default endpoint
        assert str(model.root_client.base_url) == "https://api.deepseek.com/v1/"

    def test_bind_tools_with_strict_mode(self) -> None:
        """Test that bind_tools adds strict to tool definitions."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the current weather in a given location."""
            location: str = Field(..., description="The city and state") # pyright: ignore[reportUndefinedVariable]

        model = ChatDeepSeek(
            model=MODEL_NAME,
            api_key=SecretStr("test-key"),
            strict=True,
        )

        # Bind tools
        model_with_tools = model.bind_tools([GetWeather])

        # Check that tools were bound
        assert 'tools' in model_with_tools.kwargs

        # Verify that tools have strict property set
        tools = model_with_tools.kwargs['tools']
        assert len(tools) > 0
        assert tools[0]['function']['strict'] is True
    def test_bind_tools_override_strict(self) -> None:
        """Test that bind_tools can override instance strict setting."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the current weather in a given location."""
            location: str = Field(..., description="The city and state")

        model = ChatDeepSeek(
            model=MODEL_NAME,
            api_key=SecretStr("test-key"),
            strict=False,
        )

        # Override with strict=True in bind_tools
        model_with_tools = model.bind_tools([GetWeather], strict=True)

        # Check that strict was passed to kwargs
        assert 'tools' in model_with_tools.kwargs
        tools = model_with_tools.kwargs['tools']
        assert tools[0]['function']['strict'] is True
