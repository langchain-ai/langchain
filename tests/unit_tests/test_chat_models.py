from typing import Any, Dict, Literal
from typing_extensions import TypeAlias

from langchain_core.messages import AIMessageChunk
from langchain_tests.unit_tests import ChatModelUnitTests
from openai import BaseModel
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from pydantic import SecretStr
from pydantic.fields import FieldInfo
from unittest.mock import MagicMock

from langchain_deepseek.chat_models import ChatDeepSeek

IncEx: TypeAlias = "set[int] | set[str] | dict[int, Any] | dict[str, Any] | None"

class MockOpenAIResponse:
    def __init__(self, choices: list, error: None = None):
        self.choices = choices
        self.error = error

    def model_dump(self) -> Dict[str, Any]:
        choices_list: list[dict[str, Any]] = []
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


class TestChatDeepSeekUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatDeepSeek]:
        return ChatDeepSeek

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "DEEPSEEK_API_KEY": "api_key",
                "DEEPSEEK_API_BASE": "api_base",
            },
            {
                "model": "deepseek-chat",
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "deepseek-chat",
            "api_key": "api_key",
        }


class TestChatDeepSeekCustomUnit:
    """Custom tests specific to DeepSeek chat model."""

    def test_create_chat_result_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from response."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        mock_message = MagicMock()
        mock_message.content = "Main content"
        mock_message.reasoning_content = "This is the reasoning content"
        mock_message.role = "assistant"
        mock_response = MockOpenAIResponse(
            choices=[MagicMock(message=mock_message)], error=None
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning_content")
            == "This is the reasoning content"
        )

    def test_create_chat_result_with_model_extra_reasoning(self) -> None:
        """Test that reasoning is properly extracted from model_extra."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        message = ChatCompletionMessage(
            role="assistant", 
            content="Main content",
            model_extra={"reasoning": "This is the reasoning"}
        )
        choice = Choice(message=message, finish_reason="stop", index=0)
        mock_response = ChatCompletion(
            id="test",
            choices=[choice],
            created=1234567890,
            model="deepseek-chat",
            object="chat.completion",
        )

        result = chat_model._create_chat_result(mock_response)
        assert (
            result.generations[0].message.additional_kwargs.get("reasoning")
            == "This is the reasoning"
        )

    def test_convert_chunk_with_reasoning_content(self) -> None:
        """Test that reasoning_content is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        chunk: Dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning_content": "Streaming reasoning content",
                    }
                }
            ]
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk, AIMessageChunk, None
        )
        assert chunk_result is not None
        assert (
            chunk_result.message.additional_kwargs.get("reasoning_content")
            == "Streaming reasoning content"
        )

    def test_convert_chunk_with_reasoning(self) -> None:
        """Test that reasoning is properly extracted from streaming chunk."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        chunk: Dict[str, Any] = {
            "choices": [
                {
                    "delta": {
                        "content": "Main content",
                        "reasoning": "Streaming reasoning",
                    }
                }
            ]
        }

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk, AIMessageChunk, None
        )
        assert chunk_result is not None
        assert (
            chunk_result.message.additional_kwargs.get("reasoning")
            == "Streaming reasoning"
        )

    def test_convert_chunk_without_reasoning(self) -> None:
        """Test that chunk without reasoning fields works correctly."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        chunk: Dict[str, Any] = {"choices": [{"delta": {"content": "Main content"}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk, AIMessageChunk, None
        )
        assert chunk_result is not None
        assert chunk_result.message.additional_kwargs.get("reasoning") is None
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None

    def test_convert_chunk_with_empty_delta(self) -> None:
        """Test that chunk with empty delta works correctly."""
        chat_model = ChatDeepSeek(model="deepseek-chat", api_key=SecretStr("api_key"))
        chunk: Dict[str, Any] = {"choices": [{"delta": {}}]}

        chunk_result = chat_model._convert_chunk_to_generation_chunk(
            chunk, AIMessageChunk, None
        )
        assert chunk_result is not None
        assert chunk_result.message.additional_kwargs.get("reasoning") is None
        assert chunk_result.message.additional_kwargs.get("reasoning_content") is None