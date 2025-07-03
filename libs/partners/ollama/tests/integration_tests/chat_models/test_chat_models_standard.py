"""Test chat model integration using standard integration tests."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ConnectError
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests
from ollama import ResponseError
from pydantic import ValidationError

from langchain_ollama.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": DEFAULT_MODEL_NAME}

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return False  # TODO: update after Ollama implements

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "Will sometime encounter AssertionErrors where tool responses are "
            "`'3'` instead of `3`"
        )
    )
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(
        reason=(
            "Will sometime encounter AssertionErrors where tool responses are "
            "`'3'` instead of `3`"
        )
    )
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @patch("langchain_ollama.chat_models.Client.list")
    def test_init_model_not_found(self, mock_list: MagicMock) -> None:
        """Test that a ValueError is raised when the model is not found."""
        mock_list.side_effect = ValueError("Test model not found")
        with pytest.raises(ValueError) as excinfo:
            ChatOllama(model="non-existent-model", validate_model_on_init=True)
        assert "Test model not found" in str(excinfo.value)

    @patch("langchain_ollama.chat_models.Client.list")
    def test_init_connection_error(self, mock_list: MagicMock) -> None:
        """Test that a ValidationError is raised on connect failure during init."""
        mock_list.side_effect = ConnectError("Test connection error")

        with pytest.raises(ValidationError) as excinfo:
            ChatOllama(model="any-model", validate_model_on_init=True)
        assert "not found in Ollama" in str(excinfo.value)

    @patch("langchain_ollama.chat_models.Client.list")
    def test_init_response_error(self, mock_list: MagicMock) -> None:
        """Test that a ResponseError is raised."""
        mock_list.side_effect = ResponseError("Test response error")

        with pytest.raises(ValidationError) as excinfo:
            ChatOllama(model="any-model", validate_model_on_init=True)
        assert "Received an error from the Ollama API" in str(excinfo.value)
