"""Test chat model integration using standard integration tests."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ConnectError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolCallChunk
from langchain_core.tools import tool
from langchain_tests.integration_tests import ChatModelIntegrationTests
from ollama import ResponseError
from pydantic import ValidationError

from langchain_ollama.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"


@tool
def get_current_weather(location: str) -> dict:
    """Gets the current weather in a given location."""
    if "boston" in location.lower():
        return {"temperature": "15°F", "conditions": "snow"}
    return {"temperature": "unknown", "conditions": "unknown"}


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
        # TODO: update after Ollama implements
        # https://github.com/ollama/ollama/blob/main/docs/openai.md#supported-request-fields
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return True

    def test_tool_streaming(self, model: BaseChatModel) -> None:
        """Test that the model can stream tool calls."""
        chat_model_with_tools = model.bind_tools([get_current_weather])

        prompt = [HumanMessage("What is the weather today in Boston?")]

        # Flags and collectors for validation
        tool_chunk_found = False
        final_tool_calls = []
        collected_tool_chunks: list[ToolCallChunk] = []

        # Stream the response and inspect the chunks
        for chunk in chat_model_with_tools.stream(prompt):
            assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

            if chunk.tool_call_chunks:
                tool_chunk_found = True
                collected_tool_chunks.extend(chunk.tool_call_chunks)

            if chunk.tool_calls:
                final_tool_calls.extend(chunk.tool_calls)

        assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
        assert len(final_tool_calls) == 1, (
            f"Expected 1 final tool call, but got {len(final_tool_calls)}"
        )

        final_tool_call = final_tool_calls[0]
        assert final_tool_call["name"] == "get_current_weather"
        assert final_tool_call["args"] == {"location": "Boston"}

        assert len(collected_tool_chunks) > 0
        assert collected_tool_chunks[0]["name"] == "get_current_weather"

        # The ID should be consistent across chunks that have it
        tool_call_id = collected_tool_chunks[0].get("id")
        assert tool_call_id is not None
        assert all(
            chunk.get("id") == tool_call_id
            for chunk in collected_tool_chunks
            if chunk.get("id")
        )
        assert final_tool_call["id"] == tool_call_id

    async def test_tool_astreaming(self, model: BaseChatModel) -> None:
        """Test that the model can stream tool calls."""
        chat_model_with_tools = model.bind_tools([get_current_weather])

        prompt = [HumanMessage("What is the weather today in Boston?")]

        # Flags and collectors for validation
        tool_chunk_found = False
        final_tool_calls = []
        collected_tool_chunks: list[ToolCallChunk] = []

        # Stream the response and inspect the chunks
        async for chunk in chat_model_with_tools.astream(prompt):
            assert isinstance(chunk, AIMessageChunk), "Expected AIMessageChunk type"

            if chunk.tool_call_chunks:
                tool_chunk_found = True
                collected_tool_chunks.extend(chunk.tool_call_chunks)

            if chunk.tool_calls:
                final_tool_calls.extend(chunk.tool_calls)

        assert tool_chunk_found, "Tool streaming did not produce any tool_call_chunks."
        assert len(final_tool_calls) == 1, (
            f"Expected 1 final tool call, but got {len(final_tool_calls)}"
        )

        final_tool_call = final_tool_calls[0]
        assert final_tool_call["name"] == "get_current_weather"
        assert final_tool_call["args"] == {"location": "Boston"}

        assert len(collected_tool_chunks) > 0
        assert collected_tool_chunks[0]["name"] == "get_current_weather"

        # The ID should be consistent across chunks that have it
        tool_call_id = collected_tool_chunks[0].get("id")
        assert tool_call_id is not None
        assert all(
            chunk.get("id") == tool_call_id
            for chunk in collected_tool_chunks
            if chunk.get("id")
        )
        assert final_tool_call["id"] == tool_call_id

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
        assert "Failed to connect to Ollama" in str(excinfo.value)

    @patch("langchain_ollama.chat_models.Client.list")
    def test_init_response_error(self, mock_list: MagicMock) -> None:
        """Test that a ResponseError is raised."""
        mock_list.side_effect = ResponseError("Test response error")

        with pytest.raises(ValidationError) as excinfo:
            ChatOllama(model="any-model", validate_model_on_init=True)
        assert "Received an error from the Ollama API" in str(excinfo.value)
