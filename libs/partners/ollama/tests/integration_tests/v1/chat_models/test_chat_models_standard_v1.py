"""Test chat model v1 integration using standard integration tests."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ConnectError
from langchain_core.messages.content_blocks import ToolCallChunk, is_reasoning_block
from langchain_core.tools import tool
from langchain_core.v1.chat_models import BaseChatModel
from langchain_core.v1.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_tests.integration_tests.chat_models_v1 import ChatModelV1IntegrationTests
from ollama import ResponseError
from pydantic import ValidationError

from langchain_ollama.v1.chat_models import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"
REASONING_MODEL_NAME = "deepseek-r1:1.5b"


@tool
def get_current_weather(location: str) -> dict:
    """Gets the current weather in a given location."""
    if "boston" in location.lower():
        return {"temperature": "15°F", "conditions": "snow"}
    return {"temperature": "unknown", "conditions": "unknown"}


class TestChatOllamaV1(ChatModelV1IntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": DEFAULT_MODEL_NAME}

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """ChatOllama supports reasoning content blocks."""
        return True

    @property
    def supports_image_content_blocks(self) -> bool:
        """ChatOllama supports image content blocks."""
        return True

    @property
    def has_tool_calling(self) -> bool:
        """ChatOllama supports tool calling."""
        return True

    @property
    def supports_invalid_tool_calls(self) -> bool:
        """ChatOllama supports invalid tool call handling."""
        return True

    @property
    def supports_non_standard_blocks(self) -> bool:
        """ChatOllama does not support non-standard content blocks."""
        return False

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # TODO: update after Ollama implements
        # https://github.com/ollama/ollama/blob/main/docs/openai.md#supported-request-fields
        return False

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
        """Test that the model can stream tool calls asynchronously."""
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
        reason=("Ollama does not yet support tool_choice forcing, may be unreliable")
    )
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        # TODO: shouldn't need to xfail this
        super().test_tool_calling_with_no_arguments(model)

    @pytest.mark.xfail(
        reason=(
            "Ollama does not support tool_choice forcing, agent loop may be unreliable"
        )
    )
    def test_agent_loop(self, model: BaseChatModel) -> None:
        super().test_agent_loop(model)

    @pytest.mark.xfail(
        reason=(
            "No single Ollama model supports both multimodal content and reasoning. "
            "Override skips test due to model limitations."
        )
    )
    def test_multimodal_reasoning(self, model: BaseChatModel) -> None:
        """Test complex reasoning with multiple content types.

        This test overrides the default model to use a reasoning-capable model
        with reasoning mode explicitly enabled. Note that this test requires
        both multimodal support AND reasoning support.
        """
        if not self.supports_multimodal_reasoning:
            pytest.skip("Model does not support multimodal reasoning.")

        pytest.skip(
            "TODO: Update this when we have a model that supports both multimodal and "
            "reasoning."
        )

    @pytest.mark.xfail(
        reason=(
            f"{DEFAULT_MODEL_NAME} does not support reasoning. Override uses "
            "reasoning-capable model with `reasoning=True` enabled."
        ),
        strict=False,
    )
    def test_reasoning_content_blocks_basic(self, model: BaseChatModel) -> None:
        """Test that the model can generate ``ReasoningContentBlock``.

        This test overrides the default model to use a reasoning-capable model
        with reasoning mode explicitly enabled.
        """
        if not self.supports_reasoning_content_blocks:
            pytest.skip("Model does not support ReasoningContentBlock.")

        reasoning_enabled_model = ChatOllama(
            model=REASONING_MODEL_NAME, reasoning=True, validate_model_on_init=True
        )

        message = HumanMessage("Think step by step: What is 2 + 2?")
        result = reasoning_enabled_model.invoke([message])
        assert isinstance(result, AIMessage)
        if isinstance(result.content, list):
            reasoning_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_reasoning_block(block)
            ]
            assert len(reasoning_blocks) > 0, (
                "Expected reasoning content blocks but found none. "
                f"Content blocks: {[block.get('type') for block in result.content]}"
            )

    # Additional Ollama reasoning tests in v1/chat_models/test_chat_models_v1.py

    @patch("langchain_ollama.v1.chat_models.Client.list")
    def test_init_model_not_found(self, mock_list: MagicMock) -> None:
        """Test that a ValueError is raised when the model is not found."""
        mock_list.side_effect = ValueError("Test model not found")
        with pytest.raises(ValueError) as excinfo:
            ChatOllama(model="non-existent-model", validate_model_on_init=True)
        assert "Test model not found" in str(excinfo.value)

    @patch("langchain_ollama.v1.chat_models.Client.list")
    def test_init_connection_error(self, mock_list: MagicMock) -> None:
        """Test that a ValidationError is raised on connect failure during init."""
        mock_list.side_effect = ConnectError("Test connection error")

        with pytest.raises(ValidationError) as excinfo:
            ChatOllama(model="any-model", validate_model_on_init=True)
        assert "Failed to connect to Ollama" in str(excinfo.value)

    @patch("langchain_ollama.v1.chat_models.Client.list")
    def test_init_response_error(self, mock_list: MagicMock) -> None:
        """Test that a ResponseError is raised."""
        mock_list.side_effect = ResponseError("Test response error")

        with pytest.raises(ValidationError) as excinfo:
            ChatOllama(model="any-model", validate_model_on_init=True)
        assert "Received an error from the Ollama API" in str(excinfo.value)
