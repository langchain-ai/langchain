"""Unit tests for ChatOllama v1 format support."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langchain_core.messages.block_translators.ollama import (
    translate_content,
    translate_content_chunk,
)
from langchain_core.utils.utils import LC_AUTO_PREFIX

from langchain_ollama.chat_models import ChatOllama

MODEL_NAME = "llama3.1"


class TestV1BlockTranslator:
    """Test block translator functions."""

    def test_translate_content_text_only(self) -> None:
        """Test translation of text message to v1 format."""

        message = AIMessage(
            content="Hello, world!",
        )

        blocks = translate_content(message)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello, world!"
        assert "id" not in blocks[0]  # ID should not be added during translation

    def test_translate_content_with_reasoning(self) -> None:
        """Test translation of message with reasoning to v1 format."""

        message = AIMessage(
            content="The answer is 42.",
            additional_kwargs={"reasoning_content": "Let me think about this..."},
        )

        blocks = translate_content(message)

        assert len(blocks) == 2

        # Reasoning should come before main content
        reasoning_block = blocks[0]
        assert reasoning_block["type"] == "reasoning"
        assert reasoning_block.get("reasoning") == "Let me think about this..."
        assert "id" not in reasoning_block

        text_block = blocks[1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "The answer is 42."

    def test_translate_content_with_tool_calls(self) -> None:
        """Test translation of message with tool calls to v1 format."""

        tool_call = {
            "name": "multiply",
            "args": {"a": 3, "b": 4},
            "id": "call_123",
        }

        message = AIMessage(
            content="I'll multiply these numbers.",
            tool_calls=[tool_call],
        )

        blocks = translate_content(message)

        assert len(blocks) == 2

        text_block = blocks[0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "I'll multiply these numbers."

        tool_call_block = blocks[1]
        assert tool_call_block["type"] == "tool_call"
        assert tool_call_block["name"] == "multiply"
        assert tool_call_block["args"] == {"a": 3, "b": 4}
        assert tool_call_block["id"] == "call_123"

    def test_translate_content_chunk(self) -> None:
        """Test translation of chunk to v1 format."""

        chunk = AIMessageChunk(
            content="Hello",
            additional_kwargs={"reasoning_content": "Thinking..."},
        )

        blocks = translate_content_chunk(chunk)

        assert len(blocks) == 2

        reasoning_block = blocks[0]
        assert reasoning_block["type"] == "reasoning"
        assert reasoning_block.get("reasoning") == "Thinking..."

        text_block = blocks[1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Hello"


class TestV1ChatOllama:
    """Test ChatOllama with v1 format functionality."""

    def test_v1_parameter(self) -> None:
        llm = ChatOllama(model=MODEL_NAME)
        assert llm.output_version == "v0"

        llm = ChatOllama(model=MODEL_NAME, output_version="v1")
        assert llm.output_version == "v1"

    def test_v0_output_format(self) -> None:
        """Test that previous output format is retained."""
        mock_response = [
            {
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!",
                    "thinking": "I should greet the user.",
                },
                "done": True,
                "done_reason": "stop",
            }
        ]

        with patch("langchain_ollama.chat_models.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response

            llm = ChatOllama(model=MODEL_NAME, reasoning=True)
            result = llm.invoke([HumanMessage("Hello")])

            # Should be v0 format - reasoning in additional_kwargs
            assert isinstance(result.content, str)
            assert result.content == "Hello, world!"
            reasoning_content = result.additional_kwargs.get("reasoning_content")
            assert reasoning_content == "I should greet the user."

            # Test using `.content_blocks`
            content_blocks = result.content_blocks
            assert isinstance(content_blocks, list)
            assert len(content_blocks) == 2

            # First block should be reasoning
            assert content_blocks[0]["type"] == "reasoning"
            assert content_blocks[0].get("reasoning") == "I should greet the user."
            # assert "reasoning_content" not in result.additional_kwargs TODO: use this?

            assert content_blocks[1]["type"] == "text"
            assert content_blocks[1]["text"] == "Hello, world!"

            # ID should not be added unless flag is v1
            assert "id" not in content_blocks[1]

    def test_v1_output_format(self) -> None:
        mock_response = [
            {
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!",
                    "thinking": "I should greet the user.",
                },
                "done": True,
                "done_reason": "stop",
            }
        ]

        with patch("langchain_ollama.chat_models.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response

            llm = ChatOllama(model=MODEL_NAME, output_version="v1", reasoning=True)
            result = llm.invoke([HumanMessage("Hello")])

            # Should be stored in v1 format (content blocks) since flag is set
            assert isinstance(result.content, list)
            assert len(result.content) == 2

            reasoning_block = result.content[0]
            assert isinstance(reasoning_block, dict)
            assert reasoning_block["type"] == "reasoning"
            assert reasoning_block["reasoning"] == "I should greet the user."
            assert "id" in reasoning_block
            assert reasoning_block["id"] is not None
            assert type(reasoning_block["id"]) is str
            assert reasoning_block["id"].startswith(LC_AUTO_PREFIX)

            text_block = result.content[1]
            assert isinstance(text_block, dict)
            assert text_block["type"] == "text"
            assert text_block["text"] == "Hello, world!"
            assert "id" in text_block
            assert text_block["id"] is not None
            assert type(text_block["id"]) is str
            assert text_block["id"].startswith(LC_AUTO_PREFIX)

            assert "reasoning_content" not in result.additional_kwargs

    def test_v1_streaming_output_format(self) -> None:
        """Test that v1 format works with streaming."""
        mock_responses = [
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            },
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": ", world!"},
                "done": True,
                "done_reason": "stop",
            },
        ]

        with patch("langchain_ollama.chat_models.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_responses

            llm = ChatOllama(model=MODEL_NAME, output_version="v1")
            chunks = list(llm.stream([HumanMessage("Hello")]))

            for chunk in chunks:
                assert isinstance(chunk.content, list)
                if chunk.content:  # Skip empty chunks
                    content_block = chunk.content[0]
                    assert isinstance(content_block, dict)
                    assert content_block["type"] == "text"

                    # TODO: should chunks have auto-generated IDs? what to do here?

    def test_v1_with_tool_calls(self) -> None:
        """Test v1 format with tool calls."""
        mock_response = [
            {
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "multiply",
                                "arguments": {"a": 3, "b": 4},
                            }
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
            }
        ]

        with patch("langchain_ollama.chat_models.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.chat.return_value = mock_response

            llm = ChatOllama(model=MODEL_NAME, output_version="v1")
            result = llm.invoke([HumanMessage("Multiply 3 and 4")])

            assert isinstance(result.content, list)
            tool_call_block = None
            for block in result.content:
                if isinstance(block, dict) and block.get("type") == "tool_call":
                    tool_call_block = block
                    break
            assert tool_call_block is not None
            assert tool_call_block["name"] == "multiply"
            assert tool_call_block["args"] == {"a": 3, "b": 4}

    async def test_v1_async_generation(self) -> None:
        """Test v1 format with async generation."""
        mock_response = [
            {
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "Hello, async world!",
                    "thinking": "Async thinking...",
                },
                "done": True,
                "done_reason": "stop",
            }
        ]

        with patch("langchain_ollama.chat_models.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # The chat method should return a coroutine that yields the async iterator
            async def async_chat_coroutine(*args: Any, **kwargs: Any) -> Any:
                async def async_generator() -> Any:
                    for response in mock_response:
                        yield response

                return async_generator()

            mock_client.chat = async_chat_coroutine

            llm = ChatOllama(model=MODEL_NAME, output_version="v1", reasoning=True)
            result = await llm.ainvoke([HumanMessage("Hello")])

            # Should be v1 format
            assert isinstance(result.content, list)
            assert len(result.content) == 2

            reasoning_block = result.content[0]
            assert isinstance(reasoning_block, dict)
            assert reasoning_block["type"] == "reasoning"
            assert reasoning_block["reasoning"] == "Async thinking..."

            text_block = result.content[1]
            assert isinstance(text_block, dict)
            assert text_block["type"] == "text"
            assert text_block["text"] == "Hello, async world!"

    async def test_v1_async_streaming(self) -> None:
        """Test v1 format with async streaming."""
        mock_responses = [
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": "Async"},
                "done": False,
            },
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": " streaming!"},
                "done": True,
                "done_reason": "stop",
            },
        ]

        with patch("langchain_ollama.chat_models.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            async def async_chat_coroutine(*args: Any, **kwargs: Any) -> Any:
                async def async_generator() -> Any:
                    for response in mock_responses:
                        yield response

                return async_generator()

            mock_client.chat = async_chat_coroutine

            llm = ChatOllama(model=MODEL_NAME, output_version="v1")
            chunks = [chunk async for chunk in llm.astream([HumanMessage("Hello")])]

            # Each chunk should have v1 format
            for chunk in chunks:
                # chunk is an AIMessageChunk directly
                assert isinstance(chunk.content, list)
                if chunk.content:  # Skip empty chunks
                    content_block = chunk.content[0]
                    assert isinstance(content_block, dict)
                    assert content_block["type"] == "text"


class TestV1EdgeCases:
    """Test edge cases for v1 format support."""

    def test_empty_content_handling(self) -> None:
        """Test handling of empty content."""

        message = AIMessage(content="", additional_kwargs={})
        blocks = translate_content(message)

        assert isinstance(blocks, list)
        assert len(blocks) == 0

    def test_list_content_passthrough(self) -> None:
        """Test that existing list content is handled properly."""

        message = AIMessage(content=[{"type": "text", "text": "Already a list"}])
        blocks = translate_content(message)

        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Already a list"

        # IDs should be preserved if they exist
        message = AIMessage(
            content=[{"type": "text", "text": "Already a list", "id": "existing_id"}]
        )
        blocks = translate_content(message)

        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0].get("id") == "existing_id"

    def test_multimodal_content_preservation(self) -> None:
        """Test that multimodal content is preserved in v1 conversion."""

        image_block = {"type": "image", "url": "https://example.com/image.png"}

        message = AIMessage(content=[image_block])
        blocks = translate_content(message)

        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0] == image_block

    def test_unknown_content_block_preservation(self) -> None:
        """Test that unknown content block types are preserved."""
        # TODO: check this, shouldn't this become NonStandardContentBlock?

        unknown_block = {"type": "custom_block", "custom_field": "custom_value"}

        message = AIMessage(content=[unknown_block])
        blocks = translate_content(message)

        assert isinstance(blocks, list)
        assert len(blocks) == 1
        assert blocks[0] == unknown_block
