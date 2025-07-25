"""Test chat model v1 format conversion."""

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from langchain_ollama._compat import (
    _convert_from_v1_message,
    _convert_to_v1_chunk,
    _convert_to_v1_message,
    _convert_unknown_content_block_to_non_standard,
)
from langchain_ollama.chat_models import (
    ChatOllama,
)
from langchain_ollama.chat_models import (
    _convert_unknown_content_block_to_non_standard as chat_convert_unknown,
)


class TestV1MessageConversion:
    """Test v1 message format conversion functions."""

    def test_convert_from_v1_message_with_text_content(self) -> None:
        """Test converting v1 message with text content to v0 format."""
        # Create a v1 message with TextContentBlock
        v1_message = AIMessage(content=[{"type": "text", "text": "Hello world"}])

        result = _convert_from_v1_message(v1_message)

        assert result.content == "Hello world"
        assert isinstance(result.content, str)

    def test_convert_from_v1_message_with_reasoning_content(self) -> None:
        """Test converting v1 message with reasoning content to v0 format."""

        v1_message = AIMessage(
            content=[
                {  # ReasoningContentBlock
                    "type": "reasoning",
                    "reasoning": "I need to think about this",
                },
                # TextContentBlock
                {"type": "text", "text": "Hello world"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        assert result.content == "Hello world"
        assert (
            result.additional_kwargs["reasoning_content"]
            == "I need to think about this"
        )

    def test_convert_from_v1_message_with_tool_call_content(self) -> None:
        """Test converting v1 message with tool call content to v0 format."""
        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Let me search for that"},
                {"type": "tool_call", "id": "tool_123"},  # ToolCallContentBlock
            ]
        )

        result = _convert_from_v1_message(v1_message)

        assert result.content == "Let me search for that"
        # Tool calls should be handled via tool_calls property, not content
        assert "tool_call" not in str(result.content)

    def test_convert_from_v1_message_with_mixed_content(self) -> None:
        """Test converting v1 message with mixed content types."""
        v1_message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Let me think"},
                {"type": "text", "text": "A"},
                {"type": "text", "text": "B"},
                {"type": "tool_call", "id": "tool_456"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        assert result.content == "AB"
        assert result.additional_kwargs["reasoning_content"] == "Let me think"

    def test_convert_from_v1_message_preserves_v0_format(self) -> None:
        """Test that v0 format messages are preserved unchanged."""
        v0_message = AIMessage(content="Hello world")

        result = _convert_from_v1_message(v0_message)

        assert result == v0_message
        assert result.content == "Hello world"

    def test_convert_to_v1_message_with_text_content(self) -> None:
        """Test converting v0 message with text content to v1 format."""
        v0_message = AIMessage(content="Hello world")

        result = _convert_to_v1_message(v0_message)

        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert result.content[0] == {"type": "text", "text": "Hello world"}

    def test_convert_to_v1_message_with_reasoning_content(self) -> None:
        """Test converting v0 message with reasoning to v1 format."""
        v0_message = AIMessage(
            content="Hello world",
            additional_kwargs={"reasoning_content": "I need to be helpful"},
        )

        result = _convert_to_v1_message(v0_message)

        assert isinstance(result.content, list)
        assert len(result.content) == 2
        expected_reasoning_block = {
            "type": "reasoning",
            "reasoning": "I need to be helpful",
        }
        assert result.content[0] == expected_reasoning_block
        assert result.content[1] == {"type": "text", "text": "Hello world"}
        assert "reasoning_content" not in result.additional_kwargs

    def test_convert_to_v1_message_with_tool_calls(self) -> None:
        """Test converting v0 message with tool calls to v1 format."""
        v0_message = AIMessage(
            content="Let me search for that",
            tool_calls=[
                {"name": "search", "args": {"query": "test"}, "id": "tool_123"}
            ],
        )

        result = _convert_to_v1_message(v0_message)

        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[0] == {"type": "text", "text": "Let me search for that"}
        assert result.content[1] == {"type": "tool_call", "id": "tool_123"}

    def test_convert_to_v1_message_with_empty_content(self) -> None:
        """Test converting v0 message with empty content to v1 format."""
        v0_message = AIMessage(content="")

        result = _convert_to_v1_message(v0_message)

        assert isinstance(result.content, list)
        assert len(result.content) == 0

    def test_convert_to_v1_chunk(self) -> None:
        """Test converting v0 chunk to v1 format."""
        v0_chunk = AIMessageChunk(content="Hello")

        result = _convert_to_v1_chunk(v0_chunk)

        assert isinstance(result, AIMessageChunk)
        assert isinstance(result.content, list)
        assert result.content == [{"type": "text", "text": "Hello"}]


class TestNonStandardContentBlockHandling:
    """Test handling of unknown content blocks via NonStandardContentBlock."""

    def test_convert_unknown_content_block_to_non_standard(self) -> None:
        """Test conversion of unknown content block to NonStandardContentBlock."""
        unknown_block = {
            "type": "custom_block_type",
            "custom_field": "some_value",
            "data": {"nested": "content"},
        }

        result = _convert_unknown_content_block_to_non_standard(unknown_block)

        assert result["type"] == "non_standard"
        assert result["value"] == unknown_block

    def test_chat_models_convert_unknown_content_block_to_non_standard(self) -> None:
        """Test conversion of unknown content block in chat_models module."""
        unknown_block = {
            "type": "audio_transcript",
            "transcript": "Hello world",
            "confidence": 0.95,
        }

        result = chat_convert_unknown(unknown_block)

        assert result["type"] == "non_standard"
        assert result["value"] == unknown_block

    def test_v1_content_with_unknown_type_creates_non_standard_block(self) -> None:
        """Test v1 content with unknown block type creates NonStandardContentBlock."""
        unknown_block = {
            "type": "future_block_type",
            "future_field": "future_value",
        }

        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                unknown_block,
                {"type": "text", "text": " world"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        # Unknown types should be converted to NonStandardContentBlock
        # and skipped in content processing, text content should be preserved
        assert result.content == "Hello world"

    def test_multiple_unknown_blocks_handled_gracefully(self) -> None:
        """Test multiple unknown content blocks are handled gracefully."""
        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Start"},
                {"type": "unknown_1", "data": "first"},
                {"type": "text", "text": " middle"},
                {"type": "unknown_2", "data": "second"},
                {"type": "text", "text": " end"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        # All unknown blocks should be converted and skipped
        assert result.content == "Start middle end"

    def test_non_standard_content_block_structure(self) -> None:
        """Test that NonStandardContentBlock has correct structure."""
        original_block = {
            "type": "custom_provider_block",
            "provider_specific_field": "value",
            "metadata": {"version": "1.0"},
        }

        non_standard_block = _convert_unknown_content_block_to_non_standard(
            original_block
        )

        # Verify it matches NonStandardContentBlock structure
        assert isinstance(non_standard_block, dict)
        assert non_standard_block["type"] == "non_standard"
        assert "value" in non_standard_block
        assert non_standard_block["value"] == original_block

    def test_empty_unknown_block_handling(self) -> None:
        """Test handling of empty unknown blocks."""
        empty_block = {"type": "empty_block"}

        result = _convert_unknown_content_block_to_non_standard(empty_block)

        assert result["type"] == "non_standard"
        assert result["value"] == empty_block


class TestChatOllamaV1Integration:
    """Test ChatOllama integration with v1 format."""

    def test_chat_ollama_default_output_version(self) -> None:
        """Test that ChatOllama defaults to v0 output format."""
        llm = ChatOllama(model="test-model")
        assert llm.output_version == "v0"

    def test_chat_ollama_v1_output_version_setting(self) -> None:
        """Test setting ChatOllama to v1 output format."""
        llm = ChatOllama(model="test-model", output_version="v1")
        assert llm.output_version == "v1"

    def test_convert_messages_handles_v1_input(self) -> None:
        """Test that _convert_messages_to_ollama_messages handles v1 input."""
        llm = ChatOllama(model="test-model", output_version="v0")

        # Create a v1 format AIMessage
        v1_message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Let me think"},
                {"type": "text", "text": "Hello world"},
            ]
        )

        messages = [HumanMessage("Hi"), v1_message]
        result = llm._convert_messages_to_ollama_messages(messages)

        # Should have processed both messages
        assert len(result) == 2

        # The AI message should have been converted to v0 format for Ollama API
        ai_msg = result[1]
        assert ai_msg["content"] == "Hello world"

    def test_convert_messages_preserves_v0_input(self) -> None:
        """Test that _convert_messages_to_ollama_messages preserves v0 input."""
        llm = ChatOllama(model="test-model", output_version="v0")

        messages = [HumanMessage("Hi"), AIMessage(content="Hello world")]
        result = llm._convert_messages_to_ollama_messages(messages)

        # Should have processed both messages normally
        assert len(result) == 2
        assert result[1]["content"] == "Hello world"


class TestV1BackwardsCompatibility:
    """Test backwards compatibility with v0 format."""

    def test_v0_messages_unchanged_with_v0_output(self) -> None:
        """Test that v0 messages are unchanged when output_version=v0."""
        llm = ChatOllama(model="test-model", output_version="v0")

        # This test would require mocking the actual ollama calls
        # TODO complete this test with a mock or fixture
        assert llm.output_version == "v0"

    def test_v1_input_works_with_v0_output(self) -> None:
        """Test that v1 input messages work even when output_version=v0."""
        llm = ChatOllama(model="test-model", output_version="v0")

        v1_message = AIMessage(content=[{"type": "text", "text": "Hello"}])

        messages: list[BaseMessage] = [v1_message]
        result = llm._convert_messages_to_ollama_messages(messages)

        # Should handle v1 input regardless of output_version setting
        assert len(result) == 1
        assert result[0]["content"] == "Hello"


class TestV1EdgeCases:
    """Test edge cases in v1 format handling."""

    def test_empty_v1_content_list(self) -> None:
        """Test handling empty v1 content list."""
        v1_message = AIMessage(content=[])

        result = _convert_from_v1_message(v1_message)

        assert result.content == ""

    def test_v1_content_with_unknown_type(self) -> None:
        """Test v1 content with unknown block type converted to NonStandard."""
        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "unknown_type", "data": "converted_to_non_standard"},
                {"type": "text", "text": " world"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        # Unknown types should be converted to NonStandardContentBlock and skipped
        # in content processing, text content should be preserved
        assert result.content == "Hello world"

    def test_v1_content_with_malformed_blocks(self) -> None:
        """Test v1 content with malformed blocks."""
        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text"},  # Missing 'text' field
                {"type": "reasoning"},  # Missing 'reasoning' field
                {"type": "text", "text": " world"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        # Should handle malformed blocks gracefully
        assert result.content == "Hello world"

    def test_non_dict_blocks_ignored(self) -> None:
        """Test that non-dict items in content list are ignored."""
        v1_message = AIMessage(
            content=[
                {"type": "text", "text": "Hello"},
                "invalid_block",  # Not a dict
                {"type": "text", "text": " world"},
            ]
        )

        result = _convert_from_v1_message(v1_message)

        assert result.content == "Hello world"
