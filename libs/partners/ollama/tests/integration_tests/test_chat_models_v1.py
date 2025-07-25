"""Integration tests for ChatOllama v1 format functionality."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_ollama import ChatOllama

DEFAULT_MODEL_NAME = "llama3.1"
REASONING_MODEL_NAME = "deepseek-r1:8b"


@pytest.mark.requires("ollama")
class TestChatOllamaV1Integration:
    """Integration tests for ChatOllama v1 format functionality."""

    def test_v1_output_format_basic_chat(self) -> None:
        """Test basic chat functionality with v1 output format."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        message = HumanMessage("Say hello")
        result = llm.invoke([message])

        # Result should be in v1 format (content as list)
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)
        assert len(result.content) > 0

        # Should have at least one TextContentBlock
        text_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        assert len(text_blocks) > 0
        assert "text" in text_blocks[0]

    def test_v1_output_format_streaming(self) -> None:
        """Test streaming functionality with v1 output format."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        message = HumanMessage("Count to 3")
        chunks = list(llm.stream([message]))

        # All chunks should be in v1 format
        for chunk in chunks:
            assert isinstance(chunk.content, list)
            # Each chunk should have content blocks
            if chunk.content:  # Some chunks might be empty
                for block in chunk.content:
                    assert isinstance(block, dict)
                    assert "type" in block

    def test_v1_input_with_v0_output(self) -> None:
        """Test that v1 input works with v0 output format."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v0")

        # Send v1 format message as input
        v1_message = AIMessage(
            content=[{"type": "text", "text": "Hello, how are you?"}]
        )
        human_message = HumanMessage("Fine, thanks!")

        result = llm.invoke([v1_message, human_message])

        # Output should be in v0 format (content as string)
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)

    def test_v1_input_with_v1_output(self) -> None:
        """Test that v1 input works with v1 output format."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        # Send v1 format message as input
        v1_message = AIMessage(
            content=[{"type": "text", "text": "Hello, how are you?"}]
        )
        human_message = HumanMessage("Fine, thanks!")

        result = llm.invoke([v1_message, human_message])

        # Output should be in v1 format (content as list)
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)

    def test_v0_input_with_v1_output(self) -> None:
        """Test that v0 input works with v1 output format."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        # Send v0 format message as input
        v0_message = AIMessage(content="Hello, how are you?")
        human_message = HumanMessage("Fine, thanks!")

        result = llm.invoke([v0_message, human_message])

        # Output should be in v1 format (content as list)
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)

    @pytest.mark.parametrize("output_version", ["v0", "v1"])
    def test_mixed_message_formats_input(self, output_version: str) -> None:
        """Test handling mixed v0 and v1 format messages in input.

        Rare case but you never know...
        """
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version=output_version)

        messages = [
            HumanMessage("Hello"),
            AIMessage(content="Hi there!"),  # v0 format
            HumanMessage("How are you?"),
            # v1 format
            AIMessage(content=[{"type": "text", "text": "I'm doing well!"}]),
            HumanMessage("Great!"),
        ]

        result = llm.invoke(messages)

        # Output format should match output_version setting
        assert isinstance(result, AIMessage)
        if output_version == "v0":
            assert isinstance(result.content, str)
        else:
            assert isinstance(result.content, list)


@pytest.mark.requires("ollama")
class TestChatOllamaV1WithReasoning:
    """Integration tests for ChatOllama v1 format with reasoning functionality."""

    def test_v1_output_with_reasoning_enabled(self) -> None:
        """Test v1 output format with reasoning enabled."""
        # Note: This test requires a reasoning-capable model
        llm = ChatOllama(
            model=REASONING_MODEL_NAME,
            output_version="v1",
            reasoning=True,
        )

        message = HumanMessage("What is 2+2? Think step by step.")
        result = llm.invoke([message])

        # Result should be in v1 format with reasoning block
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)

        # Should have both reasoning and text blocks
        reasoning_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "reasoning"
        ]
        text_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]

        # Should have reasoning content when reasoning=True
        assert len(reasoning_blocks) > 0 or len(text_blocks) > 0

        # Should be able to use the reasoning property on the AIMessage
        # TODO

    def test_v1_input_with_reasoning_content(self) -> None:
        """Test v1 input format with reasoning content blocks."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        # Send message with reasoning content
        reasoning_message = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "I need to be helpful and accurate"},
                {"type": "text", "text": "I'll do my best to help you."},
            ]
        )
        human_message = HumanMessage("Thank you!")

        result = llm.invoke([reasoning_message, human_message])

        # Should process the input correctly and return v1 format
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)


@pytest.mark.requires("ollama")
class TestChatOllamaV1WithTools:
    """Integration tests for ChatOllama v1 format with tool calling."""

    def test_v1_output_with_tool_calls(self) -> None:
        """Test v1 output format with tool calls."""
        from langchain_core.tools import tool

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"The weather in {location} is sunny."

        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")
        llm_with_tools = llm.bind_tools([get_weather])

        message = HumanMessage("What's the weather in Paris?")
        result = llm_with_tools.invoke([message])

        # Result should be in v1 format
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)

        # If tool calls were made, should have tool_call blocks
        if result.tool_calls:
            tool_call_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and block.get("type") == "tool_call"
            ]
            assert len(tool_call_blocks) == len(result.tool_calls)

    def test_v1_input_with_tool_call_blocks(self) -> None:
        """Test v1 input format with tool call content blocks."""
        llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version="v1")

        # Send message with tool call content
        tool_message = AIMessage(
            content=[
                {"type": "text", "text": "I'll check the weather for you."},
                {"type": "tool_call", "id": "call_123"},
            ],
            tool_calls=[
                {"name": "get_weather", "args": {"location": "Paris"}, "id": "call_123"}
            ],
        )
        human_message = HumanMessage("Thanks!")

        result = llm.invoke([tool_message, human_message])

        # Should process the input correctly and return v1 format
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)


@pytest.mark.requires("ollama")
class TestV1BackwardsCompatibility:
    """Test backwards compatibility when using v1 format."""

    def test_existing_v0_code_still_works(self) -> None:
        """Test that existing v0 code continues to work unchanged."""
        # This is the default behavior - should not break existing code
        llm = ChatOllama(model=DEFAULT_MODEL_NAME)  # defaults to v0

        message = HumanMessage("Hello")
        result = llm.invoke([message])

        # Should return v0 format (string content)
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)

    def test_gradual_migration_v0_to_v1(self) -> None:
        """Test gradual migration from v0 to v1 format."""
        # Test that the same code works with both formats
        for output_version in ["v0", "v1"]:
            llm = ChatOllama(model=DEFAULT_MODEL_NAME, output_version=output_version)

            message = HumanMessage("Hello")
            result = llm.invoke([message])

            assert isinstance(result, AIMessage)
            if output_version == "v0":
                assert isinstance(result.content, str)
            else:
                assert isinstance(result.content, list)
                # Should have at least one content block
                assert len(result.content) > 0
