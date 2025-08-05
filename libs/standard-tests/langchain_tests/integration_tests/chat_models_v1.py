"""Integration tests for v1 chat models.

This module provides comprehensive integration tests for the new messages and standard
content block system introduced in ``langchain_core.messages.content_blocks``.
"""

from typing import Any, Union, cast

import langchain_core.messages.content_blocks as types
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.content_blocks import (
    AudioContentBlock,
    Citation,
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
    FileContentBlock,
    ImageContentBlock,
    InvalidToolCall,
    NonStandardContentBlock,
    PlainTextContentBlock,
    ReasoningContentBlock,
    TextContentBlock,
    ToolCall,
    ToolCallChunk,
    VideoContentBlock,
    WebSearchCall,
    WebSearchResult,
    create_audio_block,
    create_image_block,
    create_plaintext_block,
    create_text_block,
    create_video_block,
    is_reasoning_block,
    is_text_block,
    is_tool_call_block,
)
from langchain_core.tools import tool
from langchain_core.v1.chat_models import BaseChatModel
from langchain_core.v1.messages import AIMessage, AIMessageChunk, HumanMessage

from langchain_tests.unit_tests.chat_models_v1 import ChatModelV1Tests

# Content block type definitions for testing
ContentBlock = Union[
    TextContentBlock,
    ImageContentBlock,
    VideoContentBlock,
    AudioContentBlock,
    PlainTextContentBlock,
    FileContentBlock,
    ReasoningContentBlock,
    NonStandardContentBlock,
    ToolCall,
    InvalidToolCall,
    ToolCallChunk,
    WebSearchCall,
    WebSearchResult,
    Citation,
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
]


def _get_test_image_base64() -> str:
    """Get a small test image as base64 for testing."""
    # 1x1 pixel transparent PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # noqa: E501


def _get_test_audio_base64() -> str:
    """Get a small test audio file as base64 for testing."""
    # Minimal WAV file (1 second of silence)
    return (
        "UklGRjIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQ4AAAAAAAAAAAAAAAAAAA=="
    )


def _get_test_video_base64() -> str:
    """Get a small test video file as base64 for testing."""
    # Minimal valid video file would be much larger; for testing we use a placeholder
    return "PLACEHOLDER_VIDEO_DATA"


def _validate_tool_call_message(message: AIMessage) -> None:
    """Validate that a message contains tool calls in content blocks format."""

    if isinstance(message.content, list):
        # Check for tool calls in content blocks
        tool_call_blocks = [
            block
            for block in message.content
            if isinstance(block, dict) and is_tool_call_block(block)
        ]
        assert len(tool_call_blocks) >= 1

        tool_call = tool_call_blocks[0]
        assert "name" in tool_call
        assert "args" in tool_call
        assert "id" in tool_call
    # TODO: review if this is necessary
    # else:
    #     # Fallback to legacy tool_calls attribute
    #     assert hasattr(message, "tool_calls")
    #     assert len(message.tool_calls) >= 1


def _validate_multimodal_content_blocks(
    message: BaseMessage, expected_types: list[str]
) -> None:
    """Validate that a message contains expected content block types."""
    assert isinstance(message, (HumanMessage, AIMessage))
    assert isinstance(message.content, list)

    found_types = []
    for block in message.content:
        if isinstance(block, dict) and "type" in block:
            found_types.append(block["type"])

    for type_ in expected_types:
        assert type_ in found_types, f"Expected content block type '{type_}' not found"


class ChatModelV1IntegrationTests(ChatModelV1Tests):
    """Integration tests for v1 chat models with standard content blocks support.

    Inherits from ``ChatModelV1Tests`` to provide comprehensive testing of content
    block functionality with real external services.
    """

    # Additional multimodal support properties for integration testing
    @property
    def supports_multimodal_reasoning(self) -> bool:
        """Whether the model can reason about multimodal content."""
        return (
            self.supports_image_content_blocks
            and self.supports_reasoning_content_blocks
        )

    @property
    def supports_code_interpreter(self) -> bool:
        """Whether the model supports code interpreter blocks."""
        return False

    @property
    def supports_structured_citations(self) -> bool:
        """Whether the model supports structured citation generation."""
        return self.supports_citations

    @property
    def requires_api_key(self) -> bool:
        """Whether integration tests require an API key."""
        return True

    # Multimodal testing
    def test_image_content_blocks_with_analysis(self, model: BaseChatModel) -> None:
        """Test image analysis using ``ImageContentBlock``s."""
        if not self.supports_image_content_blocks:
            pytest.skip("Model does not support image inputs.")

        image_block = create_image_block(
            base64=_get_test_image_base64(),
            mime_type="image/png",
        )
        text_block = create_text_block("Analyze this image in detail.")

        result = model.invoke([HumanMessage([text_block, image_block])])

        assert isinstance(result, AIMessage)
        text_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and is_text_block(block)
        ]
        assert len(text_blocks) > 0
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_video_content_blocks(self, model: BaseChatModel) -> None:
        """Test video content block processing."""
        if not self.supports_video_content_blocks:
            pytest.skip("Model does not support video inputs.")

        video_block = create_video_block(
            base64=_get_test_video_base64(),
            mime_type="video/mp4",
        )
        text_block = create_text_block("Describe what you see in this video.")

        result = model.invoke([HumanMessage([text_block, video_block])])

        assert isinstance(result, AIMessage)
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_audio_content_blocks_processing(self, model: BaseChatModel) -> None:
        """Test audio content block processing with transcription."""
        if not self.supports_audio_content_blocks:
            pytest.skip("Model does not support audio inputs.")

        audio_block = create_audio_block(
            base64=_get_test_audio_base64(),
            mime_type="audio/wav",
        )
        text_block = create_text_block("Transcribe this audio file.")

        result = model.invoke([HumanMessage([text_block, audio_block])])

        assert isinstance(result, AIMessage)
        if result.text:
            assert len(result.text) > 10  # Substantial response

    def test_complex_multimodal_reasoning(self, model: BaseChatModel) -> None:
        """Test complex reasoning with multiple content types."""
        # TODO: come back to this, seems like a unique scenario
        if not self.supports_multimodal_reasoning:
            pytest.skip("Model does not support multimodal reasoning.")

        content_blocks: list[ContentBlock] = [
            create_text_block(
                "Compare these media files and provide reasoning analysis:"
            ),
            create_image_block(
                base64=_get_test_image_base64(),
                mime_type="image/png",
            ),
        ]

        if self.supports_audio_content_blocks:
            content_blocks.append(
                create_audio_block(
                    base64=_get_test_audio_base64(),
                    mime_type="audio/wav",
                )
            )

        message = HumanMessage(content=cast("list[types.ContentBlock]", content_blocks))
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for reasoning blocks in response
        if self.supports_reasoning_content_blocks:
            reasoning_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_reasoning_block(block)
            ]
            assert len(reasoning_blocks) > 0

    def test_citation_generation_with_sources(self, model: BaseChatModel) -> None:
        """Test that the model can generate ``Citations`` with source links."""
        if not self.supports_structured_citations:
            pytest.skip("Model does not support structured citations.")

        message = HumanMessage(
            "Provide factual information about the distance to the moon with proper "
            "citations to scientific sources."
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for text blocks with citations
        text_blocks_with_citations = []
        for block in result.content:
            if (
                isinstance(block, dict)
                and is_text_block(block)
                and "annotations" in block
            ):
                annotations = cast("list[dict[str, Any]]", block.get("annotations", []))
                citations = [
                    ann
                    for ann in annotations
                    if isinstance(ann, dict) and ann.get("type") == "citation"
                ]
                if citations:
                    text_blocks_with_citations.append(block)
        assert len(text_blocks_with_citations) > 0

        # Validate citation structure
        for block in text_blocks_with_citations:
            annotations = cast("list[dict[str, Any]]", block.get("annotations", []))
            for annotation in annotations:
                if annotation.get("type") == "citation":
                    # TODO: evaluate these since none are *technically* required
                    # This may be a test that needs adjustment on per-integration basis
                    assert "cited_text" in annotation
                    assert "start_index" in annotation
                    assert "end_index" in annotation

    def test_web_search_integration(self, model: BaseChatModel) -> None:
        """Test web search content blocks integration."""
        if not self.supports_web_search_blocks:
            pytest.skip("Model does not support web search blocks.")

        message = HumanMessage(
            "Search for the latest developments in quantum computing."
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for web search blocks
        search_call_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "web_search_call"
        ]
        search_result_blocks = [
            block
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "web_search_result"
        ]
        # TODO: should this be one or the other or both?
        assert len(search_call_blocks) > 0 or len(search_result_blocks) > 0

    def test_code_interpreter_blocks(self, model: BaseChatModel) -> None:
        """Test code interpreter content blocks."""
        if not self.supports_code_interpreter:
            pytest.skip("Model does not support code interpreter blocks.")

        message = HumanMessage("Calculate the factorial of 10 using Python code.")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)

        # Check for code interpreter blocks
        code_blocks = [
            block
            for block in result.content
            if isinstance(block, dict)
            and block.get("type")
            in [
                "code_interpreter_call",
                "code_interpreter_output",
                "code_interpreter_result",
            ]
        ]
        # TODO: should we require all three types or just an output/result?
        assert len(code_blocks) > 0

    def test_tool_calling_with_content_blocks(self, model: BaseChatModel) -> None:
        """Test tool calling with content blocks."""
        if not self.has_tool_calling:
            pytest.skip("Model does not support tool calls.")

        @tool
        def calculate_area(length: float, width: float) -> str:
            """Calculate the area of a rectangle."""
            area = length * width
            return f"The area is {area} square units."

        model_with_tools = model.bind_tools([calculate_area])
        message = HumanMessage(
            "Calculate the area of a rectangle with length 5 and width 3."
        )

        result = model_with_tools.invoke([message])
        _validate_tool_call_message(result)

    def test_plaintext_content_blocks_from_documents(
        self, model: BaseChatModel
    ) -> None:
        """Test PlainTextContentBlock for document plaintext content."""
        if not self.supports_plaintext_content_blocks:
            pytest.skip("Model does not support PlainTextContentBlock.")

        # Test with PlainTextContentBlock (plaintext from document)
        plaintext_block = create_plaintext_block(
            text="This is plaintext content extracted from a document.",
            file_id="doc_123",
        )

        message = HumanMessage(
            content=cast("list[types.ContentBlock]", [plaintext_block])
        )
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        # TODO expand

    def test_content_block_streaming_integration(self, model: BaseChatModel) -> None:
        """Test streaming with content blocks."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Write a detailed explanation of machine learning.",
                }
            ]
        )

        chunks = []
        for chunk in model.stream([message]):
            chunks.append(chunk)
            assert isinstance(chunk, (AIMessage, AIMessageChunk))

        assert len(chunks) > 1  # Should receive multiple chunks

        # Aggregate chunks
        final_message = chunks[0]
        for chunk in chunks[1:]:
            final_message = final_message + chunk

        assert isinstance(final_message.content, list)

    def test_error_handling_with_invalid_content_blocks(
        self, model: BaseChatModel
    ) -> None:
        """Test error handling with various invalid content block configurations."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        test_cases = [
            {"type": "text"},  # Missing text field
            {"type": "image"},  # Missing url/mime_type
            {"type": "tool_call", "name": "test"},  # Missing args/id
        ]

        for invalid_block in test_cases:
            message = HumanMessage([invalid_block])  # type: ignore[list-item]

            # Should either handle gracefully or raise appropriate error
            try:
                result = model.invoke([message])
                assert isinstance(result, AIMessage)
            except (ValueError, TypeError, KeyError) as e:
                # Acceptable to raise validation errors
                assert len(str(e)) > 0

    async def test_async_content_blocks_processing(self, model: BaseChatModel) -> None:
        """Test asynchronous processing of content blocks."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        message = HumanMessage("Generate a creative story about space exploration.")

        result = await model.ainvoke([message])
        assert isinstance(result, AIMessage)

    def test_content_blocks_with_callbacks(self, model: BaseChatModel) -> None:
        """Test that content blocks work correctly with callback handlers."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        class ContentBlockCallbackHandler(BaseCallbackHandler):
            def __init__(self) -> None:
                self.messages_seen: list[BaseMessage] = []

            def on_chat_model_start(
                self,
                serialized: Any,  # noqa: ARG002
                messages: Any,
                **kwargs: Any,  # noqa: ARG002
            ) -> None:
                self.messages_seen.extend(messages)

        callback_handler = ContentBlockCallbackHandler()

        message = HumanMessage("Test message for callback handling.")

        result = model.invoke([message], config={"callbacks": [callback_handler]})

        assert isinstance(result, AIMessage)
        assert len(callback_handler.messages_seen) > 0
        assert any(
            hasattr(msg, "content") and isinstance(msg.content, list)
            for msg in callback_handler.messages_seen
        )
