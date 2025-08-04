""":autodoc-options: autoproperty.

Standard unit tests for chat models supporting v1 messages.

This module provides updated test patterns for the new messages introduced in
``langchain_core.messages.content_blocks``. Notably, this includes the standardized
content blocks system.
"""

from typing import Literal, cast

import pytest
from langchain_core.language_models.v1.chat_models import BaseChatModelV1
from langchain_core.load import dumpd, load
from langchain_core.messages.content_blocks import (
    ContentBlock,
    InvalidToolCall,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_non_standard_block,
    create_text_block,
    is_reasoning_block,
    is_text_block,
    is_tool_call_block,
)
from langchain_core.messages.v1 import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain_tests.base import BaseStandardTests


class ChatModelV1Tests(BaseStandardTests):
    """Test suite for v1 chat models.

    This class provides comprehensive testing for the new message system introduced in
    LangChain v1, including the standardized content block format.

    :private:
    """

    # Core Model Properties - these should be implemented by subclasses
    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool calling."""
        return False

    @property
    def has_structured_output(self) -> bool:
        """Whether the model supports structured output."""
        return False

    @property
    def supports_json_mode(self) -> bool:
        """Whether the model supports JSON mode."""
        return False

    # Content Block Support Properties
    @property
    def supports_content_blocks_v1(self) -> bool:
        """Whether the model supports content blocks v1 format.

        Defualts to True. This should not be overridden by a ChatV1 subclass. You may
        override the following properties to enable specific content block support.
        Each defaults to False:

        - ``supports_reasoning_content_blocks``
        - ``supports_plaintext_content_blocks``
        - ``supports_file_content_blocks``
        - ``supports_image_content_blocks``
        - ``supports_audio_content_blocks``
        - ``supports_video_content_blocks``
        - ``supports_citations``
        - ``supports_web_search_blocks``
        - ``supports_enhanced_tool_calls``
        - ``supports_invalid_tool_calls``
        - ``supports_tool_call_chunks``

        """
        return True

    @property
    def supports_non_standard_blocks(self) -> bool:
        """Whether the model supports ``NonStandardContentBlock``."""
        return True

    @property
    def supports_text_content_blocks(self) -> bool:
        """Whether the model supports ``TextContentBlock``.

        This is a minimum requirement for v1 chat models.

        """
        return self.supports_content_blocks_v1

    @property
    def supports_reasoning_content_blocks(self) -> bool:
        """Whether the model supports ``ReasoningContentBlock``."""
        return False

    @property
    def supports_plaintext_content_blocks(self) -> bool:
        """Whether the model supports ``PlainTextContentBlock``."""
        return False

    @property
    def supports_file_content_blocks(self) -> bool:
        """Whether the model supports ``FileContentBlock``.

        Replaces ``supports_pdf_inputs`` from v0.

        """
        return False

    @property
    def supports_image_content_blocks(self) -> bool:
        """Whether the model supports ``ImageContentBlock``.

        Replaces ``supports_image_inputs`` from v0.

        """
        return False

    @property
    def supports_audio_content_blocks(self) -> bool:
        """Whether the model supports ``AudioContentBlock``.

        Replaces ``supports_audio_inputs`` from v0.

        """
        return False

    @property
    def supports_video_content_blocks(self) -> bool:
        """Whether the model supports ``VideoContentBlock``.

        Replaces ``supports_video_inputs`` from v0.

        """
        return False

    @property
    def supports_citations(self) -> bool:
        """Whether the model supports ``Citation`` annotations."""
        return False

    @property
    def supports_web_search_blocks(self) -> bool:
        """Whether the model supports ``WebSearchCall``/``WebSearchResult`` blocks."""
        return False

    @property
    def supports_invalid_tool_calls(self) -> bool:
        """Whether the model can handle ``InvalidToolCall`` blocks."""
        return False

    @property
    def has_tool_choice(self) -> bool:
        """Whether the model supports forcing tool calling via ``tool_choice``."""
        return False

    @property
    def structured_output_kwargs(self) -> dict:
        """Additional kwargs for ``with_structured_output``."""
        return {}

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the model supports Anthropic-style inputs."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Whether the model returns usage metadata on invoke and streaming."""
        return True

    @property
    def supported_usage_metadata_details(
        self,
    ) -> dict[
        Literal["invoke", "stream"],
        list[
            Literal[
                "audio_input",
                "audio_output",
                "reasoning_output",
                "cache_read_input",
                "cache_creation_input",
            ]
        ],
    ]:
        """What usage metadata details are emitted in ``invoke()`` and ``stream()``."""
        return {"invoke": [], "stream": []}

    @property
    def enable_vcr_tests(self) -> bool:
        """Whether to enable VCR tests for the chat model."""
        return False


class ChatModelV1UnitTests(ChatModelV1Tests):
    """Unit tests for chat models with content blocks v1 support.

    These tests run in isolation without external dependencies.
    """

    # Core Method Tests
    def test_invoke_basic(self, model: BaseChatModelV1) -> None:
        """Test basic invoke functionality with simple string input."""
        result = model.invoke("Hello, world!")
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_invoke_with_message_list(self, model: BaseChatModelV1) -> None:
        """Test invoke with list of messages."""
        messages = [HumanMessage("Hello, world!")]
        result = model.invoke(messages)
        assert isinstance(result, AIMessage)
        assert result.content is not None

    async def test_ainvoke_basic(self, model: BaseChatModelV1) -> None:
        """Test basic async invoke functionality."""
        result = await model.ainvoke("Hello, world!")
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_stream_basic(self, model: BaseChatModelV1) -> None:
        """Test basic streaming functionality."""
        chunks = []
        for chunk in model.stream("Hello, world!"):
            chunks.append(chunk)
            assert hasattr(chunk, "content")

        assert len(chunks) > 0
        # Verify chunks can be aggregated
        if chunks:
            final_message = chunks[0]
            for chunk in chunks[1:]:
                final_message = final_message + chunk
            assert isinstance(final_message.content, (str, list))

    async def test_astream_basic(self, model: BaseChatModelV1) -> None:
        """Test basic async streaming functionality."""
        chunks = []
        async for chunk in model.astream("Hello, world!"):
            chunks.append(chunk)
            assert hasattr(chunk, "content")

        assert len(chunks) > 0
        # Verify chunks can be aggregated
        if chunks:
            final_message = chunks[0]
            for chunk in chunks[1:]:
                final_message = final_message + chunk
            assert isinstance(final_message.content, (str, list))

    # Property Tests
    def test_llm_type_property(self, model: BaseChatModelV1) -> None:
        """Test that ``_llm_type`` property is implemented and returns a string."""
        llm_type = model._llm_type
        assert isinstance(llm_type, str)
        assert len(llm_type) > 0

    def test_identifying_params_property(self, model: BaseChatModelV1) -> None:
        """Test that ``_identifying_params`` property returns a mapping."""
        params = model._identifying_params
        assert isinstance(params, dict)  # Should be dict-like mapping

    # Serialization Tests
    def test_dump_serialization(self, model: BaseChatModelV1) -> None:
        """Test that ``dump()`` returns proper serialization."""
        dumped = model.dump()
        assert isinstance(dumped, dict)
        assert "_type" in dumped
        assert dumped["_type"] == model._llm_type

        # Should contain identifying parameters
        for key, value in model._identifying_params.items():
            assert key in dumped
            assert dumped[key] == value

    # Input Conversion Tests
    def test_input_conversion_string(self, model: BaseChatModelV1) -> None:
        """Test that string input is properly converted to messages."""
        # This test verifies the _convert_input method works correctly
        result = model.invoke("Test string input")
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_input_conversion_empty_string(self, model: BaseChatModelV1) -> None:
        """Test that empty string input is handled gracefully."""
        result = model.invoke("")
        assert isinstance(result, AIMessage)
        # Content might be empty or some default response

    def test_input_conversion_message_v1_list(self, model: BaseChatModelV1) -> None:
        """Test that v1 message list input is handled correctly."""
        messages = [HumanMessage("Test message")]
        result = model.invoke(messages)
        assert isinstance(result, AIMessage)
        assert result.content is not None

    # Batch Processing Tests
    def test_batch_basic(self, model: BaseChatModelV1) -> None:
        """Test basic batch processing functionality."""
        inputs = ["Hello", "How are you?", "Goodbye"]
        results = model.batch(inputs)  # type: ignore[arg-type]

        assert isinstance(results, list)
        assert len(results) == len(inputs)
        for result in results:
            assert isinstance(result, AIMessage)
            assert result.content is not None

    async def test_abatch_basic(self, model: BaseChatModelV1) -> None:
        """Test basic async batch processing functionality."""
        inputs = ["Hello", "How are you?", "Goodbye"]
        results = await model.abatch(inputs)  # type: ignore[arg-type]

        assert isinstance(results, list)
        assert len(results) == len(inputs)
        for result in results:
            assert isinstance(result, AIMessage)
            assert result.content is not None

    # Content Block Tests
    def test_text_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test that the model can handle the ``TextContentBlock`` format.

        This test verifies that the model correctly processes messages containing
        ``TextContentBlock`` objects instead of plain strings.
        """
        if not self.supports_text_content_blocks:
            pytest.skip("Model does not support TextContentBlock (rare!)")

        text_block = create_text_block("Hello, world!")
        message = HumanMessage(content=[text_block])

        result = model.invoke([message])
        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_mixed_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test that the model can handle messages with mixed content blocks."""
        if not (
            self.supports_text_content_blocks and self.supports_image_content_blocks
        ):
            pytest.skip(
                "Model doesn't support mixed content blocks (concurrent text and image)"
            )

        content_blocks: list[ContentBlock] = [
            create_text_block("Describe this image:"),
            create_image_block(
                base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                mime_type="image/png",
            ),
        ]

        message = HumanMessage(content=content_blocks)
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        assert result.content is not None

    def test_reasoning_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test that the model can generate ``ReasoningContentBlock``."""
        if not self.supports_reasoning_content_blocks:
            pytest.skip("Model does not support ReasoningContentBlock.")

        message = HumanMessage("Think step by step: What is 2 + 2?")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        if isinstance(result.content, list):
            reasoning_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_reasoning_block(block)
            ]
            assert len(reasoning_blocks) > 0

    def test_citations_in_response(self, model: BaseChatModelV1) -> None:
        """Test that the model can generate ``Citations`` in text blocks."""
        if not self.supports_citations:
            pytest.skip("Model does not support citations.")

        message = HumanMessage("Provide information about Python with citations.")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        if isinstance(result.content, list):
            content_list = result.content
            text_blocks_with_citations: list[TextContentBlock] = []
            for block in content_list:
                if (
                    isinstance(block, dict)
                    and is_text_block(block)
                    and "annotations" in block
                    and isinstance(block.get("annotations"), list)
                    and len(cast(list, block.get("annotations", []))) > 0
                ):
                    text_block = cast(TextContentBlock, block)
                    text_blocks_with_citations.append(text_block)
            assert len(text_blocks_with_citations) > 0

            # Verify that at least one known citation type is present
            has_citation = any(
                "citation" in annotation.get("type", "")
                for block in text_blocks_with_citations
                for annotation in block.get("annotations", [])
            ) or any(
                "non_standard_annotation" in annotation.get("type", "")
                for block in text_blocks_with_citations
                for annotation in block.get("annotations", [])
            )
            assert has_citation, "No citations found in text blocks."

    def test_non_standard_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test that the model can handle ``NonStandardContentBlock``."""
        if not self.supports_non_standard_blocks:
            pytest.skip("Model does not support NonStandardContentBlock.")

        non_standard_block = create_non_standard_block(
            {
                "custom_field": "custom_value",
                "data": [1, 2, 3],
            }
        )

        message = HumanMessage(content=[non_standard_block])

        # Should not raise an error
        result = model.invoke([message])
        assert isinstance(result, AIMessage)

    def test_enhanced_tool_calls_with_content_blocks(
        self, model: BaseChatModelV1
    ) -> None:
        """Test enhanced tool calling with content blocks format."""
        if not self.has_tool_calling:
            pytest.skip("Model does not support enhanced tool calls.")

        @tool
        def sample_tool(query: str) -> str:
            """A sample tool for testing."""
            return f"Result for: {query}"

        model_with_tools = model.bind_tools([sample_tool])
        message = HumanMessage("Use the sample tool with query 'test'")

        result = model_with_tools.invoke([message])
        assert isinstance(result, AIMessage)

        # Check if tool calls are in content blocks format
        if isinstance(result.content, list):
            tool_call_blocks = [
                block
                for block in result.content
                if isinstance(block, dict) and is_tool_call_block(block)
            ]
            assert len(tool_call_blocks) > 0
        # Backwards compat?
        # else:
        #     # Fallback to legacy tool_calls attribute
        #     assert hasattr(result, "tool_calls") and result.tool_calls

    def test_invalid_tool_call_handling(self, model: BaseChatModelV1) -> None:
        """Test that the model can handle ``InvalidToolCall`` blocks gracefully."""
        if not self.supports_invalid_tool_calls:
            pytest.skip("Model does not support InvalidToolCall handling.")

        invalid_tool_call: InvalidToolCall = {
            "type": "invalid_tool_call",
            "name": "nonexistent_tool",
            "args": None,
            "id": "invalid_123",
            "error": "Tool not found",
        }

        # Create a message with invalid tool call in history
        ai_message = AIMessage(content=[invalid_tool_call])
        follow_up = HumanMessage("Please try again with a valid approach.")

        result = model.invoke([ai_message, follow_up])
        assert isinstance(result, AIMessage)
        assert result.content is not None
        # TODO: enhance/double check this

    def test_web_search_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test generating ``WebSearchCall``/``WebSearchResult`` blocks."""
        if not self.supports_web_search_blocks:
            pytest.skip("Model does not support web search blocks.")

        message = HumanMessage("Search for recent news about AI developments.")
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        if isinstance(result.content, list):
            search_blocks = [
                block
                for block in result.content
                if isinstance(block, dict)
                and block.get("type") in ["web_search_call", "web_search_result"]
            ]
            assert len(search_blocks) > 0

    def test_file_content_blocks(self, model: BaseChatModelV1) -> None:
        """Test that the model can handle ``FileContentBlock``."""
        if not self.supports_file_content_blocks:
            pytest.skip("Model does not support FileContentBlock.")

        file_block = create_file_block(
            base64="SGVsbG8sIHdvcmxkIQ==",  # "Hello, world!"
            mime_type="text/plain",
        )

        message = HumanMessage(content=[file_block])
        result = model.invoke([message])

        assert isinstance(result, AIMessage)
        assert result.content is not None
        # TODO: make more robust?

    def test_content_block_streaming(self, model: BaseChatModelV1) -> None:
        """Test that content blocks work correctly with streaming."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        text_block = create_text_block("Generate a short story.")
        message = HumanMessage(content=[text_block])

        chunks = []
        for chunk in model.stream([message]):
            chunks.append(chunk)
            assert hasattr(chunk, "content")

        assert len(chunks) > 0

        # Verify final aggregated message
        final_message = chunks[0]
        for chunk in chunks[1:]:
            final_message = final_message + chunk

        assert isinstance(final_message.content, (str, list))

    def test_content_block_serialization(self, model: BaseChatModelV1) -> None:
        """Test that messages with content blocks can be serialized/deserialized."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        text_block = create_text_block("Test serialization")
        message = HumanMessage(content=[text_block])

        # Test serialization
        serialized = dumpd(message)
        assert isinstance(serialized, dict)

        # Test deserialization
        deserialized = load(serialized)
        assert isinstance(deserialized, HumanMessage)
        assert deserialized.content == message.content
        # TODO: make more robust

    def test_backwards_compatibility(self, model: BaseChatModelV1) -> None:
        """Test that models still work with legacy string content."""
        # This should work regardless of content blocks support
        legacy_message = HumanMessage("Hello, world!")
        result = model.invoke([legacy_message])

        assert isinstance(result, AIMessage)
        assert result.content is not None

        legacy_message_named_param = HumanMessage(content="Hello, world!")
        result_named_param = model.invoke([legacy_message_named_param])

        assert isinstance(result_named_param, AIMessage)
        assert result_named_param.content is not None

    def test_content_block_validation(self, model: BaseChatModelV1) -> None:
        """Test that invalid content blocks are handled gracefully."""
        if not self.supports_content_blocks_v1:
            pytest.skip("Model does not support content blocks v1.")

        # Test with invalid content block structure
        invalid_block = {"type": "invalid_type", "invalid_field": "value"}
        message = HumanMessage(content=[invalid_block])  # type: ignore[list-item]

        # Should handle gracefully (either convert to NonStandardContentBlock or reject)
        try:
            result = model.invoke([message])
            assert isinstance(result, AIMessage)
        except (ValueError, TypeError) as e:
            # Acceptable to raise validation errors for truly invalid blocks
            assert "invalid" in str(e).lower() or "unknown" in str(e).lower()
