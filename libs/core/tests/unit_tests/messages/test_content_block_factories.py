"""Unit tests for ContentBlock factory functions."""

from uuid import UUID

import pytest

from langchain_core.messages.content_blocks import (
    CodeInterpreterCall,
    CodeInterpreterOutput,
    CodeInterpreterResult,
    InvalidToolCall,
    ToolCallChunk,
    WebSearchCall,
    WebSearchResult,
    create_audio_block,
    create_citation,
    create_file_block,
    create_image_block,
    create_non_standard_block,
    create_plaintext_block,
    create_reasoning_block,
    create_text_block,
    create_tool_call,
    create_video_block,
)


def _validate_lc_uuid(id_value: str) -> None:
    """Validate that the ID has ``lc_`` prefix and valid UUID suffix.

    Args:
        id_value: The ID string to validate.

    Raises:
        AssertionError: If the ID doesn't have ``lc_`` prefix or invalid UUID.
    """
    assert id_value.startswith("lc_"), f"ID should start with 'lc_' but got: {id_value}"
    # Validate the UUID part after the lc_ prefix
    UUID(id_value[3:])


class TestTextBlockFactory:
    """Test create_text_block factory function."""

    def test_basic_creation(self) -> None:
        """Test basic text block creation."""
        block = create_text_block("Hello world")

        assert block["type"] == "text"
        assert block.get("text") == "Hello world"
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_custom_id(self) -> None:
        """Test text block creation with custom ID."""
        custom_id = "custom-123"
        block = create_text_block("Hello", id=custom_id)

        assert block.get("id") == custom_id

    def test_with_annotations(self) -> None:
        """Test text block creation with annotations."""
        citation = create_citation(url="https://example.com", title="Example")
        block = create_text_block("Hello", annotations=[citation])

        assert block.get("annotations") == [citation]

    def test_with_index(self) -> None:
        """Test text block creation with index."""
        block = create_text_block("Hello", index=42)

        assert block.get("index") == 42

    def test_optional_fields_not_present_when_none(self) -> None:
        """Test that optional fields are not included when None."""
        block = create_text_block("Hello")

        assert "annotations" not in block
        assert "index" not in block


class TestImageBlockFactory:
    """Test create_image_block factory function."""

    def test_with_url(self) -> None:
        """Test image block creation with URL."""
        block = create_image_block(url="https://example.com/image.jpg")

        assert block["type"] == "image"
        assert block.get("url") == "https://example.com/image.jpg"
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_base64(self) -> None:
        """Test image block creation with base64 data."""
        block = create_image_block(
            base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ", mime_type="image/png"
        )

        assert block.get("base64") == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        assert block.get("mime_type") == "image/png"

    def test_with_file_id(self) -> None:
        """Test image block creation with file ID."""
        block = create_image_block(file_id="file-123")

        assert block.get("file_id") == "file-123"

    def test_no_source_raises_error(self) -> None:
        """Test that missing all sources raises ValueError."""
        with pytest.raises(
            ValueError, match="Must provide one of: url, base64, or file_id"
        ):
            create_image_block()

    def test_with_index(self) -> None:
        """Test image block creation with index."""
        block = create_image_block(url="https://example.com/image.jpg", index=1)

        assert block.get("index") == 1

    def test_optional_fields_not_present_when_not_provided(self) -> None:
        """Test that optional fields are not included when not provided."""
        block = create_image_block(url="https://example.com/image.jpg")

        assert "base64" not in block
        assert "file_id" not in block
        assert "mime_type" not in block
        assert "index" not in block


class TestVideoBlockFactory:
    """Test create_video_block factory function."""

    def test_with_url(self) -> None:
        """Test video block creation with URL."""
        block = create_video_block(url="https://example.com/video.mp4")

        assert block["type"] == "video"
        assert block.get("url") == "https://example.com/video.mp4"

    def test_with_base64(self) -> None:
        """Test video block creation with base64 data."""
        block = create_video_block(
            base64="UklGRnoGAABXQVZFZm10IBAAAAABAAEA", mime_type="video/mp4"
        )

        assert block.get("base64") == "UklGRnoGAABXQVZFZm10IBAAAAABAAEA"
        assert block.get("mime_type") == "video/mp4"

    def test_no_source_raises_error(self) -> None:
        """Test that missing all sources raises ValueError."""
        with pytest.raises(
            ValueError, match="Must provide one of: url, base64, or file_id"
        ):
            create_video_block()


class TestAudioBlockFactory:
    """Test create_audio_block factory function."""

    def test_with_url(self) -> None:
        """Test audio block creation with URL."""
        block = create_audio_block(url="https://example.com/audio.mp3")

        assert block["type"] == "audio"
        assert block.get("url") == "https://example.com/audio.mp3"

    def test_with_base64(self) -> None:
        """Test audio block creation with base64 data."""
        block = create_audio_block(
            base64="UklGRnoGAABXQVZFZm10IBAAAAABAAEA", mime_type="audio/mp3"
        )

        assert block.get("base64") == "UklGRnoGAABXQVZFZm10IBAAAAABAAEA"
        assert block.get("mime_type") == "audio/mp3"

    def test_no_source_raises_error(self) -> None:
        """Test that missing all sources raises ValueError."""
        with pytest.raises(
            ValueError, match="Must provide one of: url, base64, or file_id"
        ):
            create_audio_block()


class TestFileBlockFactory:
    """Test create_file_block factory function."""

    def test_with_url(self) -> None:
        """Test file block creation with URL."""
        block = create_file_block(url="https://example.com/document.pdf")

        assert block["type"] == "file"
        assert block.get("url") == "https://example.com/document.pdf"

    def test_with_base64(self) -> None:
        """Test file block creation with base64 data."""
        block = create_file_block(
            base64="JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwo=",
            mime_type="application/pdf",
        )

        assert (
            block.get("base64")
            == "JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwo="
        )
        assert block.get("mime_type") == "application/pdf"

    def test_no_source_raises_error(self) -> None:
        """Test that missing all sources raises ValueError."""
        with pytest.raises(
            ValueError, match="Must provide one of: url, base64, or file_id"
        ):
            create_file_block()


class TestPlainTextBlockFactory:
    """Test create_plain_text_block factory function."""

    def test_basic_creation(self) -> None:
        """Test basic plain text block creation."""
        block = create_plaintext_block("This is plain text content.")

        assert block["type"] == "text-plain"
        assert block.get("mime_type") == "text/plain"
        assert block.get("text") == "This is plain text content."
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_title_and_context(self) -> None:
        """Test plain text block creation with title and context."""
        block = create_plaintext_block(
            "Document content here.",
            title="Important Document",
            context="This document contains important information.",
        )

        assert block.get("title") == "Important Document"
        assert block.get("context") == "This document contains important information."

    def test_with_url(self) -> None:
        """Test plain text block creation with URL."""
        block = create_plaintext_block(
            "Content", url="https://example.com/document.txt"
        )

        assert block.get("url") == "https://example.com/document.txt"


class TestToolCallFactory:
    """Test create_tool_call factory function."""

    def test_basic_creation(self) -> None:
        """Test basic tool call creation."""
        block = create_tool_call("search", {"query": "python"})

        assert block["type"] == "tool_call"
        assert block["name"] == "search"
        assert block["args"] == {"query": "python"}
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_custom_id(self) -> None:
        """Test tool call creation with custom ID."""
        block = create_tool_call("search", {"query": "python"}, id="tool-123")

        assert block.get("id") == "tool-123"

    def test_with_index(self) -> None:
        """Test tool call creation with index."""
        block = create_tool_call("search", {"query": "python"}, index=2)

        assert block.get("index") == 2


class TestReasoningBlockFactory:
    """Test create_reasoning_block factory function."""

    def test_basic_creation(self) -> None:
        """Test basic reasoning block creation."""
        block = create_reasoning_block("Let me think about this problem...")

        assert block["type"] == "reasoning"
        assert block.get("reasoning") == "Let me think about this problem..."
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    @pytest.mark.xfail(reason="Optional fields not implemented yet")
    def test_with_signatures(self) -> None:
        """Test reasoning block creation with signatures."""
        block = create_reasoning_block(
            "Thinking...",
            thought_signature="thought-sig-123",  # type: ignore[call-arg]
            signature="auth-sig-456",  # type: ignore[call-arg, unused-ignore]
        )

        assert block.get("thought_signature") == "thought-sig-123"
        assert block.get("signature") == "auth-sig-456"

    def test_with_index(self) -> None:
        """Test reasoning block creation with index."""
        block = create_reasoning_block("Thinking...", index=3)

        assert block.get("index") == 3


class TestCitationFactory:
    """Test create_citation factory function."""

    def test_basic_creation(self) -> None:
        """Test basic citation creation."""
        block = create_citation()

        assert block["type"] == "citation"
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_all_fields(self) -> None:
        """Test citation creation with all fields."""
        block = create_citation(
            url="https://example.com/source",
            title="Source Document",
            start_index=10,
            end_index=50,
            cited_text="This is the cited text.",
        )

        assert block.get("url") == "https://example.com/source"
        assert block.get("title") == "Source Document"
        assert block.get("start_index") == 10
        assert block.get("end_index") == 50
        assert block.get("cited_text") == "This is the cited text."

    def test_optional_fields_not_present_when_none(self) -> None:
        """Test that optional fields are not included when None."""
        block = create_citation()

        assert "url" not in block
        assert "title" not in block
        assert "start_index" not in block
        assert "end_index" not in block
        assert "cited_text" not in block


class TestNonStandardBlockFactory:
    """Test create_non_standard_block factory function."""

    def test_basic_creation(self) -> None:
        """Test basic non-standard block creation."""
        value = {"custom_field": "custom_value", "number": 42}
        block = create_non_standard_block(value)

        assert block["type"] == "non_standard"
        assert block["value"] == value
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        _validate_lc_uuid(id_value)

    def test_with_index(self) -> None:
        """Test non-standard block creation with index."""
        value = {"data": "test"}
        block = create_non_standard_block(value, index=5)

        assert block.get("index") == 5

    def test_optional_fields_not_present_when_none(self) -> None:
        """Test that optional fields are not included when None."""
        value = {"data": "test"}
        block = create_non_standard_block(value)

        assert "index" not in block


class TestUUIDValidation:
    """Test UUID generation and validation behavior."""

    def test_custom_id_bypasses_lc_prefix_requirement(self) -> None:
        """Test that custom IDs can use any format (don't require lc_ prefix)."""
        custom_id = "custom-123"
        block = create_text_block("Hello", id=custom_id)

        assert block.get("id") == custom_id
        # Custom IDs should not be validated with lc_ prefix requirement

    def test_generated_ids_are_unique(self) -> None:
        """Test that multiple factory calls generate unique IDs."""
        blocks = [create_text_block("test") for _ in range(10)]
        ids = [block.get("id") for block in blocks]

        # All IDs should be unique
        assert len(set(ids)) == len(ids)

        # All generated IDs should have lc_ prefix
        for id_value in ids:
            _validate_lc_uuid(id_value or "")

    def test_empty_string_id_generates_new_uuid(self) -> None:
        """Test that empty string ID generates new UUID with lc_ prefix."""
        block = create_text_block("Hello", id="")

        id_value: str = block.get("id", "")
        assert id_value != ""
        _validate_lc_uuid(id_value)

    def test_generated_id_length(self) -> None:
        """Test that generated IDs have correct length (UUID4 + lc_ prefix)."""
        block = create_text_block("Hello")

        id_value = block.get("id")
        assert id_value is not None

        # UUID4 string length is 36 chars, plus 3 for "lc_" prefix = 39 total
        expected_length = 36 + 3
        assert len(id_value) == expected_length, (
            f"Expected length {expected_length}, got {len(id_value)}"
        )

        # Validate it's properly formatted
        _validate_lc_uuid(id_value)


class TestFactoryTypeConsistency:
    """Test that factory functions return correctly typed objects."""

    def test_factories_return_correct_types(self) -> None:
        """Test that all factory functions return the expected TypedDict types."""
        text_block = create_text_block("test")
        assert isinstance(text_block, dict)
        assert text_block["type"] == "text"

        image_block = create_image_block(url="https://example.com/image.jpg")
        assert isinstance(image_block, dict)
        assert image_block["type"] == "image"

        video_block = create_video_block(url="https://example.com/video.mp4")
        assert isinstance(video_block, dict)
        assert video_block["type"] == "video"

        audio_block = create_audio_block(url="https://example.com/audio.mp3")
        assert isinstance(audio_block, dict)
        assert audio_block["type"] == "audio"

        file_block = create_file_block(url="https://example.com/file.pdf")
        assert isinstance(file_block, dict)
        assert file_block["type"] == "file"

        plain_text_block = create_plaintext_block("content")
        assert isinstance(plain_text_block, dict)
        assert plain_text_block["type"] == "text-plain"

        tool_call = create_tool_call("tool", {"arg": "value"})
        assert isinstance(tool_call, dict)
        assert tool_call["type"] == "tool_call"

        reasoning_block = create_reasoning_block("reasoning")
        assert isinstance(reasoning_block, dict)
        assert reasoning_block["type"] == "reasoning"

        citation = create_citation()
        assert isinstance(citation, dict)
        assert citation["type"] == "citation"

        non_standard_block = create_non_standard_block({"data": "value"})
        assert isinstance(non_standard_block, dict)
        assert non_standard_block["type"] == "non_standard"


class TestExtraItems:
    """Test that content blocks support extra items via __extra_items__ field."""

    def test_text_block_extra_items(self) -> None:
        """Test that TextContentBlock can store extra provider-specific fields."""
        block = create_text_block("Hello world")

        block["openai_metadata"] = {"model": "gpt-4", "temperature": 0.7}
        block["anthropic_usage"] = {"input_tokens": 10, "output_tokens": 20}
        block["custom_field"] = "any value"

        assert block["type"] == "text"
        assert block["text"] == "Hello world"
        assert "id" in block
        assert block.get("openai_metadata") == {"model": "gpt-4", "temperature": 0.7}
        assert block.get("anthropic_usage") == {"input_tokens": 10, "output_tokens": 20}
        assert block.get("custom_field") == "any value"

    def test_text_block_extras_field(self) -> None:
        """Test that TextContentBlock properly supports the explicit extras field."""
        block = create_text_block("Hello world")

        # Test direct assignment to extras field
        block["extras"] = {
            "openai_metadata": {"model": "gpt-4", "temperature": 0.7},
            "anthropic_usage": {"input_tokens": 10, "output_tokens": 20},
            "custom_field": "any value",
        }

        assert block["type"] == "text"
        assert block["text"] == "Hello world"
        assert "id" in block
        assert "extras" in block

        extras = block.get("extras", {})
        assert extras.get("openai_metadata") == {"model": "gpt-4", "temperature": 0.7}
        expected_usage = {"input_tokens": 10, "output_tokens": 20}
        assert extras.get("anthropic_usage") == expected_usage
        assert extras.get("custom_field") == "any value"

    def test_mixed_extra_items_types(self) -> None:
        """Test that extra items can be various types (str, int, bool, dict, list)."""
        block = create_text_block("Test content")

        # Add various types of extra fields
        block["string_field"] = "string value"
        block["int_field"] = 42
        block["float_field"] = 3.14
        block["bool_field"] = True
        block["list_field"] = ["item1", "item2", "item3"]
        block["dict_field"] = {"nested": {"deeply": "nested value"}}
        block["none_field"] = None

        # Verify all types are preserved
        assert block.get("string_field") == "string value"
        assert block.get("int_field") == 42
        assert block.get("float_field") == 3.14
        assert block.get("bool_field") is True
        assert block.get("list_field") == ["item1", "item2", "item3"]
        dict_field = block.get("dict_field", {})
        assert isinstance(dict_field, dict)
        nested = dict_field.get("nested", {})
        assert isinstance(nested, dict)
        assert nested.get("deeply") == "nested value"
        assert block.get("none_field") is None

    def test_extra_items_do_not_interfere_with_standard_fields(self) -> None:
        """Test that extra items don't interfere with standard field access."""
        block = create_text_block("Original text", index=1)

        # Add many extra fields
        for i in range(10):
            block[f"extra_field_{i}"] = f"value_{i}"  # type: ignore[literal-required]

        # Standard fields should still work correctly
        assert block["type"] == "text"
        assert block["text"] == "Original text"
        assert block["index"] == 1 if "index" in block else None
        assert "id" in block

        # Extra fields should also be accessible
        for i in range(10):
            assert block.get(f"extra_field_{i}") == f"value_{i}"

    def test_extra_items_can_be_modified(self) -> None:
        """Test that extra items can be modified after creation."""
        block = create_image_block(url="https://example.com/image.jpg")

        # Add an extra field
        block["status"] = "pending"
        assert block.get("status") == "pending"

        # Modify the extra field
        block["status"] = "processed"
        assert block.get("status") == "processed"

        # Add more fields
        block["metadata"] = {"version": 1}
        metadata = block.get("metadata", {})
        assert isinstance(metadata, dict)
        assert metadata.get("version") == 1

        # Modify nested extra field
        block["metadata"]["version"] = 2  # type: ignore[typeddict-item]
        metadata = block.get("metadata", {})
        assert isinstance(metadata, dict)
        assert metadata.get("version") == 2

    def test_all_content_blocks_support_extra_items(self) -> None:
        """Test that all content block types support extra items."""
        # Test each content block type
        text_block = create_text_block("test")
        text_block["extra"] = "text_extra"
        assert text_block.get("extra") == "text_extra"

        image_block = create_image_block(url="https://example.com/image.jpg")
        image_block["extra"] = "image_extra"
        assert image_block.get("extra") == "image_extra"

        video_block = create_video_block(url="https://example.com/video.mp4")
        video_block["extra"] = "video_extra"
        assert video_block.get("extra") == "video_extra"

        audio_block = create_audio_block(url="https://example.com/audio.mp3")
        audio_block["extra"] = "audio_extra"
        assert audio_block.get("extra") == "audio_extra"

        file_block = create_file_block(url="https://example.com/file.pdf")
        file_block["extra"] = "file_extra"
        assert file_block.get("extra") == "file_extra"

        plain_text_block = create_plaintext_block("content")
        plain_text_block["extra"] = "plaintext_extra"
        assert plain_text_block.get("extra") == "plaintext_extra"

        tool_call = create_tool_call("tool", {"arg": "value"})
        tool_call["extra"] = "tool_extra"
        assert tool_call.get("extra") == "tool_extra"

        reasoning_block = create_reasoning_block("reasoning")
        reasoning_block["extra"] = "reasoning_extra"
        assert reasoning_block.get("extra") == "reasoning_extra"

        non_standard_block = create_non_standard_block({"data": "value"})
        non_standard_block["extra"] = "non_standard_extra"
        assert non_standard_block.get("extra") == "non_standard_extra"


class TestExtrasField:
    """Test the explicit extras field across all content block types."""

    def test_all_content_blocks_support_extras_field(self) -> None:
        """Test that all content block types support the explicit extras field."""
        provider_metadata = {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        # Test TextContentBlock
        text_block = create_text_block("test")
        text_block["extras"] = provider_metadata
        assert text_block.get("extras") == provider_metadata
        assert text_block["type"] == "text"

        # Test ImageContentBlock
        image_block = create_image_block(url="https://example.com/image.jpg")
        image_block["extras"] = provider_metadata
        assert image_block.get("extras") == provider_metadata
        assert image_block["type"] == "image"

        # Test VideoContentBlock
        video_block = create_video_block(url="https://example.com/video.mp4")
        video_block["extras"] = provider_metadata
        assert video_block.get("extras") == provider_metadata
        assert video_block["type"] == "video"

        # Test AudioContentBlock
        audio_block = create_audio_block(url="https://example.com/audio.mp3")
        audio_block["extras"] = provider_metadata
        assert audio_block.get("extras") == provider_metadata
        assert audio_block["type"] == "audio"

        # Test FileContentBlock
        file_block = create_file_block(url="https://example.com/file.pdf")
        file_block["extras"] = provider_metadata
        assert file_block.get("extras") == provider_metadata
        assert file_block["type"] == "file"

        # Test PlainTextContentBlock
        plain_text_block = create_plaintext_block("content")
        plain_text_block["extras"] = provider_metadata
        assert plain_text_block.get("extras") == provider_metadata
        assert plain_text_block["type"] == "text-plain"

        # Test ToolCall
        tool_call = create_tool_call("tool", {"arg": "value"})
        tool_call["extras"] = provider_metadata
        assert tool_call.get("extras") == provider_metadata
        assert tool_call["type"] == "tool_call"

        # Test ReasoningContentBlock
        reasoning_block = create_reasoning_block("reasoning")
        reasoning_block["extras"] = provider_metadata
        assert reasoning_block.get("extras") == provider_metadata
        assert reasoning_block["type"] == "reasoning"

        # Test Citation
        citation = create_citation()
        citation["extras"] = provider_metadata
        assert citation.get("extras") == provider_metadata
        assert citation["type"] == "citation"

    def test_extras_field_is_optional(self) -> None:
        """Test that the extras field is optional and blocks work without it."""
        # Create blocks without extras
        text_block = create_text_block("test")
        image_block = create_image_block(url="https://example.com/image.jpg")
        tool_call = create_tool_call("tool", {"arg": "value"})
        reasoning_block = create_reasoning_block("reasoning")
        citation = create_citation()

        # Verify blocks work correctly without extras
        assert text_block["type"] == "text"
        assert image_block["type"] == "image"
        assert tool_call["type"] == "tool_call"
        assert reasoning_block["type"] == "reasoning"
        assert citation["type"] == "citation"

        # Verify extras field is not present when not set
        assert "extras" not in text_block
        assert "extras" not in image_block
        assert "extras" not in tool_call
        assert "extras" not in reasoning_block
        assert "extras" not in citation

    def test_extras_field_can_be_modified(self) -> None:
        """Test that the extras field can be modified after creation."""
        block = create_text_block("test")

        # Add extras
        block["extras"] = {"initial": "value"}
        assert block.get("extras") == {"initial": "value"}

        # Modify extras
        block["extras"] = {"updated": "value", "count": 42}
        extras = block.get("extras", {})
        assert extras.get("updated") == "value"
        assert extras.get("count") == 42
        assert "initial" not in extras

        # Update nested values in extras
        if "extras" in block:
            block["extras"]["nested"] = {"deep": "value"}
            extras = block.get("extras", {})
            nested = extras.get("nested", {})
            assert isinstance(nested, dict)
            assert nested.get("deep") == "value"

    def test_extras_field_supports_various_data_types(self) -> None:
        """Test that the extras field can store various data types."""
        block = create_text_block("test")

        complex_extras = {
            "string_val": "test string",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": ["item1", "item2", {"nested": "in_list"}],
            "dict_val": {"nested": {"deeply": {"nested": "value"}}},
        }

        block["extras"] = complex_extras

        extras = block.get("extras", {})
        assert extras.get("string_val") == "test string"
        assert extras.get("int_val") == 42
        assert extras.get("float_val") == 3.14
        assert extras.get("bool_val") is True
        assert extras.get("none_val") is None

        list_val = extras.get("list_val", [])
        assert isinstance(list_val, list)
        assert len(list_val) == 3
        assert list_val[0] == "item1"
        assert list_val[1] == "item2"
        assert isinstance(list_val[2], dict)
        assert list_val[2].get("nested") == "in_list"

        dict_val = extras.get("dict_val", {})
        assert isinstance(dict_val, dict)
        nested = dict_val.get("nested", {})
        assert isinstance(nested, dict)
        deeply = nested.get("deeply", {})
        assert isinstance(deeply, dict)
        assert deeply.get("nested") == "value"

    def test_extras_field_does_not_interfere_with_standard_fields(self) -> None:
        """Test that the extras field doesn't interfere with standard fields."""
        # Create a complex block with all standard fields
        block = create_text_block(
            "Test content",
            annotations=[create_citation(url="https://example.com")],
            index=42,
        )

        # Add extensive extras
        large_extras = {f"field_{i}": f"value_{i}" for i in range(100)}
        block["extras"] = large_extras

        # Verify all standard fields still work
        assert block["type"] == "text"
        assert block["text"] == "Test content"
        assert block.get("index") == 42
        assert "id" in block
        assert "annotations" in block

        annotations = block.get("annotations", [])
        assert len(annotations) == 1
        assert annotations[0]["type"] == "citation"

        # Verify extras field works
        extras = block.get("extras", {})
        assert len(extras) == 100
        for i in range(100):
            assert extras.get(f"field_{i}") == f"value_{i}"

    def test_special_content_blocks_support_extras_field(self) -> None:
        """Test that special content blocks support extras field."""
        provider_metadata = {
            "provider": "openai",
            "request_id": "req_12345",
            "timing": {"start": 1234567890, "end": 1234567895},
        }

        # Test ToolCallChunk
        tool_call_chunk: ToolCallChunk = {
            "type": "tool_call_chunk",
            "id": "tool_123",
            "name": "search",
            "args": '{"query": "test"}',
            "index": 0,
            "extras": provider_metadata,
        }
        assert tool_call_chunk.get("extras") == provider_metadata
        assert tool_call_chunk["type"] == "tool_call_chunk"

        # Test InvalidToolCall
        invalid_tool_call: InvalidToolCall = {
            "type": "invalid_tool_call",
            "id": "invalid_123",
            "name": "bad_tool",
            "args": "invalid json",
            "error": "JSON parse error",
            "extras": provider_metadata,
        }
        assert invalid_tool_call.get("extras") == provider_metadata
        assert invalid_tool_call["type"] == "invalid_tool_call"

        # Test WebSearchCall
        web_search_call: WebSearchCall = {
            "type": "web_search_call",
            "id": "search_123",
            "query": "python langchain",
            "index": 0,
            "extras": provider_metadata,
        }
        assert web_search_call.get("extras") == provider_metadata
        assert web_search_call["type"] == "web_search_call"

        # Test WebSearchResult
        web_search_result: WebSearchResult = {
            "type": "web_search_result",
            "id": "result_123",
            "urls": ["https://example.com", "https://test.com"],
            "index": 0,
            "extras": provider_metadata,
        }
        assert web_search_result.get("extras") == provider_metadata
        assert web_search_result["type"] == "web_search_result"

        # Test CodeInterpreterCall
        code_interpreter_call: CodeInterpreterCall = {
            "type": "code_interpreter_call",
            "id": "code_123",
            "language": "python",
            "code": "print('hello world')",
            "index": 0,
            "extras": provider_metadata,
        }
        assert code_interpreter_call.get("extras") == provider_metadata
        assert code_interpreter_call["type"] == "code_interpreter_call"

        # Test CodeInterpreterOutput
        code_interpreter_output: CodeInterpreterOutput = {
            "type": "code_interpreter_output",
            "id": "output_123",
            "return_code": 0,
            "stderr": "",
            "stdout": "hello world\n",
            "file_ids": ["file_123"],
            "index": 0,
            "extras": provider_metadata,
        }
        assert code_interpreter_output.get("extras") == provider_metadata
        assert code_interpreter_output["type"] == "code_interpreter_output"

        # Test CodeInterpreterResult
        code_interpreter_result: CodeInterpreterResult = {
            "type": "code_interpreter_result",
            "id": "result_123",
            "output": [code_interpreter_output],
            "index": 0,
            "extras": provider_metadata,
        }
        assert code_interpreter_result.get("extras") == provider_metadata
        assert code_interpreter_result["type"] == "code_interpreter_result"

    def test_extras_field_is_not_required_for_special_blocks(self) -> None:
        """Test that extras field is optional for all special content blocks."""
        # Create blocks without extras field
        tool_call_chunk: ToolCallChunk = {
            "id": "tool_123",
            "name": "search",
            "args": '{"query": "test"}',
            "index": 0,
        }

        invalid_tool_call: InvalidToolCall = {
            "type": "invalid_tool_call",
            "id": "invalid_123",
            "name": "bad_tool",
            "args": "invalid json",
            "error": "JSON parse error",
        }

        web_search_call: WebSearchCall = {
            "type": "web_search_call",
            "query": "python langchain",
        }

        web_search_result: WebSearchResult = {
            "type": "web_search_result",
            "urls": ["https://example.com"],
        }

        code_interpreter_call: CodeInterpreterCall = {
            "type": "code_interpreter_call",
            "code": "print('hello')",
        }

        code_interpreter_output: CodeInterpreterOutput = {
            "type": "code_interpreter_output",
            "stdout": "hello\n",
        }

        code_interpreter_result: CodeInterpreterResult = {
            "type": "code_interpreter_result",
            "output": [code_interpreter_output],
        }

        # Verify they work without extras
        assert tool_call_chunk.get("name") == "search"
        assert invalid_tool_call["type"] == "invalid_tool_call"
        assert web_search_call["type"] == "web_search_call"
        assert web_search_result["type"] == "web_search_result"
        assert code_interpreter_call["type"] == "code_interpreter_call"
        assert code_interpreter_output["type"] == "code_interpreter_output"
        assert code_interpreter_result["type"] == "code_interpreter_result"

        # Verify extras field is not present
        assert "extras" not in tool_call_chunk
        assert "extras" not in invalid_tool_call
        assert "extras" not in web_search_call
        assert "extras" not in web_search_result
        assert "extras" not in code_interpreter_call
        assert "extras" not in code_interpreter_output
        assert "extras" not in code_interpreter_result
