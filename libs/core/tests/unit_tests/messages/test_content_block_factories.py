"""Unit tests for ContentBlock factory functions."""

from uuid import UUID

import pytest

from langchain_core.messages.content_blocks import (
    create_audio_block,
    create_citation,
    create_file_block,
    create_image_block,
    create_non_standard_block,
    create_plain_text_block,
    create_reasoning_block,
    create_text_block,
    create_tool_call,
    create_video_block,
)


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
        UUID(id_value[3:])

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
        UUID(id_value[3:])

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

    def test_base64_without_mime_type_raises_error(self) -> None:
        """Test that base64 without mime_type raises ValueError."""
        with pytest.raises(
            ValueError, match="mime_type is required when using base64 data"
        ):
            create_image_block(base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ")

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

    def test_base64_without_mime_type_raises_error(self) -> None:
        """Test that base64 without mime_type raises ValueError."""
        with pytest.raises(
            ValueError, match="mime_type is required when using base64 data"
        ):
            create_video_block(base64="UklGRnoGAABXQVZFZm10IBAAAAABAAEA")


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
        block = create_plain_text_block("This is plain text content.")

        assert block["type"] == "text-plain"
        assert block.get("mime_type") == "text/plain"
        assert block.get("text") == "This is plain text content."
        assert "id" in block
        id_value = block.get("id")
        assert id_value is not None, "block id is None"
        UUID(id_value[3:])

    def test_with_title_and_context(self) -> None:
        """Test plain text block creation with title and context."""
        block = create_plain_text_block(
            "Document content here.",
            title="Important Document",
            context="This document contains important information.",
        )

        assert block.get("title") == "Important Document"
        assert block.get("context") == "This document contains important information."

    def test_with_url(self) -> None:
        """Test plain text block creation with URL."""
        block = create_plain_text_block(
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
        UUID(id_value[3:])

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
        UUID(id_value[3:])

    @pytest.mark.xfail(reason="Optional fields not implemented yet")
    def test_with_signatures(self) -> None:
        """Test reasoning block creation with signatures."""
        block = create_reasoning_block(
            "Thinking...",
            thought_signature="thought-sig-123",  # type: ignore[call-arg]
            signature="auth-sig-456",  # type: ignore[call-arg]
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
        UUID(id_value[3:])

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
        UUID(id_value[3:])

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

        plain_text_block = create_plain_text_block("content")
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
