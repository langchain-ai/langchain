"""Tests for BaseMessage.pretty_repr()."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import _render_block


def test_pretty_repr_string_content() -> None:
    """String content renders as-is."""
    msg = HumanMessage(content="What is the capital of France?")
    result = msg.pretty_repr()
    assert "What is the capital of France?" in result
    assert "Human Message" in result


def test_pretty_repr_list_of_strings() -> None:
    """List of strings is joined with newlines, not shown as Python repr."""
    msg = AIMessage(content=["First", "Second", "Third"])
    result = msg.pretty_repr()
    assert "First" in result
    assert "Second" in result
    assert "Third" in result
    # Must NOT fall back to Python list repr
    assert "['First', 'Second', 'Third']" not in result


def test_pretty_repr_list_with_text_blocks() -> None:
    """Dicts with type='text' have their text extracted."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
    )
    result = msg.pretty_repr()
    assert "Hello" in result
    assert "World" in result
    assert "{'type': 'text'" not in result


def test_pretty_repr_list_with_non_text_blocks() -> None:
    """image_url blocks render with their URL, not as a raw dict."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Here is an image:"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
    )
    result = msg.pretty_repr()
    assert "Here is an image:" in result
    assert "[Image: http://example.com/img.png]" in result


def test_pretty_repr_mixed_list_content() -> None:
    """Mixed string and dict blocks all render correctly."""
    msg = HumanMessage(
        content=[
            "Look at this:",
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
            {"type": "text", "text": "What do you see?"},
        ]
    )
    result = msg.pretty_repr()
    assert "Look at this:" in result
    assert "[Image: http://example.com/img.png]" in result
    assert "What do you see?" in result


def test_pretty_repr_with_name() -> None:
    """Name is included in output for both string and list content."""
    msg = AIMessage(content=["Hello", "World"], name="bot")
    result = msg.pretty_repr()
    assert "Name: bot" in result
    assert "Hello" in result
    assert "World" in result


def test_pretty_repr_html_does_not_regress_string_content() -> None:
    """html=True still works correctly for plain string content."""
    msg = HumanMessage(content="Hello!")
    result = msg.pretty_repr(html=True)
    assert "Hello!" in result
    assert "Human Message" in result


def test_pretty_repr_html_list_content() -> None:
    """html=True works correctly for list content (no Python repr leak)."""
    msg = AIMessage(content=["Part one", "Part two"], name="assistant")
    result = msg.pretty_repr(html=True)
    assert "Part one" in result
    assert "Part two" in result
    assert "['Part one', 'Part two']" not in result
    assert "Name: assistant" in result


def test_pretty_repr_empty_list_content() -> None:
    """Empty list content renders without error."""
    msg = SystemMessage(content=[])
    result = msg.pretty_repr()
    assert "System Message" in result


def test_pretty_repr_unknown_block_type() -> None:
    """Unknown block type dict renders as [type] placeholder."""
    msg = AIMessage(content=[{"type": "tool_use", "id": "123", "name": "search"}])
    result = msg.pretty_repr()
    assert "[tool_use]" in result


# --- _render_block unit tests for multimodal block types ---


def test_render_block_image_url_with_url() -> None:
    """image_url block renders URL."""
    result = _render_block(
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
    )
    assert result == "[Image: https://example.com/img.png]"


def test_render_block_image_url_with_detail() -> None:
    """image_url block includes detail when present."""
    result = _render_block(
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png", "detail": "high"},
        }
    )
    assert result == "[Image: https://example.com/img.png, detail=high]"


def test_render_block_image_url_missing_url() -> None:
    """image_url block without url falls back to placeholder."""
    result = _render_block({"type": "image_url", "image_url": {}})
    assert result == "[image_url]"


def test_render_block_image_with_url() -> None:
    """Image block with url renders URL."""
    result = _render_block({"type": "image", "url": "https://example.com/photo.jpg"})
    assert result == "[Image: https://example.com/photo.jpg]"


def test_render_block_image_with_file_id() -> None:
    """Image block with file_id renders file_id."""
    result = _render_block({"type": "image", "file_id": "file-abc123"})
    assert result == "[Image: file_id=file-abc123]"


def test_render_block_image_with_base64() -> None:
    """Image block with base64 shows byte count and mime type."""
    result = _render_block(
        {"type": "image", "base64": "abc123", "mime_type": "image/jpeg"}
    )
    assert result == "[Image: base64 (6 bytes, image/jpeg)]"


def test_render_block_image_with_base64_no_mime() -> None:
    """Image block with base64 but no mime_type omits the mime suffix."""
    result = _render_block({"type": "image", "base64": "abc123"})
    assert result == "[Image: base64 (6 bytes)]"


def test_render_block_image_no_source() -> None:
    """Image block with no url/file_id/base64 renders as [Image]."""
    result = _render_block({"type": "image"})
    assert result == "[Image]"


def test_render_block_audio() -> None:
    """Audio block renders with URL."""
    result = _render_block(
        {"type": "audio", "url": "https://example.com/audio.mp3"}
    )
    assert result == "[Audio: https://example.com/audio.mp3]"


def test_render_block_video() -> None:
    """Video block renders with URL."""
    result = _render_block(
        {"type": "video", "url": "https://example.com/video.mp4"}
    )
    assert result == "[Video: https://example.com/video.mp4]"


def test_render_block_file_with_file_id() -> None:
    """File block renders with file_id."""
    result = _render_block({"type": "file", "file_id": "file-xyz"})
    assert result == "[File: file_id=file-xyz]"


def test_render_block_reasoning() -> None:
    """Reasoning block renders its text prefixed with [Reasoning]:."""
    result = _render_block({"type": "reasoning", "reasoning": "Let me think..."})
    assert result == "[Reasoning]: Let me think..."


def test_render_block_reasoning_empty() -> None:
    """Reasoning block with no text renders as [Reasoning]."""
    result = _render_block({"type": "reasoning"})
    assert result == "[Reasoning]"


def test_render_block_text_plain() -> None:
    """text-plain block renders its text content."""
    result = _render_block({"type": "text-plain", "text": "Plain text content"})
    assert result == "Plain text content"
