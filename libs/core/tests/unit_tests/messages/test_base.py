"""Tests for BaseMessage.pretty_repr with non-string content."""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def test_pretty_repr_string_content() -> None:
    """String content should be returned as-is."""
    msg = HumanMessage(content="Hello, world!")
    result = msg.pretty_repr()
    assert "Hello, world!" in result


def test_pretty_repr_list_of_strings() -> None:
    """List of plain strings should be joined with newlines."""
    msg = HumanMessage(content=["First", "Second", "Third"])
    result = msg.pretty_repr()
    assert "First" in result
    assert "Second" in result
    assert "Third" in result
    # Should NOT contain raw Python list repr
    assert "['First'" not in result


def test_pretty_repr_text_blocks() -> None:
    """Text content blocks should have their text extracted."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Hello from text block"},
            {"type": "text", "text": "Another block"},
        ]
    )
    result = msg.pretty_repr()
    assert "Hello from text block" in result
    assert "Another block" in result


def test_pretty_repr_image_url_block() -> None:
    """Image URL blocks should be rendered with a placeholder."""
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Look at this:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.png"},
            },
        ]
    )
    result = msg.pretty_repr()
    assert "Look at this:" in result
    assert "[Image:" in result
    assert "https://example.com/img.png" in result


def test_pretty_repr_mixed_content() -> None:
    """Mixed string and dict content should all be formatted."""
    msg = AIMessage(
        content=[
            "Plain text first",
            {"type": "text", "text": "Text block"},
            {"type": "custom", "data": "something"},
        ],
        name="bot",
    )
    result = msg.pretty_repr(html=True)
    assert "Plain text first" in result
    assert "Text block" in result
    assert '"type": "custom"' in result
    assert "Name: bot" in result


def test_pretty_repr_html_flag() -> None:
    """html=True should still work for bold title."""
    msg = HumanMessage(content=["Hello"])
    result_plain = msg.pretty_repr(html=False)
    result_html = msg.pretty_repr(html=True)
    # Both should contain the message text
    assert "Hello" in result_plain
    assert "Hello" in result_html


def test_format_content_with_empty_list() -> None:
    """Empty list content should produce empty string."""
    result = BaseMessage._format_content([])
    assert result == ""


def test_format_content_preserves_string() -> None:
    """String content should be returned unchanged."""
    result = BaseMessage._format_content("just a string")
    assert result == "just a string"
