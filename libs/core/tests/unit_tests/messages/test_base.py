"""Tests for BaseMessage.pretty_repr()."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


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
    """Non-text block types render as [type] placeholders."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Here is an image:"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
    )
    result = msg.pretty_repr()
    assert "Here is an image:" in result
    assert "[image_url]" in result


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
    assert "[image_url]" in result
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
