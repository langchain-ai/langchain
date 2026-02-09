from langchain_core.messages import BaseMessage


def test_pretty_repr_html_formats_blocks_as_html() -> None:
    message = BaseMessage(content=["First", {"type": "text", "text": "Second"}], type="ai", name="bot")

    rendered = message.pretty_repr(html=True)

    assert '<div class="lc-message"' in rendered
    assert "First" in rendered
    assert "Second" in rendered
    # Ensure we don't fall back to Python list repr
    assert "['First'" not in rendered


def test_pretty_repr_formats_reasoning_block() -> None:
    message = BaseMessage(content=[{"type": "reasoning", "reasoning": "step one"}], type="ai")

    rendered_html = message.pretty_repr(html=True)
    assert "<pre" in rendered_html
    assert "step one" in rendered_html

    rendered_text = message.pretty_repr()
    assert "step one" in rendered_text
