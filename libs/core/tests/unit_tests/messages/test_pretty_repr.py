
import pytest
from langchain_core.messages import HumanMessage, AIMessage

def test_pretty_repr_string_content() -> None:
    msg = HumanMessage(content="Hello World")
    assert "Hello World" in msg.pretty_repr()

def test_pretty_repr_list_text_content() -> None:
    msg = HumanMessage(content=[
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " World"}
    ])
    assert "Hello World" in msg.pretty_repr()

def test_pretty_repr_mixed_content() -> None:
    msg = HumanMessage(content=[
        {"type": "text", "text": "Look at this: "},
        {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}
    ])
    repr_output = msg.pretty_repr()
    assert "Look at this: " in repr_output
    assert "[image_url]" in repr_output

def test_pretty_repr_tool_call_content() -> None:
    # content block style tool call (e.g. Anthropic)
    msg = AIMessage(content=[
        {"type": "text", "text": "I will use a tool."},
        {"type": "tool_use", "name": "calculator", "id": "123"}
    ])
    repr_output = msg.pretty_repr()
    assert "I will use a tool." in repr_output
    assert "[tool_use: calculator]" in repr_output


def test_pretty_repr_unknown_block_content() -> None:
    msg = AIMessage(content=[
        {"type": "unknown_type", "value": "mystery"}
    ])
    # Should fallback to string representation of the dict
    assert "{'type': 'unknown_type', 'value': 'mystery'}" in msg.pretty_repr()
