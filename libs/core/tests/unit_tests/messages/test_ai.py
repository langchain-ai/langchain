from langchain_core.load import dumpd, load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    InvalidToolCall,
    ToolCall,
    ToolCallChunk,
)


def test_serdes_message() -> None:
    msg = AIMessage(
        content=[{"text": "blah", "type": "text"}],
        tool_calls=[ToolCall(name="foo", args={"bar": 1}, id="baz")],
        invalid_tool_calls=[
            InvalidToolCall(name="foobad", args="blah", id="booz", error="bad")
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [{"name": "foo", "args": {"bar": 1}, "id": "baz"}],
            "invalid_tool_calls": [
                {"name": "foobad", "args": "blah", "id": "booz", "error": "bad"}
            ],
        },
    }
    actual = dumpd(msg)
    assert actual == expected
    assert load(actual) == msg


def test_serdes_message_chunk() -> None:
    chunk = AIMessageChunk(
        content=[{"text": "blah", "type": "text"}],
        tool_call_chunks=[
            ToolCallChunk(name="foo", args='{"bar": 1}', id="baz", index=0),
            ToolCallChunk(name="foobad", args="blah", id="booz", index=1),
        ],
    )
    expected = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessageChunk"],
        "kwargs": {
            "content": [{"text": "blah", "type": "text"}],
            "tool_calls": [{"name": "foo", "args": {"bar": 1}, "id": "baz"}],
            "invalid_tool_calls": [
                {
                    "name": "foobad",
                    "args": "blah",
                    "id": "booz",
                    "error": "Malformed args.",
                }
            ],
            "tool_call_chunks": [
                {"name": "foo", "args": '{"bar": 1}', "id": "baz", "index": 0},
                {"name": "foobad", "args": "blah", "id": "booz", "index": 1},
            ],
        },
    }
    actual = dumpd(chunk)
    assert actual == expected
    assert load(actual) == chunk
