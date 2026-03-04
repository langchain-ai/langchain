from typing import Any

import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("foo", "foo"),
        (["foo"], "foo"),
        (["foo", "bar"], "foobar"),
        ([{"text": "foo", "type": "text"}], "foo"),
        (
            [
                {"type": "text", "text": "foo"},
                {"type": "reasoning", "reasoning": "..."},
                {"type": "text", "text": "bar"},
            ],
            "foobar",
        ),
        ([{"text": "foo"}], "foo"),
        ([{"text": "foo"}, "bar"], "foobar"),
    ],
)
def test_msg_with_text(
    content: str | list[str | dict[str, Any]], expected: str
) -> None:
    actual = ChatGeneration(message=AIMessage(content=content)).text
    assert actual == expected


@pytest.mark.parametrize("content", [[], [{"tool_use": {}, "type": "tool_use"}]])
def test_msg_no_text(content: str | list[str | dict[str, Any]]) -> None:
    expected = ""
    actual = ChatGeneration(message=AIMessage(content=content)).text
    assert actual == expected
