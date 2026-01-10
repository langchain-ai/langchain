from typing import Any

import pytest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


@pytest.mark.parametrize(
    "content",
    [
        "foo",
        ["foo"],
        [{"text": "foo", "type": "text"}],
        [
            {"tool_use": {}, "type": "tool_use"},
            {"text": "foo", "type": "text"},
            "bar",
        ],
    ],
)
def test_msg_with_text(content: str | list[str | dict[str, Any]]) -> None:
    expected = "foo"
    actual = ChatGeneration(message=AIMessage(content=content)).text
    assert actual == expected


@pytest.mark.parametrize("content", [[], [{"tool_use": {}, "type": "tool_use"}]])
def test_msg_no_text(content: str | list[str | dict[str, Any]]) -> None:
    expected = ""
    actual = ChatGeneration(message=AIMessage(content=content)).text
    assert actual == expected
