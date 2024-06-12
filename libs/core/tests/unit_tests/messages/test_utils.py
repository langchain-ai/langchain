from typing import Type

import pytest

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.utils import (
    filter_messages,
    merge_message_runs,
    trim_messages,
)


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str(msg_cls: Type[BaseMessage]) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    expected = [msg_cls("foo\nbar\nbaz")]
    actual = merge_message_runs(messages)
    assert actual == expected


def test_merge_message_runs_content() -> None:
    messages = [
        HumanMessage("foo", id="1"),
        HumanMessage(
            [
                {"text": "bar", "type": "text"},
                {"image_url": "...", "type": "image_url"},
            ],
            id="2",
        ),
        HumanMessage("baz", id="3"),
    ]
    expected = [
        HumanMessage(
            [
                "foo",
                {"text": "bar", "type": "text"},
                {"image_url": "...", "type": "image_url"},
                "baz",
            ],
            id="1",
        ),
    ]
    actual = merge_message_runs(messages)
    assert actual == expected


def test_filter_message() -> None:
    filter_messages


def test_trim_messages() -> None:
    trim_messages
