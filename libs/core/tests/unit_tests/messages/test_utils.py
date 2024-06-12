from langchain_core.messages import AIMessage
from langchain_core.messages.utils import (
    filter_messages,
    merge_message_runs,
    trim_messages,
)


def test_merge_message_runs_str() -> None:
    messages = [AIMessage("foo"), AIMessage("bar"), AIMessage("baz")]
    expected = [AIMessage("foobar")]
    actual = merge_message_runs(messages)
    actual == expected


def test_merge_message_runs_content() -> None:
    messages = [
        AIMessage("foo"),
        AIMessage(
            [{"text": "bar", "type": "text"}, {"image_url": "...", "type": "image_url"}]
        ),
        AIMessage("baz"),
    ]
    expected = [
        AIMessage(
            [
                "foo",
                {"text": "bar", "type": "text"},
                {"image_url": "...", "type": "image_url"},
                "baz",
            ]
        ),
    ]
    actual = merge_message_runs(messages)
    assert actual == expected


def test_filter_message() -> None:
    filter_messages


def test_trim_messages() -> None:
    trim_messages
