from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def test_msg_with_text() -> None:
    msgs = [
        AIMessage("foo"),
        AIMessage(["foo"]),
        AIMessage([{"text": "foo", "type": "text"}]),
        AIMessage(
            [
                {"tool_use": {}, "type": "tool_use"},
                {"text": "foo", "type": "text"},
                "bar",
            ]
        ),
    ]
    expected = "foo"
    for msg in msgs:
        actual = ChatGeneration(message=msg).text
        assert actual == expected


def test_msg_no_text() -> None:
    msgs = [
        AIMessage([]),
        AIMessage(
            [
                {"tool_use": {}, "type": "tool_use"},
            ]
        ),
    ]
    expected = ""
    for msg in msgs:
        actual = ChatGeneration(message=msg).text
        assert actual == expected
