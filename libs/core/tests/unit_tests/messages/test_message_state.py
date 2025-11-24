from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import as_message_state, to_message_state


def test_string_to_message_state() -> None:
    result = to_message_state("hello")
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"


def test_base_message_preserved() -> None:
    msg = AIMessage(content="hi")
    result = to_message_state(msg)
    assert result["messages"][0] is msg


def test_list_of_strings() -> None:
    result = to_message_state(["a", "b"])
    msgs = result["messages"]
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "a"
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "b"


def test_list_mixed_message_and_string() -> None:
    m = AIMessage(content="x")
    result = to_message_state(["hello", m])
    msgs = result["messages"]
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "hello"
    assert msgs[1] is m


def test_message_state_passthrough() -> None:
    existing = {"messages": [HumanMessage(content="y")]}
    result = to_message_state(existing)
    # should return the original dict, not copy
    assert result is existing


def test_none_returns_empty_list() -> None:
    result = to_message_state(None)
    assert result == {"messages": []}


def test_as_message_state_wrapper() -> None:
    fn = as_message_state()
    out = fn("yo")
    msg = out["messages"][0]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "yo"
