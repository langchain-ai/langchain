from collections.abc import Callable

import pytest

from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.utils import (
    get_function_nonlocals,
    get_lambda_source,
    indent_lines_after_first,
    as_message_state,
    to_message_state
)
from langchain_core.messages import (
    BaseMessage,
    HumanMessage
)


@pytest.mark.parametrize(
    ("func", "expected_source"),
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),  # noqa: FURB136
    ],
)
def test_get_lambda_source(func: Callable, expected_source: str) -> None:
    """Test get_lambda_source function."""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    ("text", "prefix", "expected_output"),
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function."""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output


global_agent = RunnableLambda(lambda x: x * 3)


def test_nonlocals() -> None:
    agent = RunnableLambda(lambda x: x * 2)

    def my_func(value: str, agent: dict[str, str]) -> str:
        return agent.get("agent_name", value)

    def my_func2(value: str) -> str:
        return agent.get("agent_name", value)  # type: ignore[attr-defined]

    def my_func3(value: str) -> str:
        return agent.invoke(value)

    def my_func4(value: str) -> str:
        return global_agent.invoke(value)

    def my_func5() -> tuple[Callable[[str], str], RunnableLambda]:
        global_agent = RunnableLambda(lambda x: x * 3)

        def my_func6(value: str) -> str:
            return global_agent.invoke(value)

        return my_func6, global_agent

    assert get_function_nonlocals(my_func) == []
    assert get_function_nonlocals(my_func2) == []
    assert get_function_nonlocals(my_func3) == [agent.invoke]
    assert get_function_nonlocals(my_func4) == [global_agent.invoke]
    func, nl = my_func5()
    assert get_function_nonlocals(func) == [nl.invoke]
    assert RunnableLambda(my_func3).deps == [agent]
    assert RunnableLambda(my_func4).deps == [global_agent]
    assert RunnableLambda(func).deps == [nl]


def test_to_message_state_none() -> None:
    result = to_message_state(None)
    assert result == {"messages": []}


def test_to_message_state_string() -> None:
    result = to_message_state("hello")
    assert isinstance(result["messages"][0], HumanMessage)
    assert result == {"messages": [HumanMessage(content="hello")]}


def test_to_message_state_human_message() -> None:
    msg = HumanMessage(content="hey")
    result = to_message_state(msg)
    assert result == {"messages": [msg]}


def test_to_message_state_ai_message() -> None:
    msg = AIMessage(content="ok")
    result = to_message_state(msg)
    assert result == {"messages": [msg]}


def test_to_message_state_system_message() -> None:
    msg = SystemMessage(content="sys state")
    result = to_message_state(msg)
    assert result == {"messages": [msg]}


def test_to_message_state_list_of_mixed_items() -> None:
    items = [HumanMessage(content="h"), "a", AIMessage(content="x")]
    result = to_message_state(items)
    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert messages[0].content == "h"
    assert messages[1].content == "a"
    assert messages[2].content == "x"


def test_to_message_state_existing_message_state_dict() -> None:
    msg = HumanMessage(content="ok")
    result = to_message_state({"messages": [msg]})
    assert result == {"messages": [msg]}


def test_to_message_state_invalid_type() -> None:
    with pytest.raises(TypeError):
        to_message_state(123)  # type: ignore[arg-type]


def test_as_message_state_wraps_string() -> None:
    chain = as_message_state()
    result = chain.invoke("hello")
    assert result == {"messages": [HumanMessage(content="hello")]}


def test_as_message_state_wraps_ai_message() -> None:
    chain = as_message_state()
    result = chain.invoke(AIMessage(content="ok"))
    assert result == {"messages": [AIMessage(content="ok")]}


def test_as_message_state_with_preceding_lc_chain() -> None:
    llm_output = AIMessage(content="done")

    def fake_llm(_: str) -> BaseMessage:
        return llm_output

    chain = RunnableLambda(fake_llm) | as_message_state()
    result = chain.invoke("hello")
    assert result == {"messages": [llm_output]}


def test_as_message_state_with_list_input() -> None:
    chain = as_message_state()
    items = ["a", "b", AIMessage(content="x")]
    result = chain.invoke(items)
    msgs = result["messages"]
    assert msgs[0].content == "a"
    assert msgs[1].content == "b"
    assert msgs[2].content == "x"
