"""Test parsers"""
from typing import Any, List, Optional, cast

import pytest

from langchain.automaton.runnables import (
    _apply_and_concat,
    RunnablePassthrough,
    _to_list,
    _to_runnable_parser,
    create_llm_program,
)
from langchain.automaton.tests.utils import (
    FakeChatModel,
)
from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage
from langchain.schema.runnable import RunnableLambda
from langchain.tools import BaseTool, tool


def test_apply_and_concat() -> None:
    """Test apply and concat."""
    msg = HumanMessage(content="Hello")
    llm = RunnableLambda(lambda *args, **kwargs: msg)

    # Test that it works with a runnable
    chain = llm | _apply_and_concat(RunnablePassthrough(), RunnablePassthrough())
    assert chain.invoke({}) == [msg, msg]

    chain = llm | _apply_and_concat(
        lambda msg: msg.content[0], lambda msg: msg.content[1]
    )
    assert chain.invoke({}) == ["H", "e"]


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (None, []),  # Test when input is None
        ([1, 2, 3], [1, 2, 3]),  # Test when input is a list of integers
        (5, [5]),  # Test when input is a single integer
        (["a", "b", "c"], ["a", "b", "c"]),  # Test when input is a list of strings
        ("xyz", ["xyz"]),  # Test when input is a single string
        ([], []),  # Test when input is an empty list
    ],
)
def test_to_list(input_value: Any, expected_output: List) -> None:
    assert _to_list(input_value) == expected_output


@pytest.mark.parametrize(
    "parser, expected_output",
    [
        (None, None),
        (RunnablePassthrough(), AIMessage(content="Hello")),
    ],
)
def to_runnable_parser(parser: Any, expected_output: Optional[BaseMessage]) -> None:
    """To runnable parser."""
    parser_ = _to_runnable_parser(parser)
    assert parser_.invoke(AIMessage(content="Hello")) == [expected_output]


@pytest.fixture()
def tools() -> List[BaseTool]:
    """Make a tools fixture."""

    @tool
    def get_time() -> str:
        """Get time."""
        return "9 PM"

    @tool
    def get_location() -> str:
        """Get location."""
        return "the park"

    return cast(List[BaseTool], [get_time, get_location])


def test_simple_llm_program() -> None:
    """Test simple llm program with no parser or tools."""
    get_time, _ = tools
    llm = FakeChatModel(
        message_iter=iter(
            [
                AIMessage(
                    content="Hello",
                ),
            ]
        )
    )

    program = create_llm_program(
        llm,
        prompt_generator=lambda x: x,
    )
    assert program.invoke("What time is it?") == [AIMessage(content="Hello")]


def test_llm_program_with_parser() -> None:
    """Test simple llm program with no parser or tools."""
    parser = RunnableLambda(lambda msg: AIMessage(content=msg.content + " parsed"))
    llm = FakeChatModel(
        message_iter=iter(
            [
                AIMessage(
                    content="Hello",
                ),
            ]
        )
    )

    program = create_llm_program(
        llm,
        prompt_generator=lambda x: x,
        parser=parser,
    )
    assert program.invoke("What time is it?") == [
        AIMessage(content="Hello"),
        AIMessage(content="Hello parsed"),
    ]


def test_llm_program_with_parser_and_tools(tools: List[BaseTool]) -> None:
    """Test simple llm program with no parser or tools."""
    parser = RunnableLambda(lambda msg: AIMessage(content=msg.content + " parsed"))
    llm = FakeChatModel(
        message_iter=iter(
            [
                AIMessage(
                    content="Hello",
                ),
            ]
        )
    )

    program = create_llm_program(
        llm,
        prompt_generator=lambda x: x,
        parser=parser,
        tools=tools,
    )
    assert program.invoke("What time is it?") == [
        AIMessage(content="Hello"),
        AIMessage(content="Hello parsed"),
    ]
