from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from pydantic import BaseModel

from langchain_anthropic.output_parsers import ToolsOutputParser

_CONTENT: list = [
    {
        "type": "text",
        "text": "thought",
    },
    {"type": "tool_use", "input": {"bar": 0}, "id": "1", "name": "_Foo1"},
    {
        "type": "text",
        "text": "thought",
    },
    {"type": "tool_use", "input": {"baz": "a"}, "id": "2", "name": "_Foo2"},
]

_RESULT: list = [ChatGeneration(message=AIMessage(_CONTENT))]  # type: ignore[misc]


class _Foo1(BaseModel):
    bar: int


class _Foo2(BaseModel):
    baz: Literal["a", "b"]


def test_tools_output_parser() -> None:
    output_parser = ToolsOutputParser()
    expected = [
        {
            "name": "_Foo1",
            "args": {"bar": 0},
            "id": "1",
            "index": 1,
            "type": "tool_call",
        },
        {
            "name": "_Foo2",
            "args": {"baz": "a"},
            "id": "2",
            "index": 3,
            "type": "tool_call",
        },
    ]
    actual = output_parser.parse_result(_RESULT)
    assert expected == actual


def test_tools_output_parser_args_only() -> None:
    output_parser = ToolsOutputParser(args_only=True)
    expected = [
        {"bar": 0},
        {"baz": "a"},
    ]
    actual = output_parser.parse_result(_RESULT)
    assert expected == actual

    expected = []
    actual = output_parser.parse_result([ChatGeneration(message=AIMessage(""))])  # type: ignore[misc]
    assert expected == actual


def test_tools_output_parser_first_tool_only() -> None:
    output_parser = ToolsOutputParser(first_tool_only=True)
    expected: Any = {
        "name": "_Foo1",
        "args": {"bar": 0},
        "id": "1",
        "index": 1,
        "type": "tool_call",
    }
    actual = output_parser.parse_result(_RESULT)
    assert expected == actual

    expected = None
    actual = output_parser.parse_result([ChatGeneration(message=AIMessage(""))])  # type: ignore[misc]
    assert expected == actual


def test_tools_output_parser_pydantic() -> None:
    output_parser = ToolsOutputParser(pydantic_schemas=[_Foo1, _Foo2])
    expected = [_Foo1(bar=0), _Foo2(baz="a")]
    actual = output_parser.parse_result(_RESULT)
    assert expected == actual


def test_tools_output_parser_empty_content() -> None:
    class ChartType(BaseModel):
        chart_type: Literal["pie", "line", "bar"]

    output_parser = ToolsOutputParser(
        first_tool_only=True, pydantic_schemas=[ChartType]
    )
    message = AIMessage(
        "",
        tool_calls=[
            {
                "name": "ChartType",
                "args": {"chart_type": "pie"},
                "id": "foo",
                "type": "tool_call",
            }
        ],
    )
    actual = output_parser.invoke(message)
    expected = ChartType(chart_type="pie")
    assert expected == actual
