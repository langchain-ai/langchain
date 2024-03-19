from typing import Any, Callable, Dict, List, Literal, Optional, Type

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    tool_example_to_messages,
)


@pytest.fixture()
def pydantic() -> Type[BaseModel]:
    class dummy_function(BaseModel):
        """dummy function"""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def function() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """
        pass

    return dummy_function


@pytest.fixture()
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):
        args_schema: Type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "dummy function"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture()
def json_schema() -> Dict:
    return {
        "title": "dummy_function",
        "description": "dummy function",
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    }


def test_convert_to_openai_function(
    pydantic: Type[BaseModel],
    function: Callable,
    dummy_tool: BaseTool,
    json_schema: Dict,
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }

    for fn in (pydantic, function, dummy_tool, json_schema, expected):
        actual = convert_to_openai_function(fn)  # type: ignore
        assert actual == expected


@pytest.mark.xfail(reason="Pydantic converts Optional[str] to str in .schema()")
def test_function_optional_param() -> None:
    @tool
    def func5(
        a: Optional[str],
        b: str,
        c: Optional[List[Optional[str]]],
    ) -> None:
        """A test function"""
        pass

    func = convert_to_openai_function(func5)
    req = func["parameters"]["required"]
    assert set(req) == {"b"}


class FakeCall(BaseModel):
    data: str


def test_valid_example_conversion() -> None:
    expected_messages = [
        HumanMessage(content="This is a valid example"),
        AIMessage(content="", additional_kwargs={"tool_calls": []}),
    ]
    assert (
        tool_example_to_messages(input="This is a valid example", tool_calls=[])
        == expected_messages
    )


def test_multiple_tool_calls() -> None:
    messages = tool_example_to_messages(
        input="This is an example",
        tool_calls=[
            FakeCall(data="ToolCall1"),
            FakeCall(data="ToolCall2"),
            FakeCall(data="ToolCall3"),
        ],
    )
    assert len(messages) == 5
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert isinstance(messages[3], ToolMessage)
    assert isinstance(messages[4], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall1"}'},
        },
        {
            "id": messages[3].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall2"}'},
        },
        {
            "id": messages[4].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall3"}'},
        },
    ]


def test_tool_outputs() -> None:
    messages = tool_example_to_messages(
        input="This is an example",
        tool_calls=[
            FakeCall(data="ToolCall1"),
        ],
        tool_outputs=["Output1"],
    )
    assert len(messages) == 3
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert isinstance(messages[2], ToolMessage)
    assert messages[1].additional_kwargs["tool_calls"] == [
        {
            "id": messages[2].tool_call_id,
            "type": "function",
            "function": {"name": "FakeCall", "arguments": '{"data": "ToolCall1"}'},
        },
    ]
    assert messages[2].content == "Output1"
