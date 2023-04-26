import logging
from functools import partial
from typing import Any, Optional, Type

import pydantic
import pytest
from pydantic import BaseModel

from langchain.tools.structured import (
    BaseStructuredTool,
    StructuredTool,
    structured_tool,
)


class _MockSchema(BaseModel):
    arg1: int
    arg2: bool
    arg3: Optional[dict] = None


class _MockStructuredTool(BaseStructuredTool):
    name = "structured_api"
    args_schema: Type[BaseModel] = _MockSchema
    description = "A Structured Tool"

    def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        return f"{arg1} {arg2} {arg3}"

    async def _arun(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        raise NotImplementedError


def test_structured_args() -> None:
    """Test functionality with structured arguments."""
    structured_api = _MockStructuredTool()
    assert isinstance(structured_api, BaseStructuredTool)
    assert structured_api.name == "structured_api"
    expected_result = "1 True {'foo': 'bar'}"
    args = {"arg1": 1, "arg2": True, "arg3": {"foo": "bar"}}
    assert structured_api.run(args) == expected_result


def test_subclass_annotated_base_tool_accepted() -> None:
    """Test BaseTool child w/ custom schema isn't overwritten."""

    class _ForwardRefAnnotatedTool(BaseStructuredTool):
        name = "structured_api"
        args_schema: Type[_MockSchema] = _MockSchema
        description = "A Structured Tool"

        def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError

    assert issubclass(_ForwardRefAnnotatedTool, BaseStructuredTool)
    tool = _ForwardRefAnnotatedTool()
    assert tool.args_schema == _MockSchema


def test_decorator_with_specified_schema() -> None:
    """Test that manually specified schemata are passed through to the tool."""

    @structured_tool(args_schema=_MockSchema)
    def tool_func(arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        """Return the arguments directly."""
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func, StructuredTool)
    assert tool_func.args_schema == _MockSchema


def test_decorated_function_schema_equivalent() -> None:
    """Test that a BaseTool without a schema meets expectations."""

    @structured_tool
    def structured_tool_input(
        arg1: int, arg2: bool, arg3: Optional[dict] = None
    ) -> str:
        """Return the arguments directly."""
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(structured_tool_input, StructuredTool)
    assert (
        structured_tool_input.args_schema.schema()["properties"]
        == _MockSchema.schema()["properties"]
        == structured_tool_input.args
    )


def test_tool_lambda_multi_args_schema() -> None:
    """Test args schema inference when the tool argument is a lambda function."""
    tool = StructuredTool.from_function(
        func=lambda tool_input, other_arg: f"{tool_input}{other_arg}",  # type: ignore
        name="tool",
        description="A tool",
    )
    assert set(tool.args_schema.schema()["properties"]) == {"tool_input", "other_arg"}
    expected_args = {
        "tool_input": {"title": "Tool Input"},
        "other_arg": {"title": "Other Arg"},
    }
    assert tool.args == expected_args


def test_tool_partial_function_args_schema() -> None:
    """Test args schema inference when the tool argument is a partial function."""

    def func(tool_input: str, other_arg: str) -> str:
        return tool_input + other_arg

    with pytest.raises(pydantic.error_wrappers.ValidationError):
        # We don't yet support args_schema inference for partial functions
        # so want to make sure we proactively raise an error
        StructuredTool(
            name="tool",
            description="A tool",
            func=partial(func, other_arg="foo"),
        )


def test_tool_with_kwargs() -> None:
    """Test functionality when only return direct is provided."""

    @structured_tool(return_direct=True)
    def search_api(
        arg_1: float,
        ping: str = "hi",
    ) -> str:
        """Search the API for the query."""
        return f"arg_1={arg_1}, ping={ping}"

    assert isinstance(search_api, StructuredTool)
    result = search_api.run(
        tool_input={
            "arg_1": 3.2,
            "ping": "pong",
        }
    )
    assert result == "arg_1=3.2, ping=pong"

    result = search_api.run(
        tool_input={
            "arg_1": 3.2,
        }
    )
    assert result == "arg_1=3.2, ping=hi"


def test_empty_args_decorator() -> None:
    """Test inferred schema of decorated fn with no args."""

    @structured_tool
    def empty_tool_input() -> str:
        """Return a constant."""
        return "the empty result"

    assert isinstance(empty_tool_input, StructuredTool)
    assert empty_tool_input.name == "empty_tool_input"
    assert empty_tool_input.args == {}
    assert empty_tool_input.run({}) == "the empty result"


def test_nested_pydantic_args() -> None:
    """Test inferred schema when args are nested pydantic models."""
    # This is a pattern that is common with FastAPI methods.
    # If we only parse a dict input but pass the dict
    # to the function, we are limited only to primitive types
    # in general.

    class SomeNestedInput(BaseModel):
        arg2: str

    class SomeInput(BaseModel):
        arg1: int
        arg2: SomeNestedInput

    @structured_tool
    def nested_tool(some_input: SomeInput) -> dict:
        """Return a constant."""
        return some_input.dict()

    assert isinstance(nested_tool, StructuredTool)
    assert nested_tool.name == "nested_tool"
    input_ = {"some_input": {"arg1": 1, "arg2": {"arg2": "foo"}}}
    assert nested_tool.run(input_) == input_["some_input"]


def test_warning_on_args_kwargs(caplog: pytest.LogCaptureFixture) -> None:
    """Test inferred schema when args are nested pydantic models."""

    with caplog.at_level(logging.WARNING):

        @structured_tool
        def anything_goes(*foo: Any, **bar: Any) -> str:
            """Return a constant."""
            return str(foo) + "|" + str(bar)

    # Check if the expected warning message was logged
    assert any(
        "anything_goes uses *args" in record.message for record in caplog.records
    )
    assert any(
        "anything_goes uses **kwargs" in record.message for record in caplog.records
    )
