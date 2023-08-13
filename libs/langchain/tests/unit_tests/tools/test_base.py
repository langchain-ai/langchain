"""Test the base tool implementation."""
import json
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Any, List, Optional, Type, Union

import pytest
from pydantic import BaseModel

from langchain.agents.tools import Tool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import (
    BaseTool,
    SchemaAnnotationError,
    StructuredTool,
    ToolException,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_unnamed_decorator() -> None:
    """Test functionality with unnamed decorator."""

    @tool
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, BaseTool)
    assert search_api.name == "search_api"
    assert not search_api.return_direct
    assert search_api("test") == "API result"


class _MockSchema(BaseModel):
    arg1: int
    arg2: bool
    arg3: Optional[dict] = None


class _MockStructuredTool(BaseTool):
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
    assert isinstance(structured_api, BaseTool)
    assert structured_api.name == "structured_api"
    expected_result = "1 True {'foo': 'bar'}"
    args = {"arg1": 1, "arg2": True, "arg3": {"foo": "bar"}}
    assert structured_api.run(args) == expected_result


def test_unannotated_base_tool_raises_error() -> None:
    """Test that a BaseTool without type hints raises an exception.""" ""
    with pytest.raises(SchemaAnnotationError):

        class _UnAnnotatedTool(BaseTool):
            name = "structured_api"
            # This would silently be ignored without the custom metaclass
            args_schema = _MockSchema
            description = "A Structured Tool"

            def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
                return f"{arg1} {arg2} {arg3}"

            async def _arun(
                self, arg1: int, arg2: bool, arg3: Optional[dict] = None
            ) -> str:
                raise NotImplementedError


def test_misannotated_base_tool_raises_error() -> None:
    """Test that a BaseTool with the incorrect typehint raises an exception.""" ""
    with pytest.raises(SchemaAnnotationError):

        class _MisAnnotatedTool(BaseTool):
            name = "structured_api"
            # This would silently be ignored without the custom metaclass
            args_schema: BaseModel = _MockSchema  # type: ignore
            description = "A Structured Tool"

            def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
                return f"{arg1} {arg2} {arg3}"

            async def _arun(
                self, arg1: int, arg2: bool, arg3: Optional[dict] = None
            ) -> str:
                raise NotImplementedError


def test_forward_ref_annotated_base_tool_accepted() -> None:
    """Test that a using forward ref annotation syntax is accepted.""" ""

    class _ForwardRefAnnotatedTool(BaseTool):
        name = "structured_api"
        args_schema: "Type[BaseModel]" = _MockSchema
        description = "A Structured Tool"

        def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError


def test_subclass_annotated_base_tool_accepted() -> None:
    """Test BaseTool child w/ custom schema isn't overwritten."""

    class _ForwardRefAnnotatedTool(BaseTool):
        name = "structured_api"
        args_schema: Type[_MockSchema] = _MockSchema
        description = "A Structured Tool"

        def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError

    assert issubclass(_ForwardRefAnnotatedTool, BaseTool)
    tool = _ForwardRefAnnotatedTool()
    assert tool.args_schema == _MockSchema


def test_decorator_with_specified_schema() -> None:
    """Test that manually specified schemata are passed through to the tool."""

    @tool(args_schema=_MockSchema)
    def tool_func(arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        """Return the arguments directly."""
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func, BaseTool)
    assert tool_func.args_schema == _MockSchema


def test_decorated_function_schema_equivalent() -> None:
    """Test that a BaseTool without a schema meets expectations."""

    @tool
    def structured_tool_input(
        arg1: int, arg2: bool, arg3: Optional[dict] = None
    ) -> str:
        """Return the arguments directly."""
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(structured_tool_input, BaseTool)
    assert structured_tool_input.args_schema is not None
    assert (
        structured_tool_input.args_schema.schema()["properties"]
        == _MockSchema.schema()["properties"]
        == structured_tool_input.args
    )


def test_args_kwargs_filtered() -> None:
    class _SingleArgToolWithKwargs(BaseTool):
        name = "single_arg_tool"
        description = "A  single arged tool with kwargs"

        def _run(
            self,
            some_arg: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any,
        ) -> str:
            return "foo"

        async def _arun(
            self,
            some_arg: str,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
            **kwargs: Any,
        ) -> str:
            raise NotImplementedError

    tool = _SingleArgToolWithKwargs()
    assert tool.is_single_input

    class _VarArgToolWithKwargs(BaseTool):
        name = "single_arg_tool"
        description = "A single arged tool with kwargs"

        def _run(
            self,
            *args: Any,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any,
        ) -> str:
            return "foo"

        async def _arun(
            self,
            *args: Any,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
            **kwargs: Any,
        ) -> str:
            raise NotImplementedError

    tool2 = _VarArgToolWithKwargs()
    assert tool2.is_single_input


def test_structured_args_decorator_no_infer_schema() -> None:
    """Test functionality with structured arguments parsed as a decorator."""

    @tool(infer_schema=False)
    def structured_tool_input(
        arg1: int, arg2: Union[float, datetime], opt_arg: Optional[dict] = None
    ) -> str:
        """Return the arguments directly."""
        return f"{arg1}, {arg2}, {opt_arg}"

    assert isinstance(structured_tool_input, BaseTool)
    assert structured_tool_input.name == "structured_tool_input"
    args = {"arg1": 1, "arg2": 0.001, "opt_arg": {"foo": "bar"}}
    with pytest.raises(ToolException):
        assert structured_tool_input.run(args)


def test_structured_single_str_decorator_no_infer_schema() -> None:
    """Test functionality with structured arguments parsed as a decorator."""

    @tool(infer_schema=False)
    def unstructured_tool_input(tool_input: str) -> str:
        """Return the arguments directly."""
        assert isinstance(tool_input, str)
        return f"{tool_input}"

    assert isinstance(unstructured_tool_input, BaseTool)
    assert unstructured_tool_input.args_schema is None
    assert unstructured_tool_input.run("foo") == "foo"


def test_structured_tool_types_parsed() -> None:
    """Test the non-primitive types are correctly passed to structured tools."""

    class SomeEnum(Enum):
        A = "a"
        B = "b"

    class SomeBaseModel(BaseModel):
        foo: str

    @tool
    def structured_tool(
        some_enum: SomeEnum,
        some_base_model: SomeBaseModel,
    ) -> dict:
        """Return the arguments directly."""
        return {
            "some_enum": some_enum,
            "some_base_model": some_base_model,
        }

    assert isinstance(structured_tool, StructuredTool)
    args = {
        "some_enum": SomeEnum.A.value,
        "some_base_model": SomeBaseModel(foo="bar").dict(),
    }
    result = structured_tool.run(json.loads(json.dumps(args)))
    expected = {
        "some_enum": SomeEnum.A,
        "some_base_model": SomeBaseModel(foo="bar"),
    }
    assert result == expected


def test_base_tool_inheritance_base_schema() -> None:
    """Test schema is correctly inferred when inheriting from BaseTool."""

    class _MockSimpleTool(BaseTool):
        name = "simple_tool"
        description = "A Simple Tool"

        def _run(self, tool_input: str) -> str:
            return f"{tool_input}"

        async def _arun(self, tool_input: str) -> str:
            raise NotImplementedError

    simple_tool = _MockSimpleTool()
    assert simple_tool.args_schema is None
    expected_args = {"tool_input": {"title": "Tool Input", "type": "string"}}
    assert simple_tool.args == expected_args


def test_tool_lambda_args_schema() -> None:
    """Test args schema inference when the tool argument is a lambda function."""

    tool = Tool(
        name="tool",
        description="A tool",
        func=lambda tool_input: tool_input,
    )
    assert tool.args_schema is None
    expected_args = {"tool_input": {"type": "string"}}
    assert tool.args == expected_args


def test_structured_tool_from_function_docstring() -> None:
    """Test that structured tools can be created from functions."""

    def foo(bar: int, baz: str) -> str:
        """Docstring
        Args:
            bar: int
            baz: str
        """
        raise NotImplementedError()

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "string"},
    }

    assert structured_tool.args_schema.schema() == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "title": "fooSchemaSchema",
        "type": "object",
        "required": ["bar", "baz"],
    }

    prefix = "foo(bar: int, baz: str) -> str - "
    assert foo.__doc__ is not None
    assert structured_tool.description == prefix + foo.__doc__.strip()


def test_structured_tool_from_function_docstring_complex_args() -> None:
    """Test that structured tools can be created from functions."""

    def foo(bar: int, baz: List[str]) -> str:
        """Docstring
        Args:
            bar: int
            baz: List[str]
        """
        raise NotImplementedError()

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "array", "items": {"type": "string"}},
    }

    assert structured_tool.args_schema.schema() == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "array", "items": {"type": "string"}},
        },
        "title": "fooSchemaSchema",
        "type": "object",
        "required": ["bar", "baz"],
    }

    prefix = "foo(bar: int, baz: List[str]) -> str - "
    assert foo.__doc__ is not None
    assert structured_tool.description == prefix + foo.__doc__.strip()


def test_structured_tool_lambda_multi_args_schema() -> None:
    """Test args schema inference when the tool argument is a lambda function."""
    tool = StructuredTool.from_function(
        name="tool",
        description="A tool",
        func=lambda tool_input, other_arg: f"{tool_input}{other_arg}",  # type: ignore
    )
    assert tool.args_schema is not None
    expected_args = {
        "tool_input": {"title": "Tool Input"},
        "other_arg": {"title": "Other Arg"},
    }
    assert tool.args == expected_args


def test_tool_partial_function_args_schema() -> None:
    """Test args schema inference when the tool argument is a partial function."""

    def func(tool_input: str, other_arg: str) -> str:
        assert isinstance(tool_input, str)
        assert isinstance(other_arg, str)
        return tool_input + other_arg

    tool = Tool(
        name="tool",
        description="A tool",
        func=partial(func, other_arg="foo"),
    )
    assert tool.run("bar") == "barfoo"


def test_empty_args_decorator() -> None:
    """Test inferred schema of decorated fn with no args."""

    @tool
    def empty_tool_input() -> str:
        """Return a constant."""
        return "the empty result"

    assert isinstance(empty_tool_input, BaseTool)
    assert empty_tool_input.name == "empty_tool_input"
    assert empty_tool_input.args == {}
    assert empty_tool_input.run({}) == "the empty result"


def test_tool_from_function_with_run_manager() -> None:
    """Test run of tool when using run_manager."""

    def foo(bar: str, callbacks: Optional[CallbackManagerForToolRun] = None) -> str:
        """Docstring
        Args:
            bar: str
        """
        assert callbacks is not None
        return "foo" + bar

    handler = FakeCallbackHandler()
    tool = Tool.from_function(foo, name="foo", description="Docstring")

    assert tool.run(tool_input={"bar": "bar"}, run_manager=[handler]) == "foobar"
    assert tool.run("baz", run_manager=[handler]) == "foobaz"


def test_structured_tool_from_function_with_run_manager() -> None:
    """Test args and schema of structured tool when using callbacks."""

    def foo(
        bar: int, baz: str, callbacks: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Docstring
        Args:
            bar: int
            baz: str
        """
        assert callbacks is not None
        return str(bar) + baz

    handler = FakeCallbackHandler()
    structured_tool = StructuredTool.from_function(foo)

    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "string"},
    }

    assert structured_tool.args_schema.schema() == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "title": "fooSchemaSchema",
        "type": "object",
        "required": ["bar", "baz"],
    }

    assert (
        structured_tool.run(
            tool_input={"bar": "10", "baz": "baz"}, run_manger=[handler]
        )
        == "10baz"
    )


def test_named_tool_decorator() -> None:
    """Test functionality when arguments are provided as input to decorator."""

    @tool("search")
    def search_api(query: str) -> str:
        """Search the API for the query."""
        assert isinstance(query, str)
        return f"API result - {query}"

    assert isinstance(search_api, BaseTool)
    assert search_api.name == "search"
    assert not search_api.return_direct
    assert search_api.run({"query": "foo"}) == "API result - foo"


def test_named_tool_decorator_return_direct() -> None:
    """Test functionality when arguments and return direct are provided as input."""

    @tool("search", return_direct=True)
    def search_api(query: str, *args: Any) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, BaseTool)
    assert search_api.name == "search"
    assert search_api.return_direct
    assert search_api.run({"query": "foo"}) == "API result"


def test_unnamed_tool_decorator_return_direct() -> None:
    """Test functionality when only return direct is provided."""

    @tool(return_direct=True)
    def search_api(query: str) -> str:
        """Search the API for the query."""
        assert isinstance(query, str)
        return "API result"

    assert isinstance(search_api, BaseTool)
    assert search_api.name == "search_api"
    assert search_api.return_direct
    assert search_api.run({"query": "foo"}) == "API result"


def test_tool_with_kwargs() -> None:
    """Test functionality when only return direct is provided."""

    @tool(return_direct=True)
    def search_api(
        arg_0: str,
        arg_1: float = 4.3,
        ping: str = "hi",
    ) -> str:
        """Search the API for the query."""
        return f"arg_0={arg_0}, arg_1={arg_1}, ping={ping}"

    assert isinstance(search_api, BaseTool)
    result = search_api.run(
        tool_input={
            "arg_0": "foo",
            "arg_1": 3.2,
            "ping": "pong",
        }
    )
    assert result == "arg_0=foo, arg_1=3.2, ping=pong"

    result = search_api.run(
        tool_input={
            "arg_0": "foo",
        }
    )
    assert result == "arg_0=foo, arg_1=4.3, ping=hi"
    # For backwards compatibility, we still accept a single str arg
    result = search_api.run("foobar")
    assert result == "arg_0=foobar, arg_1=4.3, ping=hi"


def test_missing_docstring() -> None:
    """Test error is raised when docstring is missing."""
    # expect to throw a value error if there's no docstring
    with pytest.raises(AssertionError, match="Function must have a docstring"):

        @tool
        def search_api(query: str) -> str:
            return "API result"


def test_create_tool_positional_args() -> None:
    """Test that positional arguments are allowed."""
    test_tool = Tool("test_name", lambda x: x, "test_description")
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"
    assert test_tool.is_single_input


def test_create_tool_keyword_args() -> None:
    """Test that keyword arguments are allowed."""
    test_tool = Tool(name="test_name", func=lambda x: x, description="test_description")
    assert test_tool.is_single_input
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"


@pytest.mark.asyncio
async def test_create_async_tool() -> None:
    """Test that async tools are allowed."""

    async def _test_func(x: str) -> str:
        return x

    test_tool = Tool(
        name="test_name",
        func=lambda x: x,
        description="test_description",
        coroutine=_test_func,
    )
    assert test_tool.is_single_input
    assert test_tool("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"
    assert test_tool.coroutine is not None
    assert await test_tool.arun("foo") == "foo"


class _FakeExceptionTool(BaseTool):
    name = "exception"
    description = "an exception-throwing tool"
    exception: Exception = ToolException()

    def _run(self) -> str:
        raise self.exception

    async def _arun(self) -> str:
        raise self.exception


def test_exception_handling_bool() -> None:
    _tool = _FakeExceptionTool(handle_tool_error=True)
    expected = "Tool execution error"
    actual = _tool.run({})
    assert expected == actual


def test_exception_handling_str() -> None:
    expected = "foo bar"
    _tool = _FakeExceptionTool(handle_tool_error=expected)
    actual = _tool.run({})
    assert expected == actual


def test_exception_handling_callable() -> None:
    expected = "foo bar"
    handling = lambda _: expected  # noqa: E731
    _tool = _FakeExceptionTool(handle_tool_error=handling)
    actual = _tool.run({})
    assert expected == actual


def test_exception_handling_non_tool_exception() -> None:
    _tool = _FakeExceptionTool(exception=ValueError())
    with pytest.raises(ValueError):
        _tool.run({})


@pytest.mark.asyncio
async def test_async_exception_handling_bool() -> None:
    _tool = _FakeExceptionTool(handle_tool_error=True)
    expected = "Tool execution error"
    actual = await _tool.arun({})
    assert expected == actual


@pytest.mark.asyncio
async def test_async_exception_handling_str() -> None:
    expected = "foo bar"
    _tool = _FakeExceptionTool(handle_tool_error=expected)
    actual = await _tool.arun({})
    assert expected == actual


@pytest.mark.asyncio
async def test_async_exception_handling_callable() -> None:
    expected = "foo bar"
    handling = lambda _: expected  # noqa: E731
    _tool = _FakeExceptionTool(handle_tool_error=handling)
    actual = await _tool.arun({})
    assert expected == actual


@pytest.mark.asyncio
async def test_async_exception_handling_non_tool_exception() -> None:
    _tool = _FakeExceptionTool(exception=ValueError())
    with pytest.raises(ValueError):
        await _tool.arun({})


def test_structured_tool_from_function() -> None:
    """Test that structured tools can be created from functions."""

    def foo(bar: int, baz: str) -> str:
        """Docstring
        Args:
            bar: int
            baz: str
        """
        raise NotImplementedError()

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "string"},
    }

    assert structured_tool.args_schema.schema() == {
        "title": "fooSchemaSchema",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "required": ["bar", "baz"],
    }

    prefix = "foo(bar: int, baz: str) -> str - "
    assert foo.__doc__ is not None
    assert structured_tool.description == prefix + foo.__doc__.strip()
