"""Test the base tool implementation."""

import inspect
import json
import sys
import textwrap
import threading
from datetime import datetime
from enum import Enum
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from typing_extensions import TypedDict, override

from langchain_core import tools
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.messages.tool import ToolOutputMixin
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    ensure_config,
)
from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    Tool,
    ToolException,
    tool,
)
from langchain_core.tools.base import (
    TOOL_MESSAGE_BLOCK_TYPES,
    ArgsSchema,
    InjectedToolArg,
    InjectedToolCallId,
    SchemaAnnotationError,
    _is_message_content_block,
    _is_message_content_type,
    get_all_basemodel_annotations,
)
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    _create_subset_model,
    create_model_v2,
)
from tests.unit_tests.fake.callbacks import FakeCallbackHandler
from tests.unit_tests.pydantic_utils import _schema


def _get_tool_call_json_schema(tool: BaseTool) -> dict:
    tool_schema = tool.tool_call_schema
    if isinstance(tool_schema, dict):
        return tool_schema

    if issubclass(tool_schema, BaseModel):
        return tool_schema.model_json_schema()
    if issubclass(tool_schema, BaseModelV1):
        return tool_schema.schema()
    return {}


def test_unnamed_decorator() -> None:
    """Test functionality with unnamed decorator."""

    @tool
    def search_api(query: str) -> str:
        """Search the API for the query."""
        return "API result"

    assert isinstance(search_api, BaseTool)
    assert search_api.name == "search_api"
    assert not search_api.return_direct
    assert search_api.invoke("test") == "API result"


class _MockSchema(BaseModel):
    """Return the arguments directly."""

    arg1: int
    arg2: bool
    arg3: Optional[dict] = None


class _MockSchemaV1(BaseModelV1):
    """Return the arguments directly."""

    arg1: int
    arg2: bool
    arg3: Optional[dict] = None


class _MockStructuredTool(BaseTool):
    name: str = "structured_api"
    args_schema: type[BaseModel] = _MockSchema
    description: str = "A Structured Tool"

    @override
    def _run(self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        return f"{arg1} {arg2} {arg3}"

    async def _arun(self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        raise NotImplementedError


def test_structured_args() -> None:
    """Test functionality with structured arguments."""
    structured_api = _MockStructuredTool()
    assert isinstance(structured_api, BaseTool)
    assert structured_api.name == "structured_api"
    expected_result = "1 True {'foo': 'bar'}"
    args = {"arg1": 1, "arg2": True, "arg3": {"foo": "bar"}}
    assert structured_api.run(args) == expected_result


def test_misannotated_base_tool_raises_error() -> None:
    """Test that a BaseTool with the incorrect typehint raises an exception."""
    with pytest.raises(SchemaAnnotationError):

        class _MisAnnotatedTool(BaseTool):
            name: str = "structured_api"
            # This would silently be ignored without the custom metaclass
            args_schema: BaseModel = _MockSchema  # type: ignore[assignment]
            description: str = "A Structured Tool"

            @override
            def _run(
                self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None
            ) -> str:
                return f"{arg1} {arg2} {arg3}"

            async def _arun(
                self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None
            ) -> str:
                raise NotImplementedError


def test_forward_ref_annotated_base_tool_accepted() -> None:
    """Test that a using forward ref annotation syntax is accepted."""

    class _ForwardRefAnnotatedTool(BaseTool):
        name: str = "structured_api"
        args_schema: "type[BaseModel]" = _MockSchema
        description: str = "A Structured Tool"

        @override
        def _run(self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError


def test_subclass_annotated_base_tool_accepted() -> None:
    """Test BaseTool child w/ custom schema isn't overwritten."""

    class _ForwardRefAnnotatedTool(BaseTool):
        name: str = "structured_api"
        args_schema: type[_MockSchema] = _MockSchema
        description: str = "A Structured Tool"

        @override
        def _run(self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, *, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError

    assert issubclass(_ForwardRefAnnotatedTool, BaseTool)
    tool = _ForwardRefAnnotatedTool()
    assert tool.args_schema == _MockSchema


def test_decorator_with_specified_schema() -> None:
    """Test that manually specified schemata are passed through to the tool."""

    @tool(args_schema=_MockSchema)
    def tool_func(*, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func, BaseTool)
    assert tool_func.args_schema == _MockSchema

    @tool(args_schema=cast("ArgsSchema", _MockSchemaV1))
    def tool_func_v1(*, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func_v1, BaseTool)
    assert tool_func_v1.args_schema == cast("ArgsSchema", _MockSchemaV1)


def test_decorated_function_schema_equivalent() -> None:
    """Test that a BaseTool without a schema meets expectations."""

    @tool
    def structured_tool_input(
        *, arg1: int, arg2: bool, arg3: Optional[dict] = None
    ) -> str:
        """Return the arguments directly."""
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(structured_tool_input, BaseTool)
    assert structured_tool_input.args_schema is not None
    assert (
        _schema(structured_tool_input.args_schema)["properties"]
        == _schema(_MockSchema)["properties"]
        == structured_tool_input.args
    )


def test_args_kwargs_filtered() -> None:
    class _SingleArgToolWithKwargs(BaseTool):
        name: str = "single_arg_tool"
        description: str = "A  single arged tool with kwargs"

        @override
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
        name: str = "single_arg_tool"
        description: str = "A single arged tool with kwargs"

        @override
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
        "some_base_model": SomeBaseModel(foo="bar").model_dump(),
    }
    result = structured_tool.run(json.loads(json.dumps(args)))
    expected = {
        "some_enum": SomeEnum.A,
        "some_base_model": SomeBaseModel(foo="bar"),
    }
    assert result == expected


def test_structured_tool_types_parsed_pydantic_v1() -> None:
    """Test the non-primitive types are correctly passed to structured tools."""

    class SomeBaseModel(BaseModelV1):
        foo: str

    class AnotherBaseModel(BaseModelV1):
        bar: str

    @tool
    def structured_tool(some_base_model: SomeBaseModel) -> AnotherBaseModel:
        """Return the arguments directly."""
        return AnotherBaseModel(bar=some_base_model.foo)

    assert isinstance(structured_tool, StructuredTool)

    expected = AnotherBaseModel(bar="baz")
    for arg in [
        SomeBaseModel(foo="baz"),
        SomeBaseModel(foo="baz").dict(),
    ]:
        args = {"some_base_model": arg}
        result = structured_tool.run(args)
        assert result == expected


def test_structured_tool_types_parsed_pydantic_mixed() -> None:
    """Test handling of tool with mixed Pydantic version arguments."""

    class SomeBaseModel(BaseModelV1):
        foo: str

    class AnotherBaseModel(BaseModel):
        bar: str

    with pytest.raises(NotImplementedError):

        @tool
        def structured_tool(
            some_base_model: SomeBaseModel, another_base_model: AnotherBaseModel
        ) -> None:
            """Return the arguments directly."""


def test_base_tool_inheritance_base_schema() -> None:
    """Test schema is correctly inferred when inheriting from BaseTool."""

    class _MockSimpleTool(BaseTool):
        name: str = "simple_tool"
        description: str = "A Simple Tool"

        @override
        def _run(self, tool_input: str) -> str:
            return f"{tool_input}"

        @override
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
        """Docstring.

        Args:
            bar: the bar value
            baz: the baz value
        """
        raise NotImplementedError

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "string"},
    }

    assert _schema(structured_tool.args_schema) == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "description": inspect.getdoc(foo),
        "title": "foo",
        "type": "object",
        "required": ["bar", "baz"],
    }

    assert foo.__doc__ is not None
    assert structured_tool.description == textwrap.dedent(foo.__doc__.strip())


def test_structured_tool_from_function_docstring_complex_args() -> None:
    """Test that structured tools can be created from functions."""

    def foo(bar: int, baz: list[str]) -> str:
        """Docstring.

        Args:
            bar: int
            baz: list[str]
        """
        raise NotImplementedError

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {
            "title": "Baz",
            "type": "array",
            "items": {"type": "string"},
        },
    }

    assert _schema(structured_tool.args_schema) == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {
                "title": "Baz",
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "description": inspect.getdoc(foo),
        "title": "foo",
        "type": "object",
        "required": ["bar", "baz"],
    }

    assert foo.__doc__ is not None
    assert structured_tool.description == textwrap.dedent(foo.__doc__).strip()


def test_structured_tool_lambda_multi_args_schema() -> None:
    """Test args schema inference when the tool argument is a lambda function."""
    tool = StructuredTool.from_function(
        name="tool",
        description="A tool",
        func=lambda tool_input, other_arg: f"{tool_input}{other_arg}",
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

    def foo(bar: str, callbacks: Optional[CallbackManagerForToolRun] = None) -> str:  # noqa: D417
        """Docstring.

        Args:
            bar: str.
        """
        assert callbacks is not None
        return "foo" + bar

    handler = FakeCallbackHandler()
    tool = Tool.from_function(foo, name="foo", description="Docstring")

    assert tool.run(tool_input={"bar": "bar"}, run_manager=[handler]) == "foobar"
    assert tool.run("baz", run_manager=[handler]) == "foobaz"


def test_structured_tool_from_function_with_run_manager() -> None:
    """Test args and schema of structured tool when using callbacks."""

    def foo(  # noqa: D417
        bar: int, baz: str, callbacks: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Docstring.

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

    assert _schema(structured_tool.args_schema) == {
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "description": inspect.getdoc(foo),
        "title": "foo",
        "type": "object",
        "required": ["bar", "baz"],
    }

    assert (
        structured_tool.run(
            tool_input={"bar": "10", "baz": "baz"}, run_manger=[handler]
        )
        == "10baz"
    )


def test_structured_tool_from_parameterless_function() -> None:
    """Test parameterless function of structured tool."""

    def foo() -> str:
        """Docstring."""
        return "invoke foo"

    structured_tool = StructuredTool.from_function(foo)

    assert structured_tool.run({}) == "invoke foo"
    assert structured_tool.run("") == "invoke foo"


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
    def search_api(query: str) -> str:
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
    with pytest.raises(ValueError, match="Function must have a docstring"):

        @tool
        def search_api(query: str) -> str:
            return "API result"

    @tool
    class MyTool(BaseModel):
        foo: str

    assert not MyTool.description  # type: ignore[attr-defined]


def test_create_tool_positional_args() -> None:
    """Test that positional arguments are allowed."""
    test_tool = Tool("test_name", lambda x: x, "test_description")
    assert test_tool.invoke("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"
    assert test_tool.is_single_input


def test_create_tool_keyword_args() -> None:
    """Test that keyword arguments are allowed."""
    test_tool = Tool(name="test_name", func=lambda x: x, description="test_description")
    assert test_tool.is_single_input
    assert test_tool.invoke("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"


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
    assert test_tool.invoke("foo") == "foo"
    assert test_tool.name == "test_name"
    assert test_tool.description == "test_description"
    assert test_tool.coroutine is not None
    assert await test_tool.arun("foo") == "foo"


class _FakeExceptionTool(BaseTool):
    name: str = "exception"
    description: str = "an exception-throwing tool"
    exception: Exception = ToolException()

    def _run(self) -> str:
        raise self.exception

    async def _arun(self) -> str:
        raise self.exception


def test_exception_handling_bool() -> None:
    tool_ = _FakeExceptionTool(handle_tool_error=True)
    expected = "Tool execution error"
    actual = tool_.run({})
    assert expected == actual


def test_exception_handling_str() -> None:
    expected = "foo bar"
    tool_ = _FakeExceptionTool(handle_tool_error=expected)
    actual = tool_.run({})
    assert expected == actual


def test_exception_handling_callable() -> None:
    expected = "foo bar"

    def handling(e: ToolException) -> str:
        return expected

    tool_ = _FakeExceptionTool(handle_tool_error=handling)
    actual = tool_.run({})
    assert expected == actual


def test_exception_handling_non_tool_exception() -> None:
    tool_ = _FakeExceptionTool(exception=ValueError("some error"))
    with pytest.raises(ValueError, match="some error"):
        tool_.run({})


async def test_async_exception_handling_bool() -> None:
    tool_ = _FakeExceptionTool(handle_tool_error=True)
    expected = "Tool execution error"
    actual = await tool_.arun({})
    assert expected == actual


async def test_async_exception_handling_str() -> None:
    expected = "foo bar"
    tool_ = _FakeExceptionTool(handle_tool_error=expected)
    actual = await tool_.arun({})
    assert expected == actual


async def test_async_exception_handling_callable() -> None:
    expected = "foo bar"

    def handling(e: ToolException) -> str:
        return expected

    tool_ = _FakeExceptionTool(handle_tool_error=handling)
    actual = await tool_.arun({})
    assert expected == actual


async def test_async_exception_handling_non_tool_exception() -> None:
    tool_ = _FakeExceptionTool(exception=ValueError("some error"))
    with pytest.raises(ValueError, match="some error"):
        await tool_.arun({})


def test_structured_tool_from_function() -> None:
    """Test that structured tools can be created from functions."""

    def foo(bar: int, baz: str) -> str:
        """Docstring thing.

        Args:
            bar: the bar value
            baz: the baz value
        """
        raise NotImplementedError

    structured_tool = StructuredTool.from_function(foo)
    assert structured_tool.name == "foo"
    assert structured_tool.args == {
        "bar": {"title": "Bar", "type": "integer"},
        "baz": {"title": "Baz", "type": "string"},
    }

    assert _schema(structured_tool.args_schema) == {
        "title": "foo",
        "type": "object",
        "description": inspect.getdoc(foo),
        "properties": {
            "bar": {"title": "Bar", "type": "integer"},
            "baz": {"title": "Baz", "type": "string"},
        },
        "required": ["bar", "baz"],
    }

    assert foo.__doc__ is not None
    assert structured_tool.description == textwrap.dedent(foo.__doc__.strip())


def test_validation_error_handling_bool() -> None:
    """Test that validation errors are handled correctly."""
    expected = "Tool input validation error"
    tool_ = _MockStructuredTool(handle_validation_error=True)
    actual = tool_.run({})
    assert expected == actual


def test_validation_error_handling_str() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"
    tool_ = _MockStructuredTool(handle_validation_error=expected)
    actual = tool_.run({})
    assert expected == actual


def test_validation_error_handling_callable() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"

    def handling(e: Union[ValidationError, ValidationErrorV1]) -> str:
        return expected

    tool_ = _MockStructuredTool(handle_validation_error=handling)
    actual = tool_.run({})
    assert expected == actual


@pytest.mark.parametrize(
    "handler",
    [
        True,
        "foo bar",
        lambda _: "foo bar",
    ],
)
def test_validation_error_handling_non_validation_error(
    *,
    handler: Union[
        bool, str, Callable[[Union[ValidationError, ValidationErrorV1]], str]
    ],
) -> None:
    """Test that validation errors are handled correctly."""

    class _RaiseNonValidationErrorTool(BaseTool):
        name: str = "raise_non_validation_error_tool"
        description: str = "A tool that raises a non-validation error"

        def _parse_input(
            self,
            tool_input: Union[str, dict],
            tool_call_id: Optional[str],
        ) -> Union[str, dict[str, Any]]:
            raise NotImplementedError

        @override
        def _run(self) -> str:
            return "dummy"

        @override
        async def _arun(self) -> str:
            return "dummy"

    tool_ = _RaiseNonValidationErrorTool(handle_validation_error=handler)
    with pytest.raises(NotImplementedError):
        tool_.run({})


async def test_async_validation_error_handling_bool() -> None:
    """Test that validation errors are handled correctly."""
    expected = "Tool input validation error"
    tool_ = _MockStructuredTool(handle_validation_error=True)
    actual = await tool_.arun({})
    assert expected == actual


async def test_async_validation_error_handling_str() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"
    tool_ = _MockStructuredTool(handle_validation_error=expected)
    actual = await tool_.arun({})
    assert expected == actual


async def test_async_validation_error_handling_callable() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"

    def handling(e: Union[ValidationError, ValidationErrorV1]) -> str:
        return expected

    tool_ = _MockStructuredTool(handle_validation_error=handling)
    actual = await tool_.arun({})
    assert expected == actual


@pytest.mark.parametrize(
    "handler",
    [
        True,
        "foo bar",
        lambda _: "foo bar",
    ],
)
async def test_async_validation_error_handling_non_validation_error(
    *,
    handler: Union[
        bool, str, Callable[[Union[ValidationError, ValidationErrorV1]], str]
    ],
) -> None:
    """Test that validation errors are handled correctly."""

    class _RaiseNonValidationErrorTool(BaseTool):
        name: str = "raise_non_validation_error_tool"
        description: str = "A tool that raises a non-validation error"

        def _parse_input(
            self,
            tool_input: Union[str, dict],
            tool_call_id: Optional[str],
        ) -> Union[str, dict[str, Any]]:
            raise NotImplementedError

        @override
        def _run(self) -> str:
            return "dummy"

        @override
        async def _arun(self) -> str:
            return "dummy"

    tool_ = _RaiseNonValidationErrorTool(handle_validation_error=handler)
    with pytest.raises(NotImplementedError):
        await tool_.arun({})


def test_optional_subset_model_rewrite() -> None:
    class MyModel(BaseModel):
        a: Optional[str] = None
        b: str
        c: Optional[list[Optional[str]]] = None

    model2 = _create_subset_model("model2", MyModel, ["a", "b", "c"])

    assert set(_schema(model2)["required"]) == {"b"}


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        # Check not required
        ({"bar": "bar"}, {"bar": "bar", "baz": 3, "buzz": "buzz"}),
        # Check overwritten
        (
            {"bar": "bar", "baz": 4, "buzz": "not-buzz"},
            {"bar": "bar", "baz": 4, "buzz": "not-buzz"},
        ),
        # Check validation error when missing
        ({}, None),
        # Check validation error when wrong type
        ({"bar": "bar", "baz": "not-an-int"}, None),
        # Check OK when None explicitly passed
        ({"bar": "bar", "baz": None}, {"bar": "bar", "baz": None, "buzz": "buzz"}),
    ],
)
def test_tool_invoke_optional_args(inputs: dict, expected: Optional[dict]) -> None:
    @tool
    def foo(bar: str, baz: Optional[int] = 3, buzz: Optional[str] = "buzz") -> dict:
        """The foo."""
        return {
            "bar": bar,
            "baz": baz,
            "buzz": buzz,
        }

    if expected is not None:
        assert foo.invoke(inputs) == expected
    else:
        with pytest.raises(ValidationError):
            foo.invoke(inputs)


def test_tool_pass_context() -> None:
    @tool
    def foo(bar: str) -> str:
        """The foo."""
        config = ensure_config()
        assert config["configurable"]["foo"] == "not-bar"
        assert bar == "baz"
        return bar

    assert foo.invoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}}) == "baz"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="requires python3.11 or higher",
)
async def test_async_tool_pass_context() -> None:
    @tool
    async def foo(bar: str) -> str:
        """The foo."""
        config = ensure_config()
        assert config["configurable"]["foo"] == "not-bar"
        assert bar == "baz"
        return bar

    assert (
        await foo.ainvoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}}) == "baz"
    )


def assert_bar(bar: Any, bar_config: RunnableConfig) -> Any:
    assert bar_config["configurable"]["foo"] == "not-bar"
    assert bar == "baz"
    return bar


@tool
def foo(bar: Any, bar_config: RunnableConfig) -> Any:
    """The foo."""
    return assert_bar(bar, bar_config)


@tool
async def afoo(bar: Any, bar_config: RunnableConfig) -> Any:
    """The foo."""
    return assert_bar(bar, bar_config)


@tool(infer_schema=False)
def simple_foo(bar: Any, bar_config: RunnableConfig) -> Any:
    """The foo."""
    return assert_bar(bar, bar_config)


@tool(infer_schema=False)
async def asimple_foo(bar: Any, bar_config: RunnableConfig) -> Any:
    """The foo."""
    return assert_bar(bar, bar_config)


class FooBase(BaseTool):
    name: str = "Foo"
    description: str = "Foo"

    @override
    def _run(self, bar: Any, bar_config: RunnableConfig, **kwargs: Any) -> Any:
        return assert_bar(bar, bar_config)


class AFooBase(FooBase):
    @override
    async def _arun(self, bar: Any, bar_config: RunnableConfig, **kwargs: Any) -> Any:
        return assert_bar(bar, bar_config)


@pytest.mark.parametrize("tool", [foo, simple_foo, FooBase(), AFooBase()])
def test_tool_pass_config(tool: BaseTool) -> None:
    assert tool.invoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}}) == "baz"

    # Test we don't mutate tool calls
    tool_call = {
        "name": tool.name,
        "args": {"bar": "baz"},
        "id": "abc123",
        "type": "tool_call",
    }
    _ = tool.invoke(tool_call, {"configurable": {"foo": "not-bar"}})
    assert tool_call["args"] == {"bar": "baz"}


class FooBaseNonPickleable(FooBase):
    @override
    def _run(self, bar: Any, bar_config: RunnableConfig, **kwargs: Any) -> Any:
        return True


def test_tool_pass_config_non_pickleable() -> None:
    tool = FooBaseNonPickleable()

    args = {"bar": threading.Lock()}
    tool_call = {
        "name": tool.name,
        "args": args,
        "id": "abc123",
        "type": "tool_call",
    }
    _ = tool.invoke(tool_call, {"configurable": {"foo": "not-bar"}})
    assert tool_call["args"] == args


@pytest.mark.parametrize(
    "tool", [foo, afoo, simple_foo, asimple_foo, FooBase(), AFooBase()]
)
async def test_async_tool_pass_config(tool: BaseTool) -> None:
    assert (
        await tool.ainvoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}})
        == "baz"
    )


def test_tool_description() -> None:
    def foo(bar: str) -> str:
        """The foo."""
        return bar

    foo1 = tool(foo)
    assert foo1.description == "The foo."

    foo2 = StructuredTool.from_function(foo)
    assert foo2.description == "The foo."


def test_tool_arg_descriptions() -> None:
    def foo(bar: str, baz: int) -> str:
        """The foo.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    foo1 = tool(foo)
    args_schema = _schema(foo1.args_schema)
    assert args_schema == {
        "title": "foo",
        "type": "object",
        "description": inspect.getdoc(foo),
        "properties": {
            "bar": {"title": "Bar", "type": "string"},
            "baz": {"title": "Baz", "type": "integer"},
        },
        "required": ["bar", "baz"],
    }

    # Test parses docstring
    foo2 = tool(foo, parse_docstring=True)
    args_schema = _schema(foo2.args_schema)
    expected = {
        "title": "foo",
        "description": "The foo.",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "description": "The bar.", "type": "string"},
            "baz": {"title": "Baz", "description": "The baz.", "type": "integer"},
        },
        "required": ["bar", "baz"],
    }
    assert args_schema == expected

    # Test parsing with run_manager does not raise error
    def foo3(  # noqa: D417
        bar: str, baz: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """The foo.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    as_tool = tool(foo3, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)
    assert args_schema["description"] == expected["description"]
    assert args_schema["properties"] == expected["properties"]

    # Test parameterless tool does not raise error for missing Args section
    # in docstring.
    def foo4() -> str:
        """The foo."""
        return "bar"

    as_tool = tool(foo4, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)
    assert args_schema["description"] == expected["description"]

    def foo5(run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """The foo."""
        return "bar"

    as_tool = tool(foo5, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)
    assert args_schema["description"] == expected["description"]


def test_docstring_parsing() -> None:
    expected = {
        "title": "foo",
        "description": "The foo.",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "description": "The bar.", "type": "string"},
            "baz": {"title": "Baz", "description": "The baz.", "type": "integer"},
        },
        "required": ["bar", "baz"],
    }

    # Simple case
    def foo(bar: str, baz: int) -> str:
        """The foo.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    as_tool = tool(foo, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)
    assert args_schema["description"] == "The foo."
    assert args_schema["properties"] == expected["properties"]

    # Multi-line description
    def foo2(bar: str, baz: int) -> str:
        """The foo.

        Additional description here.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    as_tool = tool(foo2, parse_docstring=True)
    args_schema2 = _schema(as_tool.args_schema)
    assert args_schema2["description"] == "The foo. Additional description here."
    assert args_schema2["properties"] == expected["properties"]

    # Multi-line wth Returns block
    def foo3(bar: str, baz: int) -> str:
        """The foo.

        Additional description here.

        Args:
            bar: The bar.
            baz: The baz.

        Returns:
            str: description of returned value.
        """
        return bar

    as_tool = tool(foo3, parse_docstring=True)
    args_schema3 = _schema(as_tool.args_schema)
    args_schema3["title"] = "foo2"
    assert args_schema2 == args_schema3

    # Single argument
    def foo4(bar: str) -> str:
        """The foo.

        Args:
            bar: The bar.
        """
        return bar

    as_tool = tool(foo4, parse_docstring=True)
    args_schema4 = _schema(as_tool.args_schema)
    assert args_schema4["description"] == "The foo."
    assert args_schema4["properties"] == {
        "bar": {"description": "The bar.", "title": "Bar", "type": "string"}
    }


def test_tool_invalid_docstrings() -> None:
    """Test invalid docstrings."""

    def foo3(bar: str, baz: int) -> str:
        """The foo."""
        return bar

    def foo4(bar: str, baz: int) -> str:
        """The foo.
        Args:
            bar: The bar.
            baz: The baz.
        """  # noqa: D205,D411
        return bar

    for func in {foo3, foo4}:
        with pytest.raises(ValueError, match="Found invalid Google-Style docstring"):
            _ = tool(func, parse_docstring=True)

    def foo5(bar: str, baz: int) -> str:  # noqa: D417
        """The foo.

        Args:
            banana: The bar.
            monkey: The baz.
        """
        return bar

    with pytest.raises(
        ValueError, match="Arg banana in docstring not found in function signature"
    ):
        _ = tool(foo5, parse_docstring=True)


def test_tool_annotated_descriptions() -> None:
    def foo(
        bar: Annotated[str, "this is the bar"], baz: Annotated[int, "this is the baz"]
    ) -> str:
        """The foo.

        Returns:
            The bar only.
        """
        return bar

    foo1 = tool(foo)
    args_schema = _schema(foo1.args_schema)
    assert args_schema == {
        "title": "foo",
        "type": "object",
        "description": inspect.getdoc(foo),
        "properties": {
            "bar": {"title": "Bar", "type": "string", "description": "this is the bar"},
            "baz": {
                "title": "Baz",
                "type": "integer",
                "description": "this is the baz",
            },
        },
        "required": ["bar", "baz"],
    }


def test_tool_call_input_tool_message_output() -> None:
    tool_call = {
        "name": "structured_api",
        "args": {"arg1": 1, "arg2": True, "arg3": {"img": "base64string..."}},
        "id": "123",
        "type": "tool_call",
    }
    tool = _MockStructuredTool()
    expected = ToolMessage(
        "1 True {'img': 'base64string...'}", tool_call_id="123", name="structured_api"
    )
    actual = tool.invoke(tool_call)
    assert actual == expected

    tool_call.pop("type")
    with pytest.raises(ValidationError):
        tool.invoke(tool_call)


@pytest.mark.parametrize("block_type", [*TOOL_MESSAGE_BLOCK_TYPES, "bad"])
def test_tool_content_block_output(block_type: str) -> None:
    @tool
    def my_tool(query: str) -> list[dict[str, Any]]:
        """Test tool."""
        return [{"type": block_type, "foo": "bar"}]

    tool_call = {
        "type": "tool_call",
        "name": "my_tool",
        "args": {"query": "baz"},
        "id": "call_abc123",
    }

    result = my_tool.invoke(tool_call)
    assert isinstance(result, ToolMessage)

    if block_type in TOOL_MESSAGE_BLOCK_TYPES:
        assert result.content == [{"type": block_type, "foo": "bar"}]
    else:
        assert result.content == '[{"type": "bad", "foo": "bar"}]'


class _MockStructuredToolWithRawOutput(BaseTool):
    name: str = "structured_api"
    args_schema: type[BaseModel] = _MockSchema
    description: str = "A Structured Tool"
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    @override
    def _run(
        self,
        arg1: int,
        arg2: bool,
        arg3: Optional[dict] = None,
    ) -> tuple[str, dict]:
        return f"{arg1} {arg2}", {"arg1": arg1, "arg2": arg2, "arg3": arg3}


@tool("structured_api", response_format="content_and_artifact")
def _mock_structured_tool_with_artifact(
    *, arg1: int, arg2: bool, arg3: Optional[dict] = None
) -> tuple[str, dict]:
    """A Structured Tool."""
    return f"{arg1} {arg2}", {"arg1": arg1, "arg2": arg2, "arg3": arg3}


@pytest.mark.parametrize(
    "tool", [_MockStructuredToolWithRawOutput(), _mock_structured_tool_with_artifact]
)
def test_tool_call_input_tool_message_with_artifact(tool: BaseTool) -> None:
    tool_call: dict = {
        "name": "structured_api",
        "args": {"arg1": 1, "arg2": True, "arg3": {"img": "base64string..."}},
        "id": "123",
        "type": "tool_call",
    }
    expected = ToolMessage(
        "1 True", artifact=tool_call["args"], tool_call_id="123", name="structured_api"
    )
    actual = tool.invoke(tool_call)
    assert actual == expected

    tool_call.pop("type")
    with pytest.raises(ValidationError):
        tool.invoke(tool_call)

    actual_content = tool.invoke(tool_call["args"])
    assert actual_content == expected.content


def test_convert_from_runnable_dict() -> None:
    # Test with typed dict input
    class Args(TypedDict):
        a: int
        b: list[int]

    def f(x: Args) -> str:
        return str(x["a"] * max(x["b"]))

    runnable: Runnable = RunnableLambda(f)
    as_tool = runnable.as_tool()
    args_schema = as_tool.args_schema
    assert args_schema is not None
    assert _schema(args_schema) == {
        "title": "f",
        "type": "object",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "array", "items": {"type": "integer"}},
        },
        "required": ["a", "b"],
    }
    assert as_tool.description
    result = as_tool.invoke({"a": 3, "b": [1, 2]})
    assert result == "6"

    as_tool = runnable.as_tool(name="my tool", description="test description")
    assert as_tool.name == "my tool"
    assert as_tool.description == "test description"

    # Dict without typed input-- must supply schema
    def g(x: dict[str, Any]) -> str:
        return str(x["a"] * max(x["b"]))

    # Specify via args_schema:
    class GSchema(BaseModel):
        """Apply a function to an integer and list of integers."""

        a: int = Field(..., description="Integer")
        b: list[int] = Field(..., description="List of ints")

    runnable = RunnableLambda(g)
    as_tool = runnable.as_tool(GSchema)
    as_tool.invoke({"a": 3, "b": [1, 2]})

    # Specify via arg_types:
    runnable = RunnableLambda(g)
    as_tool = runnable.as_tool(arg_types={"a": int, "b": list[int]})
    result = as_tool.invoke({"a": 3, "b": [1, 2]})
    assert result == "6"

    # Test with config
    def h(x: dict[str, Any]) -> str:
        config = ensure_config()
        assert config["configurable"]["foo"] == "not-bar"
        return str(x["a"] * max(x["b"]))

    runnable = RunnableLambda(h)
    as_tool = runnable.as_tool(arg_types={"a": int, "b": list[int]})
    result = as_tool.invoke(
        {"a": 3, "b": [1, 2]}, config={"configurable": {"foo": "not-bar"}}
    )
    assert result == "6"


def test_convert_from_runnable_other() -> None:
    # String input
    def f(x: str) -> str:
        return x + "a"

    def g(x: str) -> str:
        return x + "z"

    runnable: Runnable = RunnableLambda(f) | g
    as_tool = runnable.as_tool()
    args_schema = as_tool.args_schema
    assert args_schema is None
    assert as_tool.description

    result = as_tool.invoke("b")
    assert result == "baz"

    # Test with config
    def h(x: str) -> str:
        config = ensure_config()
        assert config["configurable"]["foo"] == "not-bar"
        return x + "a"

    runnable = RunnableLambda(h)
    as_tool = runnable.as_tool()
    result = as_tool.invoke("b", config={"configurable": {"foo": "not-bar"}})
    assert result == "ba"


@tool("foo", parse_docstring=True)
def injected_tool(x: int, y: Annotated[str, InjectedToolArg]) -> str:
    """Foo.

    Args:
        x: abc
        y: 123
    """
    return y


class InjectedTool(BaseTool):
    name: str = "foo"
    description: str = "foo."

    @override
    def _run(self, x: int, y: Annotated[str, InjectedToolArg]) -> Any:
        """Foo.

        Args:
            x: abc
            y: 123
        """
        return y


class fooSchema(BaseModel):  # noqa: N801
    """foo."""

    x: int = Field(..., description="abc")
    y: Annotated[str, "foobar comment", InjectedToolArg()] = Field(
        ..., description="123"
    )


class InjectedToolWithSchema(BaseTool):
    name: str = "foo"
    description: str = "foo."
    args_schema: type[BaseModel] = fooSchema

    @override
    def _run(self, x: int, y: str) -> Any:
        return y


@tool("foo", args_schema=fooSchema)
def injected_tool_with_schema(x: int, y: str) -> str:
    return y


@pytest.mark.parametrize("tool_", [InjectedTool()])
def test_tool_injected_arg_without_schema(tool_: BaseTool) -> None:
    assert _schema(tool_.get_input_schema()) == {
        "title": "foo",
        "description": "Foo.\n\nArgs:\n    x: abc\n    y: 123",
        "type": "object",
        "properties": {
            "x": {"title": "X", "type": "integer"},
            "y": {"title": "Y", "type": "string"},
        },
        "required": ["x", "y"],
    }
    assert _schema(tool_.tool_call_schema) == {
        "title": "foo",
        "description": "foo.",
        "type": "object",
        "properties": {"x": {"title": "X", "type": "integer"}},
        "required": ["x"],
    }
    assert tool_.invoke({"x": 5, "y": "bar"}) == "bar"
    assert tool_.invoke(
        {
            "name": "foo",
            "args": {"x": 5, "y": "bar"},
            "id": "123",
            "type": "tool_call",
        }
    ) == ToolMessage("bar", tool_call_id="123", name="foo")
    expected_error = (
        ValidationError if not isinstance(tool_, InjectedTool) else TypeError
    )
    with pytest.raises(expected_error):
        tool_.invoke({"x": 5})

    assert convert_to_openai_function(tool_) == {
        "name": "foo",
        "description": "foo.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }


@pytest.mark.parametrize(
    "tool_",
    [injected_tool_with_schema, InjectedToolWithSchema()],
)
def test_tool_injected_arg_with_schema(tool_: BaseTool) -> None:
    assert _schema(tool_.get_input_schema()) == {
        "title": "fooSchema",
        "description": "foo.",
        "type": "object",
        "properties": {
            "x": {"description": "abc", "title": "X", "type": "integer"},
            "y": {"description": "123", "title": "Y", "type": "string"},
        },
        "required": ["x", "y"],
    }
    assert _schema(tool_.tool_call_schema) == {
        "title": "foo",
        "description": "foo.",
        "type": "object",
        "properties": {"x": {"description": "abc", "title": "X", "type": "integer"}},
        "required": ["x"],
    }
    assert tool_.invoke({"x": 5, "y": "bar"}) == "bar"
    assert tool_.invoke(
        {
            "name": "foo",
            "args": {"x": 5, "y": "bar"},
            "id": "123",
            "type": "tool_call",
        }
    ) == ToolMessage("bar", tool_call_id="123", name="foo")
    expected_error = (
        ValidationError if not isinstance(tool_, InjectedTool) else TypeError
    )
    with pytest.raises(expected_error):
        tool_.invoke({"x": 5})

    assert convert_to_openai_function(tool_) == {
        "name": "foo",
        "description": "foo.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "abc"}},
            "required": ["x"],
        },
    }


def test_tool_injected_arg() -> None:
    tool_ = injected_tool
    assert _schema(tool_.get_input_schema()) == {
        "title": "foo",
        "description": "Foo.",
        "type": "object",
        "properties": {
            "x": {"description": "abc", "title": "X", "type": "integer"},
            "y": {"description": "123", "title": "Y", "type": "string"},
        },
        "required": ["x", "y"],
    }
    assert _schema(tool_.tool_call_schema) == {
        "title": "foo",
        "description": "Foo.",
        "type": "object",
        "properties": {"x": {"description": "abc", "title": "X", "type": "integer"}},
        "required": ["x"],
    }
    assert tool_.invoke({"x": 5, "y": "bar"}) == "bar"
    assert tool_.invoke(
        {
            "name": "foo",
            "args": {"x": 5, "y": "bar"},
            "id": "123",
            "type": "tool_call",
        }
    ) == ToolMessage("bar", tool_call_id="123", name="foo")
    expected_error = (
        ValidationError if not isinstance(tool_, InjectedTool) else TypeError
    )
    with pytest.raises(expected_error):
        tool_.invoke({"x": 5})

    assert convert_to_openai_function(tool_) == {
        "name": "foo",
        "description": "Foo.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "abc"}},
            "required": ["x"],
        },
    }


def test_tool_inherited_injected_arg() -> None:
    class BarSchema(BaseModel):
        """bar."""

        y: Annotated[str, "foobar comment", InjectedToolArg()] = Field(
            ..., description="123"
        )

    class FooSchema(BarSchema):
        """foo."""

        x: int = Field(..., description="abc")

    class InheritedInjectedArgTool(BaseTool):
        name: str = "foo"
        description: str = "foo."
        args_schema: type[BaseModel] = FooSchema

        @override
        def _run(self, x: int, y: str) -> Any:
            return y

    tool_ = InheritedInjectedArgTool()
    assert tool_.get_input_schema().model_json_schema() == {
        "title": "FooSchema",  # Matches the title from the provided schema
        "description": "foo.",
        "type": "object",
        "properties": {
            "x": {"description": "abc", "title": "X", "type": "integer"},
            "y": {"description": "123", "title": "Y", "type": "string"},
        },
        "required": ["y", "x"],
    }
    # Should not include `y` since it's annotated as an injected tool arg
    assert _get_tool_call_json_schema(tool_) == {
        "title": "foo",
        "description": "foo.",
        "type": "object",
        "properties": {"x": {"description": "abc", "title": "X", "type": "integer"}},
        "required": ["x"],
    }
    assert tool_.invoke({"x": 5, "y": "bar"}) == "bar"
    assert tool_.invoke(
        {
            "name": "foo",
            "args": {"x": 5, "y": "bar"},
            "id": "123",
            "type": "tool_call",
        }
    ) == ToolMessage("bar", tool_call_id="123", name="foo")
    expected_error = (
        ValidationError if not isinstance(tool_, InjectedTool) else TypeError
    )
    with pytest.raises(expected_error):
        tool_.invoke({"x": 5})

    assert convert_to_openai_function(tool_) == {
        "name": "foo",
        "description": "foo.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer", "description": "abc"}},
            "required": ["x"],
        },
    }


def _get_parametrized_tools() -> list:
    def my_tool(x: int, y: str, some_tool: Annotated[Any, InjectedToolArg]) -> str:
        """my_tool."""
        return some_tool

    async def my_async_tool(
        x: int, y: str, *, some_tool: Annotated[Any, InjectedToolArg]
    ) -> str:
        """my_tool."""
        return some_tool

    return [my_tool, my_async_tool]


@pytest.mark.parametrize("tool_", _get_parametrized_tools())
def test_fn_injected_arg_with_schema(tool_: Callable) -> None:
    assert convert_to_openai_function(tool_) == {
        "name": tool_.__name__,
        "description": "my_tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"},
            },
            "required": ["x", "y"],
        },
    }


def generate_models() -> list[Any]:
    """Generate a list of base models depending on the pydantic version."""

    class FooProper(BaseModel):
        a: int
        b: str

    return [FooProper]


def generate_backwards_compatible_v1() -> list[Any]:
    """Generate a model with pydantic 2 from the v1 namespace."""

    class FooV1Namespace(BaseModelV1):
        a: int
        b: str

    return [FooV1Namespace]


# This generates a list of models that can be used for testing that our APIs
# behave well with either pydantic 1 proper,
# pydantic v1 from pydantic 2,
# or pydantic 2 proper.
TEST_MODELS = generate_models() + generate_backwards_compatible_v1()


@pytest.mark.parametrize("pydantic_model", TEST_MODELS)
def test_args_schema_as_pydantic(pydantic_model: Any) -> None:
    class SomeTool(BaseTool):
        args_schema: type[pydantic_model] = pydantic_model

        @override
        def _run(self, *args: Any, **kwargs: Any) -> str:
            return "foo"

    tool = SomeTool(
        name="some_tool", description="some description", args_schema=pydantic_model
    )

    assert tool.args == {
        "a": {"title": "A", "type": "integer"},
        "b": {"title": "B", "type": "string"},
    }

    input_schema = tool.get_input_schema()
    if issubclass(input_schema, BaseModel):
        input_json_schema = input_schema.model_json_schema()
    elif issubclass(input_schema, BaseModelV1):
        input_json_schema = input_schema.schema()
    else:
        msg = "Unknown input schema type"
        raise TypeError(msg)

    assert input_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": pydantic_model.__name__,
        "type": "object",
    }

    tool_json_schema = _get_tool_call_json_schema(tool)
    assert tool_json_schema == {
        "description": "some description",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": "some_tool",
        "type": "object",
    }


def test_args_schema_explicitly_typed() -> None:
    """This should test that one can type the args schema as a pydantic model.

    Please note that this will test using pydantic 2 even though BaseTool
    is a pydantic 1 model!

    """

    class Foo(BaseModel):
        a: int
        b: str

    class SomeTool(BaseTool):
        # type ignoring here since we're allowing overriding a type
        # signature of pydantic.v1.BaseModel with pydantic.BaseModel
        # for pydantic 2!
        args_schema: type[BaseModel] = Foo

        @override
        def _run(self, *args: Any, **kwargs: Any) -> str:
            return "foo"

    tool = SomeTool(name="some_tool", description="some description")

    assert tool.get_input_schema().model_json_schema() == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": "Foo",
        "type": "object",
    }

    assert _get_tool_call_json_schema(tool) == {
        "description": "some description",
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": "some_tool",
        "type": "object",
    }


@pytest.mark.parametrize("pydantic_model", TEST_MODELS)
def test_structured_tool_with_different_pydantic_versions(pydantic_model: Any) -> None:
    """This should test that one can type the args schema as a pydantic model."""

    def foo(a: int, b: str) -> str:
        """Hahaha."""
        return "foo"

    foo_tool = StructuredTool.from_function(
        func=foo,
        args_schema=pydantic_model,
    )

    assert foo_tool.invoke({"a": 5, "b": "hello"}) == "foo"

    args_schema = cast("type[BaseModel]", foo_tool.args_schema)
    if issubclass(args_schema, BaseModel):
        args_json_schema = args_schema.model_json_schema()
    elif issubclass(args_schema, BaseModelV1):
        args_json_schema = args_schema.schema()
    else:
        msg = "Unknown input schema type"
        raise TypeError(msg)
    assert args_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": pydantic_model.__name__,
        "type": "object",
    }

    input_schema = foo_tool.get_input_schema()
    if issubclass(input_schema, BaseModel):
        input_json_schema = input_schema.model_json_schema()
    elif issubclass(input_schema, BaseModelV1):
        input_json_schema = input_schema.schema()
    else:
        msg = "Unknown input schema type"
        raise TypeError(msg)
    assert input_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": pydantic_model.__name__,
        "type": "object",
    }


valid_tool_result_blocks = [
    "foo",
    {"type": "text", "text": "foo"},
    {"type": "text", "blah": "foo"},  # note, only 'type' key is currently checked
    {"type": "image_url", "image_url": {}},  # openai format
    {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "123",
        },
    },  # anthropic format
    {"type": "json", "json": {}},  # bedrock format
]
invalid_tool_result_blocks = [
    {"text": "foo"},  # missing type
    {"results": "foo"},  # not content blocks
]


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        *([[block, True] for block in valid_tool_result_blocks]),
        *([[block, False] for block in invalid_tool_result_blocks]),
    ],
)
def test__is_message_content_block(obj: Any, *, expected: bool) -> None:
    assert _is_message_content_block(obj) is expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ("foo", True),
        (valid_tool_result_blocks, True),
        (invalid_tool_result_blocks, False),
    ],
)
def test__is_message_content_type(obj: Any, *, expected: bool) -> None:
    assert _is_message_content_type(obj) is expected


@pytest.mark.parametrize("use_v1_namespace", [True, False])
def test__get_all_basemodel_annotations_v2(*, use_v1_namespace: bool) -> None:
    A = TypeVar("A")

    if use_v1_namespace:

        class ModelA(BaseModelV1, Generic[A], extra="allow"):
            a: A

    else:

        class ModelA(BaseModel, Generic[A]):  # type: ignore[no-redef]
            a: A
            model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    class ModelB(ModelA[str]):
        b: Annotated[ModelA[dict[str, Any]], "foo"]

    class Mixin:
        def foo(self) -> str:
            return "foo"

    class ModelC(Mixin, ModelB):
        c: dict

    expected = {"a": str, "b": Annotated[ModelA[dict[str, Any]], "foo"], "c": dict}
    actual = get_all_basemodel_annotations(ModelC)
    assert actual == expected

    expected = {"a": str, "b": Annotated[ModelA[dict[str, Any]], "foo"]}
    actual = get_all_basemodel_annotations(ModelB)
    assert actual == expected

    expected = {"a": Any}
    actual = get_all_basemodel_annotations(ModelA)
    assert actual == expected

    expected = {"a": int}
    actual = get_all_basemodel_annotations(ModelA[int])
    assert actual == expected

    D = TypeVar("D", bound=Union[str, int])

    class ModelD(ModelC, Generic[D]):
        d: Optional[D]

    expected = {
        "a": str,
        "b": Annotated[ModelA[dict[str, Any]], "foo"],
        "c": dict,
        "d": Union[str, int, None],
    }
    actual = get_all_basemodel_annotations(ModelD)
    assert actual == expected

    expected = {
        "a": str,
        "b": Annotated[ModelA[dict[str, Any]], "foo"],
        "c": dict,
        "d": Union[int, None],
    }
    actual = get_all_basemodel_annotations(ModelD[int])
    assert actual == expected


def test_get_all_basemodel_annotations_aliases() -> None:
    class CalculatorInput(BaseModel):
        a: int = Field(description="first number", alias="A")
        b: int = Field(description="second number")

    actual = get_all_basemodel_annotations(CalculatorInput)
    assert actual == {"a": int, "b": int}


def test_tool_annotations_preserved() -> None:
    """Test that annotations are preserved when creating a tool."""

    @tool
    def my_tool(val: int, other_val: Annotated[dict, "my annotation"]) -> str:
        """Tool docstring."""
        return "foo"

    schema = my_tool.get_input_schema()

    func = my_tool.func  # type: ignore[attr-defined]

    expected_type_hints = {
        name: hint
        for name, hint in func.__annotations__.items()
        if name in inspect.signature(func).parameters
    }
    assert schema.__annotations__ == expected_type_hints


def test_create_retriever_tool() -> None:
    class MyRetriever(BaseRetriever):
        @override
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [Document(page_content=f"foo {query}"), Document(page_content="bar")]

    retriever = MyRetriever()
    retriever_tool = tools.create_retriever_tool(
        retriever, "retriever_tool_content", "Retriever Tool Content"
    )
    assert isinstance(retriever_tool, BaseTool)
    assert retriever_tool.name == "retriever_tool_content"
    assert retriever_tool.description == "Retriever Tool Content"
    assert retriever_tool.invoke("bar") == "foo bar\n\nbar"
    assert retriever_tool.invoke(
        ToolCall(
            name="retriever_tool_content",
            args={"query": "bar"},
            id="123",
            type="tool_call",
        )
    ) == ToolMessage(
        "foo bar\n\nbar", tool_call_id="123", name="retriever_tool_content"
    )

    retriever_tool_artifact = tools.create_retriever_tool(
        retriever,
        "retriever_tool_artifact",
        "Retriever Tool Artifact",
        response_format="content_and_artifact",
    )
    assert isinstance(retriever_tool_artifact, BaseTool)
    assert retriever_tool_artifact.name == "retriever_tool_artifact"
    assert retriever_tool_artifact.description == "Retriever Tool Artifact"
    assert retriever_tool_artifact.invoke("bar") == "foo bar\n\nbar"
    assert retriever_tool_artifact.invoke(
        ToolCall(
            name="retriever_tool_artifact",
            args={"query": "bar"},
            id="123",
            type="tool_call",
        )
    ) == ToolMessage(
        "foo bar\n\nbar",
        artifact=[Document(page_content="foo bar"), Document(page_content="bar")],
        tool_call_id="123",
        name="retriever_tool_artifact",
    )


def test_tool_args_schema_pydantic_v2_with_metadata() -> None:
    class Foo(BaseModel):
        x: list[int] = Field(
            description="List of integers", min_length=10, max_length=15
        )

    @tool(args_schema=Foo)
    def foo(x) -> list[int]:  # type: ignore[no-untyped-def] # noqa: ANN001
        """Foo."""
        return x

    assert _get_tool_call_json_schema(foo) == {
        "description": "Foo.",
        "properties": {
            "x": {
                "description": "List of integers",
                "items": {"type": "integer"},
                "maxItems": 15,
                "minItems": 10,
                "title": "X",
                "type": "array",
            }
        },
        "required": ["x"],
        "title": "foo",
        "type": "object",
    }

    assert foo.invoke({"x": [0] * 10})
    with pytest.raises(ValidationError):
        foo.invoke({"x": [0] * 9})


def test_imports() -> None:
    expected_all = [
        "FILTERED_ARGS",
        "SchemaAnnotationError",
        "create_schema_from_function",
        "ToolException",
        "BaseTool",
        "Tool",
        "StructuredTool",
        "tool",
        "RetrieverInput",
        "create_retriever_tool",
        "ToolsRenderer",
        "render_text_description",
        "render_text_description_and_args",
        "BaseToolkit",
        "convert_runnable_to_tool",
        "InjectedToolArg",
    ]
    for module_name in expected_all:
        assert hasattr(tools, module_name)
        assert getattr(tools, module_name) is not None


def test_structured_tool_direct_init() -> None:
    def foo(bar: str) -> str:
        return bar

    async def async_foo(bar: str) -> str:
        return bar

    class FooSchema(BaseModel):
        bar: str = Field(..., description="The bar")

    tool = StructuredTool(name="foo", args_schema=FooSchema, coroutine=async_foo)

    with pytest.raises(NotImplementedError):
        assert tool.invoke("hello") == "hello"


def test_injected_arg_with_complex_type() -> None:
    """Test that an injected tool arg can be a complex type."""

    class Foo:
        def __init__(self) -> None:
            self.value = "bar"

    @tool
    def injected_tool(x: int, foo: Annotated[Foo, InjectedToolArg]) -> str:
        """Tool that has an injected tool arg."""
        return foo.value

    assert injected_tool.invoke({"x": 5, "foo": Foo()}) == "bar"


def test_tool_injected_tool_call_id() -> None:
    @tool
    def foo(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> ToolMessage:
        """Foo."""
        return ToolMessage(x, tool_call_id=tool_call_id)  # type: ignore[call-overload]

    assert foo.invoke(
        {
            "type": "tool_call",
            "args": {"x": 0},
            "name": "foo",
            "id": "bar",
        }
    ) == ToolMessage(0, tool_call_id="bar")  # type: ignore[call-overload]

    with pytest.raises(
        ValueError,
        match="When tool includes an InjectedToolCallId argument, "
        "tool must always be invoked with a full model ToolCall",
    ):
        assert foo.invoke({"x": 0})

    @tool
    def foo2(x: int, tool_call_id: Annotated[str, InjectedToolCallId()]) -> ToolMessage:
        """Foo."""
        return ToolMessage(x, tool_call_id=tool_call_id)  # type: ignore[call-overload]

    assert foo2.invoke(
        {
            "type": "tool_call",
            "args": {"x": 0},
            "name": "foo",
            "id": "bar",
        }
    ) == ToolMessage(0, tool_call_id="bar")  # type: ignore[call-overload]


def test_tool_injected_tool_call_id_override_llm_generated() -> None:
    """Test that InjectedToolCallId overrides LLM-generated values."""

    @tool
    def foo(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> ToolMessage:
        """Foo."""
        return ToolMessage(str(x), tool_call_id=tool_call_id)

    # Test that when LLM generates the tool_call_id, it gets overridden
    result = foo.invoke(
        {
            "type": "tool_call",
            "args": {"x": 0, "tool_call_id": "fake_llm_id"},  # LLM generated this
            "name": "foo",
            "id": "real_tool_call_id",  # This should be used instead
        }
    )

    # The tool should receive the real tool call ID, not the LLM-generated one
    assert result == ToolMessage("0", tool_call_id="real_tool_call_id")


def test_tool_uninjected_tool_call_id() -> None:
    @tool
    def foo(x: int, tool_call_id: str) -> ToolMessage:
        """Foo."""
        return ToolMessage(str(x), tool_call_id=tool_call_id)

    with pytest.raises(ValueError, match="1 validation error for foo"):
        foo.invoke({"type": "tool_call", "args": {"x": 0}, "name": "foo", "id": "bar"})

    assert foo.invoke(
        {
            "type": "tool_call",
            "args": {"x": 0, "tool_call_id": "zap"},
            "name": "foo",
            "id": "bar",
        }
    ) == ToolMessage(0, tool_call_id="zap")  # type: ignore[call-overload]


def test_tool_return_output_mixin() -> None:
    class Bar(ToolOutputMixin):
        def __init__(self, x: int) -> None:
            self.x = x

        def __eq__(self, other: object) -> bool:
            return isinstance(other, self.__class__) and self.x == other.x

        def __hash__(self) -> int:
            return hash(self.x)

    @tool
    def foo(x: int) -> Bar:
        """Foo."""
        return Bar(x=x)

    assert foo.invoke(
        {
            "type": "tool_call",
            "args": {"x": 0},
            "name": "foo",
            "id": "bar",
        }
    ) == Bar(x=0)


def test_tool_mutate_input() -> None:
    class MyTool(BaseTool):
        name: str = "MyTool"
        description: str = "a tool"

        @override
        def _run(
            self,
            x: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            return "hi"

    my_input = {"x": "hi"}
    MyTool().invoke(my_input)
    assert my_input == {"x": "hi"}


def test_structured_tool_args_schema_dict() -> None:
    args_schema = {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        },
        "required": ["a", "b"],
        "title": "add",
        "type": "object",
        "description": "add two numbers",
    }
    tool = StructuredTool(
        name="add",
        args_schema=args_schema,
        func=lambda a, b: a + b,
    )
    assert tool.invoke({"a": 1, "b": 2}) == 3
    assert tool.args_schema == args_schema
    # test that the tool call schema is the same as the args schema
    assert _get_tool_call_json_schema(tool) == args_schema
    # test that the input schema is the same as the parent (Runnable) input schema
    assert (
        tool.get_input_schema().model_json_schema()
        == create_model_v2(
            tool.get_name("Input"),
            root=tool.InputType,
            module_name=tool.__class__.__module__,
        ).model_json_schema()
    )
    # test that args are extracted correctly
    assert tool.args == {
        "a": {"title": "A", "type": "integer"},
        "b": {"title": "B", "type": "integer"},
    }


def test_simple_tool_args_schema_dict() -> None:
    args_schema = {
        "properties": {
            "a": {"title": "A", "type": "integer"},
        },
        "required": ["a"],
        "title": "square",
        "type": "object",
        "description": "square a number",
    }
    tool = Tool(
        name="square",
        description="square a number",
        args_schema=args_schema,
        func=lambda a: a * a,
    )
    assert tool.invoke({"a": 2}) == 4
    assert tool.args_schema == args_schema
    # test that the tool call schema is the same as the args schema
    assert _get_tool_call_json_schema(tool) == args_schema
    # test that the input schema is the same as the parent (Runnable) input schema
    assert (
        tool.get_input_schema().model_json_schema()
        == create_model_v2(
            tool.get_name("Input"),
            root=tool.InputType,
            module_name=tool.__class__.__module__,
        ).model_json_schema()
    )
    # test that args are extracted correctly
    assert tool.args == {
        "a": {"title": "A", "type": "integer"},
    }


def test_empty_string_tool_call_id() -> None:
    @tool
    def foo(x: int) -> str:
        """Foo."""
        return "hi"

    assert foo.invoke({"type": "tool_call", "args": {"x": 0}, "id": ""}) == ToolMessage(
        content="hi", name="foo", tool_call_id=""
    )


def test_tool_decorator_description() -> None:
    # test basic tool
    @tool
    def foo(x: int) -> str:
        """Foo."""
        return "hi"

    assert foo.description == "Foo."
    assert (
        cast("BaseModel", foo.tool_call_schema).model_json_schema()["description"]
        == "Foo."
    )

    # test basic tool with description
    @tool(description="description")
    def foo_description(x: int) -> str:
        """Foo."""
        return "hi"

    assert foo_description.description == "description"
    assert (
        cast("BaseModel", foo_description.tool_call_schema).model_json_schema()[
            "description"
        ]
        == "description"
    )

    # test tool with args schema
    class ArgsSchema(BaseModel):
        """Bar."""

        x: int

    @tool(args_schema=ArgsSchema)
    def foo_args_schema(x: int) -> str:
        return "hi"

    assert foo_args_schema.description == "Bar."
    assert (
        cast("BaseModel", foo_args_schema.tool_call_schema).model_json_schema()[
            "description"
        ]
        == "Bar."
    )

    @tool(description="description", args_schema=ArgsSchema)
    def foo_args_schema_description(x: int) -> str:
        return "hi"

    assert foo_args_schema_description.description == "description"
    assert (
        cast(
            "BaseModel", foo_args_schema_description.tool_call_schema
        ).model_json_schema()["description"]
        == "description"
    )

    args_json_schema = {
        "description": "JSON Schema.",
        "properties": {
            "x": {"description": "my field", "title": "X", "type": "string"}
        },
        "required": ["x"],
        "title": "my_tool",
        "type": "object",
    }

    @tool(args_schema=args_json_schema)
    def foo_args_jsons_schema(x: int) -> str:
        return "hi"

    @tool(description="description", args_schema=args_json_schema)
    def foo_args_jsons_schema_with_description(x: int) -> str:
        return "hi"

    assert foo_args_jsons_schema.description == "JSON Schema."
    assert (
        cast("dict", foo_args_jsons_schema.tool_call_schema)["description"]
        == "JSON Schema."
    )

    assert foo_args_jsons_schema_with_description.description == "description"
    assert (
        cast("dict", foo_args_jsons_schema_with_description.tool_call_schema)[
            "description"
        ]
        == "description"
    )


def test_title_property_preserved() -> None:
    """Test that the title property is preserved when generating schema.

    https://github.com/langchain-ai/langchain/issues/30456
    """
    schema_to_be_extracted = {
        "type": "object",
        "required": [],
        "properties": {
            "title": {"type": "string", "description": "item title"},
            "due_date": {"type": "string", "description": "item due date"},
        },
        "description": "foo",
    }

    @tool(args_schema=schema_to_be_extracted)
    def extract_data(extracted_data: dict[str, Any]) -> dict[str, Any]:
        """Some documentation."""
        return extracted_data

    assert convert_to_openai_tool(extract_data) == {
        "function": {
            "description": "Some documentation.",
            "name": "extract_data",
            "parameters": {
                "properties": {
                    "due_date": {"description": "item due date", "type": "string"},
                    "title": {"description": "item title", "type": "string"},
                },
                "required": [],
                "type": "object",
            },
        },
        "type": "function",
    }


def test_nested_pydantic_fields() -> None:
    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address = Field(description="Home address")

    result = convert_to_openai_tool(Person)
    assert len(result["function"]["parameters"]["properties"]) == 2


async def test_tool_ainvoke_does_not_mutate_inputs() -> None:
    """Verify that the inputs are not mutated when invoking a tool asynchronously."""

    def sync_no_op(foo: int) -> str:
        return "good"

    async def async_no_op(foo: int) -> str:
        return "good"

    tool = StructuredTool(
        name="sample_tool",
        description="",
        args_schema={
            "type": "object",
            "required": ["foo"],
            "properties": {
                "seconds": {"type": "number", "description": "How big is foo"}
            },
        },
        coroutine=async_no_op,
        func=sync_no_op,
    )

    tool_call: ToolCall = {
        "name": "sample_tool",
        "args": {"foo": 2},
        "id": "call_0_82c17db8-95df-452f-a4c2-03f809022134",
        "type": "tool_call",
    }

    assert tool.invoke(tool_call["args"]) == "good"
    assert tool_call == {
        "name": "sample_tool",
        "args": {"foo": 2},
        "id": "call_0_82c17db8-95df-452f-a4c2-03f809022134",
        "type": "tool_call",
    }

    assert await tool.ainvoke(tool_call["args"]) == "good"

    assert tool_call == {
        "name": "sample_tool",
        "args": {"foo": 2},
        "id": "call_0_82c17db8-95df-452f-a4c2-03f809022134",
        "type": "tool_call",
    }


def test_tool_invoke_does_not_mutate_inputs() -> None:
    """Verify that the inputs are not mutated when invoking a tool synchronously."""

    def sync_no_op(foo: int) -> str:
        return "good"

    async def async_no_op(foo: int) -> str:
        return "good"

    tool = StructuredTool(
        name="sample_tool",
        description="",
        args_schema={
            "type": "object",
            "required": ["foo"],
            "properties": {
                "seconds": {"type": "number", "description": "How big is foo"}
            },
        },
        coroutine=async_no_op,
        func=sync_no_op,
    )

    tool_call: ToolCall = {
        "name": "sample_tool",
        "args": {"foo": 2},
        "id": "call_0_82c17db8-95df-452f-a4c2-03f809022134",
        "type": "tool_call",
    }

    assert tool.invoke(tool_call["args"]) == "good"
    assert tool_call == {
        "name": "sample_tool",
        "args": {"foo": 2},
        "id": "call_0_82c17db8-95df-452f-a4c2-03f809022134",
        "type": "tool_call",
    }


def test_tool_args_schema_with_annotated_type() -> None:
    @tool
    def test_tool(
        query_fragments: Annotated[
            list[str],
            "A list of query fragments",
        ],
    ) -> list[str]:
        """Search the Internet and retrieve relevant result items."""
        return []

    assert test_tool.args == {
        "query_fragments": {
            "description": "A list of query fragments",
            "items": {"type": "string"},
            "title": "Query Fragments",
            "type": "array",
        }
    }


def test_tool_args_schema_with_pydantic_validator() -> None:
    """Test that Pydantic model validators can transform input structure.

    This test verifies that when a Pydantic validator wraps input in a nested
    structure, the tool correctly processes the transformed input rather than
    filtering it back to only the original input keys.

    Before the fix, the tool would filter the result to only include keys from
    the original tool_input, which broke validators that added structure.
    """

    class InnerModel(BaseModel):
        query: str
        count: int = 10

    class OuterModel(BaseModel):
        x: InnerModel

        @model_validator(mode="before")
        @classmethod
        def wrap_in_x(cls, data: Any) -> Any:
            """Wrap flat input in nested 'x' structure if not already wrapped."""
            if isinstance(data, dict) and "x" not in data:
                return {"x": data}
            return data

    @tool(args_schema=OuterModel)
    def search_with_nested_schema(x: InnerModel) -> str:
        """Search with nested input schema and validator transformation."""
        return f"Searched for '{x.query}' with count {x.count}"

    # Test 1: Direct nested input
    nested_input = {"x": {"query": "test", "count": 5}}
    result1 = search_with_nested_schema.invoke(nested_input)
    assert result1 == "Searched for 'test' with count 5"

    # Test 2: Flat input that gets wrapped by validator
    flat_input = {"query": "test query", "count": 3}
    result2 = search_with_nested_schema.invoke(flat_input)
    assert result2 == "Searched for 'test query' with count 3"

    # Test 3: Flat input with default values
    minimal_input = {"query": "minimal test"}
    result3 = search_with_nested_schema.invoke(minimal_input)
    assert result3 == "Searched for 'minimal test' with count 10"
