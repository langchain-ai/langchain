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
)

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from typing_extensions import TypedDict

from langchain_core import tools
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import ToolMessage
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
    InjectedToolArg,
    SchemaAnnotationError,
    _is_message_content_block,
    _is_message_content_type,
    get_all_basemodel_annotations,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION, _create_subset_model
from tests.unit_tests.fake.callbacks import FakeCallbackHandler
from tests.unit_tests.pydantic_utils import _schema


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


def test_misannotated_base_tool_raises_error() -> None:
    """Test that a BaseTool with the incorrect typehint raises an exception.""" ""
    with pytest.raises(SchemaAnnotationError):

        class _MisAnnotatedTool(BaseTool):
            name: str = "structured_api"
            # This would silently be ignored without the custom metaclass
            args_schema: BaseModel = _MockSchema  # type: ignore
            description: str = "A Structured Tool"

            def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
                return f"{arg1} {arg2} {arg3}"

            async def _arun(
                self, arg1: int, arg2: bool, arg3: Optional[dict] = None
            ) -> str:
                raise NotImplementedError


def test_forward_ref_annotated_base_tool_accepted() -> None:
    """Test that a using forward ref annotation syntax is accepted.""" ""

    class _ForwardRefAnnotatedTool(BaseTool):
        name: str = "structured_api"
        args_schema: "type[BaseModel]" = _MockSchema
        description: str = "A Structured Tool"

        def _run(self, arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
            return f"{arg1} {arg2} {arg3}"

        async def _arun(
            self, arg1: int, arg2: bool, arg3: Optional[dict] = None
        ) -> str:
            raise NotImplementedError


def test_subclass_annotated_base_tool_accepted() -> None:
    """Test BaseTool child w/ custom schema isn't overwritten."""

    class _ForwardRefAnnotatedTool(BaseTool):
        name: str = "structured_api"
        args_schema: type[_MockSchema] = _MockSchema
        description: str = "A Structured Tool"

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
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func, BaseTool)
    assert tool_func.args_schema == _MockSchema

    @tool(args_schema=_MockSchemaV1)
    def tool_func_v1(arg1: int, arg2: bool, arg3: Optional[dict] = None) -> str:
        return f"{arg1} {arg2} {arg3}"

    assert isinstance(tool_func_v1, BaseTool)
    assert tool_func_v1.args_schema == _MockSchemaV1


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
        _schema(structured_tool_input.args_schema)["properties"]
        == _schema(_MockSchema)["properties"]
        == structured_tool_input.args
    )


def test_args_kwargs_filtered() -> None:
    class _SingleArgToolWithKwargs(BaseTool):
        name: str = "single_arg_tool"
        description: str = "A  single arged tool with kwargs"

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
        """Docstring

        Args:
            bar: int
            baz: List[str]
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
        """Docstring thing.

        """
        return "invoke foo"

    structured_tool = StructuredTool.from_function(foo)

    assert structured_tool.run("") == "invoke foo"
    assert structured_tool.run({}) == "invoke foo"


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
    with pytest.raises(ValueError, match="Function must have a docstring"):

        @tool
        def search_api(query: str) -> str:
            return "API result"


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

    def handling(e: ToolException) -> str:
        return expected

    _tool = _FakeExceptionTool(handle_tool_error=handling)
    actual = _tool.run({})
    assert expected == actual


def test_exception_handling_non_tool_exception() -> None:
    _tool = _FakeExceptionTool(exception=ValueError())
    with pytest.raises(ValueError):
        _tool.run({})


async def test_async_exception_handling_bool() -> None:
    _tool = _FakeExceptionTool(handle_tool_error=True)
    expected = "Tool execution error"
    actual = await _tool.arun({})
    assert expected == actual


async def test_async_exception_handling_str() -> None:
    expected = "foo bar"
    _tool = _FakeExceptionTool(handle_tool_error=expected)
    actual = await _tool.arun({})
    assert expected == actual


async def test_async_exception_handling_callable() -> None:
    expected = "foo bar"

    def handling(e: ToolException) -> str:
        return expected

    _tool = _FakeExceptionTool(handle_tool_error=handling)
    actual = await _tool.arun({})
    assert expected == actual


async def test_async_exception_handling_non_tool_exception() -> None:
    _tool = _FakeExceptionTool(exception=ValueError())
    with pytest.raises(ValueError):
        await _tool.arun({})


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
    _tool = _MockStructuredTool(handle_validation_error=True)
    actual = _tool.run({})
    assert expected == actual


def test_validation_error_handling_str() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"
    _tool = _MockStructuredTool(handle_validation_error=expected)
    actual = _tool.run({})
    assert expected == actual


def test_validation_error_handling_callable() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"

    def handling(e: Union[ValidationError, ValidationErrorV1]) -> str:
        return expected

    _tool = _MockStructuredTool(handle_validation_error=handling)
    actual = _tool.run({})
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
        ) -> Union[str, dict[str, Any]]:
            raise NotImplementedError

        def _run(self) -> str:
            return "dummy"

        async def _arun(self) -> str:
            return "dummy"

    _tool = _RaiseNonValidationErrorTool(handle_validation_error=handler)  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        _tool.run({})


async def test_async_validation_error_handling_bool() -> None:
    """Test that validation errors are handled correctly."""
    expected = "Tool input validation error"
    _tool = _MockStructuredTool(handle_validation_error=True)
    actual = await _tool.arun({})
    assert expected == actual


async def test_async_validation_error_handling_str() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"
    _tool = _MockStructuredTool(handle_validation_error=expected)
    actual = await _tool.arun({})
    assert expected == actual


async def test_async_validation_error_handling_callable() -> None:
    """Test that validation errors are handled correctly."""
    expected = "foo bar"

    def handling(e: Union[ValidationError, ValidationErrorV1]) -> str:
        return expected

    _tool = _MockStructuredTool(handle_validation_error=handling)
    actual = await _tool.arun({})
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
        ) -> Union[str, dict[str, Any]]:
            raise NotImplementedError

        def _run(self) -> str:
            return "dummy"

        async def _arun(self) -> str:
            return "dummy"

    _tool = _RaiseNonValidationErrorTool(handle_validation_error=handler)  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        await _tool.arun({})


def test_optional_subset_model_rewrite() -> None:
    class MyModel(BaseModel):
        a: Optional[str] = None
        b: str
        c: Optional[list[Optional[str]]] = None

    model2 = _create_subset_model("model2", MyModel, ["a", "b", "c"])

    assert set(_schema(model2)["required"]) == {"b"}


@pytest.mark.parametrize(
    "inputs, expected",
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
        assert foo.invoke(inputs) == expected  # type: ignore
    else:
        with pytest.raises(ValidationError):
            foo.invoke(inputs)  # type: ignore


def test_tool_pass_context() -> None:
    @tool
    def foo(bar: str) -> str:
        """The foo."""
        config = ensure_config()
        assert config["configurable"]["foo"] == "not-bar"
        assert bar == "baz"
        return bar

    assert foo.invoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}}) == "baz"  # type: ignore


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
        await foo.ainvoke({"bar": "baz"}, {"configurable": {"foo": "not-bar"}}) == "baz"  # type: ignore
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

    def _run(self, bar: Any, bar_config: RunnableConfig, **kwargs: Any) -> Any:
        return assert_bar(bar, bar_config)


FooBase.model_rebuild()


class AFooBase(FooBase):
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
    assert foo1.description == "The foo."  # type: ignore

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
    args_schema = _schema(foo1.args_schema)  # type: ignore
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
    args_schema = _schema(foo2.args_schema)  # type: ignore
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
    def foo3(
        bar: str, baz: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """The foo.

        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    as_tool = tool(foo3, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)  # type: ignore
    assert args_schema["description"] == expected["description"]
    assert args_schema["properties"] == expected["properties"]

    # Test parameterless tool does not raise error for missing Args section
    # in docstring.
    def foo4() -> str:
        """The foo."""
        return "bar"

    as_tool = tool(foo4, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)  # type: ignore
    assert args_schema["description"] == expected["description"]

    def foo5(run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """The foo."""
        return "bar"

    as_tool = tool(foo5, parse_docstring=True)
    args_schema = _schema(as_tool.args_schema)  # type: ignore
    assert args_schema["description"] == expected["description"]


def test_tool_invalid_docstrings() -> None:
    # Test invalid docstrings
    def foo3(bar: str, baz: int) -> str:
        """The foo."""
        return bar

    def foo4(bar: str, baz: int) -> str:
        """The foo.
        Args:
            bar: The bar.
            baz: The baz.
        """
        return bar

    def foo5(bar: str, baz: int) -> str:
        """The foo.

        Args:
            banana: The bar.
            monkey: The baz.
        """
        return bar

    for func in [foo3, foo4, foo5]:
        with pytest.raises(ValueError):
            _ = tool(func, parse_docstring=True)


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
    args_schema = _schema(foo1.args_schema)  # type: ignore
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


class _MockStructuredToolWithRawOutput(BaseTool):
    name: str = "structured_api"
    args_schema: type[BaseModel] = _MockSchema
    description: str = "A Structured Tool"
    response_format: Literal["content_and_artifact"] = "content_and_artifact"

    def _run(
        self, arg1: int, arg2: bool, arg3: Optional[dict] = None
    ) -> tuple[str, dict]:
        return f"{arg1} {arg2}", {"arg1": arg1, "arg2": arg2, "arg3": arg3}


@tool("structured_api", response_format="content_and_artifact")
def _mock_structured_tool_with_artifact(
    arg1: int, arg2: bool, arg3: Optional[dict] = None
) -> tuple[str, dict]:
    """A Structured Tool"""
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
    """foo.

    Args:
        x: abc
        y: 123
    """
    return y


class InjectedTool(BaseTool):
    name: str = "foo"
    description: str = "foo."

    def _run(self, x: int, y: Annotated[str, InjectedToolArg]) -> Any:
        """foo.

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

    def _run(self, x: int, y: str) -> Any:
        return y


@tool("foo", args_schema=fooSchema)
def injected_tool_with_schema(x: int, y: str) -> str:
    return y


@pytest.mark.parametrize("tool_", [InjectedTool()])
def test_tool_injected_arg_without_schema(tool_: BaseTool) -> None:
    assert _schema(tool_.get_input_schema()) == {
        "title": "foo",
        "description": "foo.\n\nArgs:\n    x: abc\n    y: 123",
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
        {"name": "foo", "args": {"x": 5, "y": "bar"}, "id": "123", "type": "tool_call"}
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
        {"name": "foo", "args": {"x": 5, "y": "bar"}, "id": "123", "type": "tool_call"}
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
        {"name": "foo", "args": {"x": 5, "y": "bar"}, "id": "123", "type": "tool_call"}
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
    assert tool_.tool_call_schema.model_json_schema() == {
        "title": "foo",
        "description": "foo.",
        "type": "object",
        "properties": {"x": {"description": "abc", "title": "X", "type": "integer"}},
        "required": ["x"],
    }
    assert tool_.invoke({"x": 5, "y": "bar"}) == "bar"
    assert tool_.invoke(
        {"name": "foo", "args": {"x": 5, "y": "bar"}, "id": "123", "type": "tool_call"}
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
    from pydantic.v1 import BaseModel as BaseModelV1

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

        def _run(self, *args: Any, **kwargs: Any) -> str:
            return "foo"

    tool = SomeTool(
        name="some_tool", description="some description", args_schema=pydantic_model
    )

    input_schema = tool.get_input_schema()
    input_json_schema = (
        input_schema.model_json_schema()
        if hasattr(input_schema, "model_json_schema")
        else input_schema.schema()
    )
    assert input_json_schema == {
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "string"},
        },
        "required": ["a", "b"],
        "title": pydantic_model.__name__,
        "type": "object",
    }

    tool_schema = tool.tool_call_schema
    tool_json_schema = (
        tool_schema.model_json_schema()
        if hasattr(tool_schema, "model_json_schema")
        else tool_schema.schema()
    )
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
    # Check with whatever pydantic model is passed in and not via v1 namespace
    from pydantic import BaseModel

    class Foo(BaseModel):
        a: int
        b: str

    class SomeTool(BaseTool):
        # type ignoring here since we're allowing overriding a type
        # signature of pydantic.v1.BaseModel with pydantic.BaseModel
        # for pydantic 2!
        args_schema: type[BaseModel] = Foo  # type: ignore[assignment]

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

    assert tool.tool_call_schema.model_json_schema() == {
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
    from langchain_core.tools import StructuredTool

    def foo(a: int, b: str) -> str:
        """Hahaha"""
        return "foo"

    foo_tool = StructuredTool.from_function(
        func=foo,
        args_schema=pydantic_model,
    )

    assert foo_tool.invoke({"a": 5, "b": "hello"}) == "foo"

    args_schema = foo_tool.args_schema
    args_json_schema = (
        args_schema.model_json_schema()
        if hasattr(args_schema, "model_json_schema")
        else args_schema.schema()
    )
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
    input_json_schema = (
        input_schema.model_json_schema()
        if hasattr(input_schema, "model_json_schema")
        else input_schema.schema()
    )
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
def test__is_message_content_block(obj: Any, expected: bool) -> None:
    assert _is_message_content_block(obj) is expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ["foo", True],
        [valid_tool_result_blocks, True],
        [invalid_tool_result_blocks, False],
    ],
)
def test__is_message_content_type(obj: Any, expected: bool) -> None:
    assert _is_message_content_type(obj) is expected


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Testing pydantic v2.")
@pytest.mark.parametrize("use_v1_namespace", [True, False])
def test__get_all_basemodel_annotations_v2(use_v1_namespace: bool) -> None:
    A = TypeVar("A")

    if use_v1_namespace:
        from pydantic.v1 import BaseModel as BaseModel1

        class ModelA(BaseModel1, Generic[A], extra="allow"):
            a: A
    else:
        from pydantic import BaseModel as BaseModel2
        from pydantic import ConfigDict

        class ModelA(BaseModel2, Generic[A]):  # type: ignore[no-redef]
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


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 1, reason="Testing pydantic v1.")
def test__get_all_basemodel_annotations_v1() -> None:
    A = TypeVar("A")

    class ModelA(BaseModel, Generic[A], extra="allow"):
        a: A

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


def test_tool_annotations_preserved() -> None:
    """Test that annotations are preserved when creating a tool."""

    @tool
    def my_tool(val: int, other_val: Annotated[dict, "my annotation"]) -> str:
        """Tool docstring."""
        return "foo"

    schema = my_tool.get_input_schema()  # type: ignore[attr-defined]

    func = my_tool.func  # type: ignore[attr-defined]

    expected_type_hints = {
        name: hint
        for name, hint in func.__annotations__.items()
        if name in inspect.signature(func).parameters
    }
    assert schema.__annotations__ == expected_type_hints


@pytest.mark.skipif(PYDANTIC_MAJOR_VERSION != 2, reason="Testing pydantic v2.")
def test_tool_args_schema_pydantic_v2_with_metadata() -> None:
    from pydantic import BaseModel as BaseModelV2
    from pydantic import Field as FieldV2
    from pydantic import ValidationError as ValidationErrorV2

    class Foo(BaseModelV2):
        x: list[int] = FieldV2(
            description="List of integers", min_length=10, max_length=15
        )

    @tool(args_schema=Foo)
    def foo(x):  # type: ignore[no-untyped-def]
        """foo"""
        return x

    assert foo.tool_call_schema.model_json_schema() == {
        "description": "foo",
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
    with pytest.raises(ValidationErrorV2):
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
        assert hasattr(tools, module_name) and getattr(tools, module_name) is not None


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

    assert injected_tool.invoke({"x": 5, "foo": Foo()}) == "bar"  # type: ignore
