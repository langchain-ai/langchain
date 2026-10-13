"""Tests for tools defined in modules with postponed annotations."""

from __future__ import annotations

from functools import update_wrapper
from inspect import Parameter, Signature
from typing import Annotated, Any

from pydantic import BaseModel

from langchain_core.tools import InjectedToolArg, StructuredTool
from langchain_core.tools.base import _DirectlyInjectedToolArg
from langchain_core.utils.pydantic import model_json_schema


class _PostponedRuntime(_DirectlyInjectedToolArg):
    """Runtime type whose annotation is postponed by the module future import."""


class _InputSchema(BaseModel):
    query: str


class _CallableTool:
    """Callable-object tool with annotations stored on `__call__`."""

    received_runtime: _PostponedRuntime | None = None

    def __call__(self, query: str, runtime: _PostponedRuntime) -> str:
        self.received_runtime = runtime
        return query


def test_callable_object_resolves_postponed_injected_arg() -> None:
    """Callable objects resolve annotations from their `__call__` method."""
    callable_tool = _CallableTool()
    tool = StructuredTool.from_function(
        func=callable_tool,
        name="callable_tool",
        description="Echo a query.",
        args_schema=_InputSchema,
    )
    runtime = _PostponedRuntime()

    assert tool._injected_args_keys == frozenset({"runtime"})
    assert tool.invoke({"query": "hello", "runtime": runtime}) == "hello"
    assert callable_tool.received_runtime is runtime

    tool_call_schema = tool.tool_call_schema
    assert not isinstance(tool_call_schema, dict)
    assert "runtime" not in model_json_schema(tool_call_schema)["properties"]


def test_callable_wrapper_uses_wrapped_function_annotations() -> None:
    """Callable wrappers resolve hints from their effective signature source."""
    received: dict[str, Any] = {}

    def wrapped(query: str, runtime: _PostponedRuntime) -> str:
        received["runtime"] = runtime
        return query

    class CallableWrapper:
        def __init__(self) -> None:
            update_wrapper(self, wrapped)

        def __call__(self, query: str, runtime: str) -> str:
            return wrapped(query, runtime)  # type: ignore[arg-type]

    callable_wrapper = CallableWrapper()
    tool = StructuredTool.from_function(
        func=callable_wrapper,
        name="callable_wrapper",
        description="Echo a query.",
        args_schema=_InputSchema,
    )
    runtime = _PostponedRuntime()

    assert tool._injected_args_keys == frozenset({"runtime"})
    assert tool.invoke({"query": "hello", "runtime": runtime}) == "hello"
    assert received["runtime"] is runtime


def test_callable_object_uses_explicit_signature_annotations() -> None:
    """An explicit signature takes precedence over `__call__` annotations."""
    received: dict[str, Any] = {}

    class CallableWithSignature:
        __signature__ = Signature(
            parameters=[
                Parameter("query", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
                Parameter(
                    "runtime",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=_PostponedRuntime,
                ),
            ]
        )

        def __call__(self, query: str, runtime: str) -> str:
            received["runtime"] = runtime
            return query

    callable_with_signature = CallableWithSignature()
    tool = StructuredTool.from_function(
        func=callable_with_signature,
        name="callable_with_signature",
        description="Echo a query.",
        args_schema=_InputSchema,
    )
    runtime = _PostponedRuntime()

    assert tool._injected_args_keys == frozenset({"runtime"})
    assert tool.invoke({"query": "hello", "runtime": runtime}) == "hello"
    assert received["runtime"] is runtime


def test_explicit_signature_resolves_string_injected_annotation() -> None:
    """String annotations on explicit signatures use callable globals."""
    received: dict[str, Any] = {}

    class CallableWithStringSignature:
        __signature__ = Signature(
            parameters=[
                Parameter("query", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
                Parameter(
                    "runtime",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation="Annotated[_PostponedRuntime, InjectedToolArg]",
                ),
            ]
        )

        def __call__(
            self, query: str, runtime: Annotated[_PostponedRuntime, InjectedToolArg]
        ) -> str:
            received["runtime"] = runtime
            return query

    tool = StructuredTool.from_function(
        func=CallableWithStringSignature(),
        name="callable_with_string_signature",
        description="Echo a query.",
        args_schema=_InputSchema,
    )
    runtime = _PostponedRuntime()

    assert tool._injected_args_keys == frozenset({"runtime"})
    assert tool.invoke({"query": "hello", "runtime": runtime}) == "hello"
    assert received["runtime"] is runtime
    tool_call_schema = tool.tool_call_schema
    assert not isinstance(tool_call_schema, dict)
    assert "runtime" not in model_json_schema(tool_call_schema)["properties"]


def test_class_callable_resolves_postponed_injected_arg() -> None:
    """Classes resolve constructor annotations postponed by the future import."""

    class ClassTool:
        def __init__(self, query: str, runtime: _PostponedRuntime) -> None:
            self.query = query
            self.runtime = runtime

    tool = StructuredTool.from_function(
        func=ClassTool,
        name="class_tool",
        description="Echo a query.",
        args_schema=_InputSchema,
    )
    runtime = _PostponedRuntime()

    assert tool._injected_args_keys == frozenset({"runtime"})
    result = tool.invoke({"query": "hello", "runtime": runtime})
    assert result.query == "hello"
    assert result.runtime is runtime

    tool_call_schema = tool.tool_call_schema
    assert not isinstance(tool_call_schema, dict)
    assert "runtime" not in model_json_schema(tool_call_schema)["properties"]
