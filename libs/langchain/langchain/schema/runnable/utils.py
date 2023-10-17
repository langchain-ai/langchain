from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from inspect import signature
from itertools import groupby
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    TypeVar,
    Union,
)

Input = TypeVar("Input", contravariant=True)
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output", covariant=True)


async def gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    """Run a coroutine with a semaphore.
    Args:
        semaphore: The semaphore to use.
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    """Gather coroutines with a limit on the number of concurrent coroutines."""
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a run_manager argument."""
    try:
        return signature(callable).parameters.get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a config argument."""
    try:
        return signature(callable).parameters.get("config") is not None
    except ValueError:
        return False


class IsLocalDict(ast.NodeVisitor):
    """Check if a name is a local dict."""

    def __init__(self, name: str, keys: Set[str]) -> None:
        self.name = name
        self.keys = keys

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            # we've found a subscript access on the name we're looking for
            self.keys.add(node.slice.value)

    def visit_Call(self, node: ast.Call) -> Any:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.name
            and node.func.attr == "get"
            and len(node.args) in (1, 2)
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            # we've found a .get() call on the name we're looking for
            self.keys.add(node.args[0].value)


class IsFunctionArgDict(ast.NodeVisitor):
    """Check if the first argument of a function is a dict."""

    def __init__(self) -> None:
        self.keys: Set[str] = set()

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)


class GetLambdaSource(ast.NodeVisitor):
    """Get the source code of a lambda function."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.source: Optional[str] = None
        self.count = 0

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function."""
        self.count += 1
        if hasattr(ast, "unparse"):
            self.source = ast.unparse(node)


def get_function_first_arg_dict_keys(func: Callable) -> Optional[List[str]]:
    """Get the keys of the first argument of a function if it is a dict."""
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = IsFunctionArgDict()
        visitor.visit(tree)
        return list(visitor.keys) if visitor.keys else None
    except (SyntaxError, TypeError, OSError):
        return None


def get_lambda_source(func: Callable) -> Optional[str]:
    """Get the source code of a lambda function.

    Args:
        func: a callable that can be a lambda function

    Returns:
        str: the source code of the lambda function
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = GetLambdaSource()
        visitor.visit(tree)
        return visitor.source if visitor.count == 1 else None
    except (SyntaxError, TypeError, OSError):
        return None


def indent_lines_after_first(text: str, prefix: str) -> str:
    """Indent all lines of text after the first line.

    Args:
        text:  The text to indent
        prefix: Used to determine the number of spaces to indent

    Returns:
        str: The indented text
    """
    n_spaces = len(prefix)
    spaces = " " * n_spaces
    lines = text.splitlines()
    return "\n".join([lines[0]] + [spaces + line for line in lines[1:]])


class AddableDict(Dict[str, Any]):
    """
    Dictionary that can be added to another dictionary.
    """

    def __add__(self, other: AddableDict) -> AddableDict:
        chunk = AddableDict(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                chunk[key] = chunk[key] + other[key]
        return chunk

    def __radd__(self, other: AddableDict) -> AddableDict:
        chunk = AddableDict(other)
        for key in self:
            if key not in chunk or chunk[key] is None:
                chunk[key] = self[key]
            elif self[key] is not None:
                chunk[key] = chunk[key] + self[key]
        return chunk


_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    """Protocol for objects that support addition."""

    def __add__(self, __x: _T_contra) -> _T_co:
        ...


Addable = TypeVar("Addable", bound=SupportsAdd[Any, Any])


def add(addables: Iterable[Addable]) -> Optional[Addable]:
    """Add a sequence of addable objects together."""
    final = None
    for chunk in addables:
        if final is None:
            final = chunk
        else:
            final = final + chunk
    return final


async def aadd(addables: AsyncIterable[Addable]) -> Optional[Addable]:
    """Asynchronously add a sequence of addable objects together."""
    final = None
    async for chunk in addables:
        if final is None:
            final = chunk
        else:
            final = final + chunk
    return final


class ConfigurableField(NamedTuple):
    """A field that can be configured by the user."""

    id: str

    name: Optional[str] = None
    description: Optional[str] = None
    annotation: Optional[Any] = None

    def __hash__(self) -> int:
        return hash((self.id, self.annotation))


class ConfigurableFieldSingleOption(NamedTuple):
    """A field that can be configured by the user with a default value."""

    id: str
    options: Mapping[str, Any]
    default: str

    name: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.items()), self.default))


class ConfigurableFieldMultiOption(NamedTuple):
    """A field that can be configured by the user with multiple default values."""

    id: str
    options: Mapping[str, Any]
    default: Sequence[str]

    name: Optional[str] = None
    description: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.items()), tuple(self.default)))


AnyConfigurableField = Union[
    ConfigurableField, ConfigurableFieldSingleOption, ConfigurableFieldMultiOption
]


class ConfigurableFieldSpec(NamedTuple):
    """A field that can be configured by the user. It is a specification of a field."""

    id: str
    name: Optional[str]
    description: Optional[str]

    default: Any
    annotation: Any


def get_unique_config_specs(
    specs: Iterable[ConfigurableFieldSpec],
) -> Sequence[ConfigurableFieldSpec]:
    """Get the unique config specs from a sequence of config specs."""
    grouped = groupby(sorted(specs, key=lambda s: s.id), lambda s: s.id)
    unique: List[ConfigurableFieldSpec] = []
    for id, dupes in grouped:
        first = next(dupes)
        others = list(dupes)
        if len(others) == 0:
            unique.append(first)
        elif all(o == first for o in others):
            unique.append(first)
        else:
            raise ValueError(
                "RunnableSequence contains conflicting config specs"
                f"for {id}: {[first] + others}"
            )
    return unique
