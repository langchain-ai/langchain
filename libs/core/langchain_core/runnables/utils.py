"""Utility code for runnables."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

from typing_extensions import TypeGuard, override

from langchain_core.runnables.schema import StreamEvent

# Re-export create-model for backwards compatibility
from langchain_core.utils.pydantic import create_model as create_model

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
    """Gather coroutines with a limit on the number of concurrent coroutines.

    Args:
        n: The number of coroutines to run concurrently.
        *coros: The coroutines to run.

    Returns:
        The results of the coroutines.
    """
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a run_manager argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a run_manager argument, False otherwise.
    """
    try:
        return signature(callable).parameters.get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a config argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a config argument, False otherwise.
    """
    try:
        return signature(callable).parameters.get("config") is not None
    except ValueError:
        return False


def accepts_context(callable: Callable[..., Any]) -> bool:
    """Check if a callable accepts a context argument.

    Args:
        callable: The callable to check.

    Returns:
        bool: True if the callable accepts a context argument, False otherwise.
    """
    try:
        return signature(callable).parameters.get("context") is not None
    except ValueError:
        return False


@lru_cache(maxsize=1)
def asyncio_accepts_context() -> bool:
    return accepts_context(asyncio.create_task)


class IsLocalDict(ast.NodeVisitor):
    """Check if a name is a local dict."""

    def __init__(self, name: str, keys: set[str]) -> None:
        """Initialize the visitor.

        Args:
            name: The name to check.
            keys: The keys to populate.
        """
        self.name = name
        self.keys = keys

    @override
    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Visit a subscript node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.name
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            # we've found a subscript access on the name we're looking for
            self.keys.add(node.slice.value)

    @override
    def visit_Call(self, node: ast.Call) -> Any:
        """Visit a call node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
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
        self.keys: set[str] = set()

    @override
    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node.body)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visit an async function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if not node.args.args:
            return
        input_arg_name = node.args.args[0].arg
        IsLocalDict(input_arg_name, self.keys).visit(node)


class NonLocals(ast.NodeVisitor):
    """Get nonlocal variables accessed."""

    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.stores: set[str] = set()

    @override
    def visit_Name(self, node: ast.Name) -> Any:
        """Visit a name node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.stores.add(node.id)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit an attribute node.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        if isinstance(node.ctx, ast.Load):
            parent = node.value
            attr_expr = node.attr
            while isinstance(parent, ast.Attribute):
                attr_expr = parent.attr + "." + attr_expr
                parent = parent.value
            if isinstance(parent, ast.Name):
                self.loads.add(parent.id + "." + attr_expr)
                self.loads.discard(parent.id)
            elif isinstance(parent, ast.Call):
                if isinstance(parent.func, ast.Name):
                    self.loads.add(parent.func.id)
                else:
                    parent = parent.func
                    attr_expr = ""
                    while isinstance(parent, ast.Attribute):
                        if attr_expr:
                            attr_expr = parent.attr + "." + attr_expr
                        else:
                            attr_expr = parent.attr
                        parent = parent.value
                    if isinstance(parent, ast.Name):
                        self.loads.add(parent.id + "." + attr_expr)


class FunctionNonLocals(ast.NodeVisitor):
    """Get the nonlocal variables accessed of a function."""

    def __init__(self) -> None:
        self.nonlocals: set[str] = set()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visit an async function definition.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)

    @override
    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        visitor = NonLocals()
        visitor.visit(node)
        self.nonlocals.update(visitor.loads - visitor.stores)


class GetLambdaSource(ast.NodeVisitor):
    """Get the source code of a lambda function."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.source: Optional[str] = None
        self.count = 0

    @override
    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function.

        Args:
            node: The node to visit.

        Returns:
            Any: The result of the visit.
        """
        self.count += 1
        if hasattr(ast, "unparse"):
            self.source = ast.unparse(node)


def get_function_first_arg_dict_keys(func: Callable) -> Optional[list[str]]:
    """Get the keys of the first argument of a function if it is a dict.

    Args:
        func: The function to check.

    Returns:
        Optional[List[str]]: The keys of the first argument if it is a dict,
            None otherwise.
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = IsFunctionArgDict()
        visitor.visit(tree)
        return sorted(visitor.keys) if visitor.keys else None
    except (SyntaxError, TypeError, OSError, SystemError):
        return None


def get_lambda_source(func: Callable) -> Optional[str]:
    """Get the source code of a lambda function.

    Args:
        func: a Callable that can be a lambda function.

    Returns:
        str: the source code of the lambda function.
    """
    try:
        name = func.__name__ if func.__name__ != "<lambda>" else None
    except AttributeError:
        name = None
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = GetLambdaSource()
        visitor.visit(tree)
    except (SyntaxError, TypeError, OSError, SystemError):
        return name
    return visitor.source if visitor.count == 1 else name


@lru_cache(maxsize=256)
def get_function_nonlocals(func: Callable) -> list[Any]:
    """Get the nonlocal variables accessed by a function.

    Args:
        func: The function to check.

    Returns:
        List[Any]: The nonlocal variables accessed by the function.
    """
    try:
        code = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(code))
        visitor = FunctionNonLocals()
        visitor.visit(tree)
        values: list[Any] = []
        closure = (
            inspect.getclosurevars(func.__wrapped__)
            if hasattr(func, "__wrapped__") and callable(func.__wrapped__)
            else inspect.getclosurevars(func)
        )
        candidates = {**closure.globals, **closure.nonlocals}
        for k, v in candidates.items():
            if k in visitor.nonlocals:
                values.append(v)
            for kk in visitor.nonlocals:
                if "." in kk and kk.startswith(k):
                    vv = v
                    for part in kk.split(".")[1:]:
                        if vv is None:
                            break
                        else:
                            try:
                                vv = getattr(vv, part)
                            except AttributeError:
                                break
                    else:
                        values.append(vv)
    except (SyntaxError, TypeError, OSError, SystemError):
        return []

    return values


def indent_lines_after_first(text: str, prefix: str) -> str:
    """Indent all lines of text after the first line.

    Args:
        text: The text to indent.
        prefix: Used to determine the number of spaces to indent.

    Returns:
        str: The indented text.
    """
    n_spaces = len(prefix)
    spaces = " " * n_spaces
    lines = text.splitlines()
    return "\n".join([lines[0]] + [spaces + line for line in lines[1:]])


class AddableDict(dict[str, Any]):
    """Dictionary that can be added to another dictionary."""

    def __add__(self, other: AddableDict) -> AddableDict:
        chunk = AddableDict(self)
        for key in other:
            if key not in chunk or chunk[key] is None:
                chunk[key] = other[key]
            elif other[key] is not None:
                try:
                    added = chunk[key] + other[key]
                except TypeError:
                    added = other[key]
                chunk[key] = added
        return chunk

    def __radd__(self, other: AddableDict) -> AddableDict:
        chunk = AddableDict(other)
        for key in self:
            if key not in chunk or chunk[key] is None:
                chunk[key] = self[key]
            elif self[key] is not None:
                try:
                    added = chunk[key] + self[key]
                except TypeError:
                    added = self[key]
                chunk[key] = added
        return chunk


_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    """Protocol for objects that support addition."""

    def __add__(self, __x: _T_contra) -> _T_co: ...


Addable = TypeVar("Addable", bound=SupportsAdd[Any, Any])


def add(addables: Iterable[Addable]) -> Optional[Addable]:
    """Add a sequence of addable objects together.

    Args:
        addables: The addable objects to add.

    Returns:
        Optional[Addable]: The result of adding the addable objects.
    """
    final: Optional[Addable] = None
    for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


async def aadd(addables: AsyncIterable[Addable]) -> Optional[Addable]:
    """Asynchronously add a sequence of addable objects together.

    Args:
        addables: The addable objects to add.

    Returns:
        Optional[Addable]: The result of adding the addable objects.
    """
    final: Optional[Addable] = None
    async for chunk in addables:
        final = chunk if final is None else final + chunk
    return final


class ConfigurableField(NamedTuple):
    """Field that can be configured by the user.

    Parameters:
        id: The unique identifier of the field.
        name: The name of the field. Defaults to None.
        description: The description of the field. Defaults to None.
        annotation: The annotation of the field. Defaults to None.
        is_shared: Whether the field is shared. Defaults to False.
    """

    id: str

    name: Optional[str] = None
    description: Optional[str] = None
    annotation: Optional[Any] = None
    is_shared: bool = False

    def __hash__(self) -> int:
        return hash((self.id, self.annotation))


class ConfigurableFieldSingleOption(NamedTuple):
    """Field that can be configured by the user with a default value.

    Parameters:
        id: The unique identifier of the field.
        options: The options for the field.
        default: The default value for the field.
        name: The name of the field. Defaults to None.
        description: The description of the field. Defaults to None.
        is_shared: Whether the field is shared. Defaults to False.
    """

    id: str
    options: Mapping[str, Any]
    default: str

    name: Optional[str] = None
    description: Optional[str] = None
    is_shared: bool = False

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), self.default))


class ConfigurableFieldMultiOption(NamedTuple):
    """Field that can be configured by the user with multiple default values.

    Parameters:
        id: The unique identifier of the field.
        options: The options for the field.
        default: The default values for the field.
        name: The name of the field. Defaults to None.
        description: The description of the field. Defaults to None.
        is_shared: Whether the field is shared. Defaults to False.
    """

    id: str
    options: Mapping[str, Any]
    default: Sequence[str]

    name: Optional[str] = None
    description: Optional[str] = None
    is_shared: bool = False

    def __hash__(self) -> int:
        return hash((self.id, tuple(self.options.keys()), tuple(self.default)))


AnyConfigurableField = Union[
    ConfigurableField, ConfigurableFieldSingleOption, ConfigurableFieldMultiOption
]


class ConfigurableFieldSpec(NamedTuple):
    """Field that can be configured by the user. It is a specification of a field.

    Parameters:
        id: The unique identifier of the field.
        annotation: The annotation of the field.
        name: The name of the field. Defaults to None.
        description: The description of the field. Defaults to None.
        default: The default value for the field. Defaults to None.
        is_shared: Whether the field is shared. Defaults to False.
        dependencies: The dependencies of the field. Defaults to None.
    """

    id: str
    annotation: Any

    name: Optional[str] = None
    description: Optional[str] = None
    default: Any = None
    is_shared: bool = False
    dependencies: Optional[list[str]] = None


def get_unique_config_specs(
    specs: Iterable[ConfigurableFieldSpec],
) -> list[ConfigurableFieldSpec]:
    """Get the unique config specs from a sequence of config specs.

    Args:
        specs: The config specs.

    Returns:
        List[ConfigurableFieldSpec]: The unique config specs.

    Raises:
        ValueError: If the runnable sequence contains conflicting config specs.
    """
    grouped = groupby(
        sorted(specs, key=lambda s: (s.id, *(s.dependencies or []))), lambda s: s.id
    )
    unique: list[ConfigurableFieldSpec] = []
    for id, dupes in grouped:
        first = next(dupes)
        others = list(dupes)
        if len(others) == 0 or all(o == first for o in others):
            unique.append(first)
        else:
            msg = (
                "RunnableSequence contains conflicting config specs"
                f"for {id}: {[first] + others}"
            )
            raise ValueError(msg)
    return unique


class _RootEventFilter:
    def __init__(
        self,
        *,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
    ) -> None:
        """Utility to filter the root event in the astream_events implementation.

        This is simply binding the arguments to the namespace to make save on
        a bit of typing in the astream_events implementation.
        """
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

    def include_event(self, event: StreamEvent, root_type: str) -> bool:
        """Determine whether to include an event."""
        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        event_tags = event.get("tags") or []

        if self.include_names is not None:
            include = include or event["name"] in self.include_names
        if self.include_types is not None:
            include = include or root_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in event_tags)

        if self.exclude_names is not None:
            include = include and event["name"] not in self.exclude_names
        if self.exclude_types is not None:
            include = include and root_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(
                tag not in self.exclude_tags for tag in event_tags
            )

        return include


def is_async_generator(
    func: Any,
) -> TypeGuard[Callable[..., AsyncIterator]]:
    """Check if a function is an async generator.

    Args:
        func: The function to check.

    Returns:
        TypeGuard[Callable[..., AsyncIterator]: True if the function is
            an async generator, False otherwise.
    """
    return (
        inspect.isasyncgenfunction(func)
        or hasattr(func, "__call__")  # noqa: B004
        and inspect.isasyncgenfunction(func.__call__)
    )


def is_async_callable(
    func: Any,
) -> TypeGuard[Callable[..., Awaitable]]:
    """Check if a function is async.

    Args:
        func: The function to check.

    Returns:
        TypeGuard[Callable[..., Awaitable]: True if the function is async,
            False otherwise.
    """
    return (
        asyncio.iscoroutinefunction(func)
        or hasattr(func, "__call__")  # noqa: B004
        and asyncio.iscoroutinefunction(func.__call__)
    )
