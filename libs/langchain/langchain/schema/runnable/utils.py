from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from inspect import signature
from typing import Any, Callable, Coroutine, List, Optional, Set, TypeVar, Union

Input = TypeVar("Input")
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output")


async def gated_coro(semaphore: asyncio.Semaphore, coro: Coroutine) -> Any:
    async with semaphore:
        return await coro


async def gather_with_concurrency(n: Union[int, None], *coros: Coroutine) -> list:
    if n is None:
        return await asyncio.gather(*coros)

    semaphore = asyncio.Semaphore(n)

    return await asyncio.gather(*(gated_coro(semaphore, c) for c in coros))


def accepts_run_manager(callable: Callable[..., Any]) -> bool:
    try:
        return signature(callable).parameters.get("run_manager") is not None
    except ValueError:
        return False


def accepts_config(callable: Callable[..., Any]) -> bool:
    try:
        return signature(callable).parameters.get("config") is not None
    except ValueError:
        return False


class IsLocalDict(ast.NodeVisitor):
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
    def __init__(self) -> None:
        self.source: Optional[str] = None
        self.count = 0

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        self.count += 1
        if hasattr(ast, "unparse"):
            self.source = ast.unparse(node)


def get_function_first_arg_dict_keys(func: Callable) -> Optional[List[str]]:
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
