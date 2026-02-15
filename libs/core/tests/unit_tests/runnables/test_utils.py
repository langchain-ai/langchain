import inspect
from collections.abc import Callable
from typing import Any, NoReturn

import pytest

from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.utils import (
    get_function_nonlocals,
    get_lambda_source,
    indent_lines_after_first,
)


@pytest.mark.parametrize(
    ("func", "expected_source"),
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),  # noqa: FURB136
    ],
)
def test_get_lambda_source(func: Callable[..., Any], expected_source: str) -> None:
    """Test get_lambda_source function."""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    ("text", "prefix", "expected_output"),
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function."""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output


global_agent = RunnableLambda[str, str](lambda x: x * 3)


def test_nonlocals() -> None:
    agent = RunnableLambda[str, str](lambda x: x * 2)

    def my_func(value: str, agent: dict[str, str]) -> str:
        return agent.get("agent_name", value)

    def my_func2(value: str) -> str:
        return str(agent.get("agent_name", value))  # type: ignore[attr-defined]

    def my_func3(value: str) -> str:
        return agent.invoke(value)

    def my_func4(value: str) -> str:
        return global_agent.invoke(value)

    def my_func5() -> tuple[Callable[[str], str], RunnableLambda]:
        global_agent = RunnableLambda[str, str](lambda x: x * 3)

        def my_func6(value: str) -> str:
            return global_agent.invoke(value)

        return my_func6, global_agent

    assert get_function_nonlocals(my_func) == []
    assert get_function_nonlocals(my_func2) == []
    assert get_function_nonlocals(my_func3) == [agent.invoke]
    assert get_function_nonlocals(my_func4) == [global_agent.invoke]
    func, nl = my_func5()
    assert get_function_nonlocals(func) == [nl.invoke]
    assert RunnableLambda(my_func3).deps == [agent]
    assert RunnableLambda(my_func4).deps == [global_agent]
    assert RunnableLambda(func).deps == [nl]


def test_deps_does_not_call_inspect_getsource() -> None:
    original = inspect.getsource
    error_message = "inspect.getsource was called while computing deps"
    def explode(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError(error_message)
    inspect.getsource = explode
    try:
        agent = RunnableLambda(lambda x: x)
        class Box:
            def __init__(self, a: RunnableLambda) -> None:
                self.agent = a
        box = Box(agent)
        def my_func(x: str) -> str:
            return box.agent.invoke(x)
        r = RunnableLambda(my_func)
        _ = r.deps
    finally:
        inspect.getsource = original


def test_deps_is_cached_on_instance() -> None:
    r = RunnableLambda(lambda x: x)
    _ = r.deps
    assert "deps" in r.__dict__
