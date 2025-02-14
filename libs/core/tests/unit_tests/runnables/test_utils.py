import sys
from typing import Callable

import pytest

from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.utils import (
    get_function_nonlocals,
    get_lambda_source,
    indent_lines_after_first,
)


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="Requires python version >= 3.9 to run."
)
@pytest.mark.parametrize(
    "func, expected_source",
    [
        (lambda x: x * 2, "lambda x: x * 2"),
        (lambda a, b: a + b, "lambda a, b: a + b"),
        (lambda x: x if x > 0 else 0, "lambda x: x if x > 0 else 0"),  # noqa: FURB136
    ],
)
def test_get_lambda_source(func: Callable, expected_source: str) -> None:
    """Test get_lambda_source function."""
    source = get_lambda_source(func)
    assert source == expected_source


@pytest.mark.parametrize(
    "text,prefix,expected_output",
    [
        ("line 1\nline 2\nline 3", "1", "line 1\n line 2\n line 3"),
        ("line 1\nline 2\nline 3", "ax", "line 1\n  line 2\n  line 3"),
    ],
)
def test_indent_lines_after_first(text: str, prefix: str, expected_output: str) -> None:
    """Test indent_lines_after_first function."""
    indented_text = indent_lines_after_first(text, prefix)
    assert indented_text == expected_output


global_agent = RunnableLambda(lambda x: x * 3)


def test_nonlocals() -> None:
    agent = RunnableLambda(lambda x: x * 2)

    def my_func(input: str, agent: dict[str, str]) -> str:
        return agent.get("agent_name", input)

    def my_func2(input: str) -> str:
        return agent.get("agent_name", input)  # type: ignore[attr-defined]

    def my_func3(input: str) -> str:
        return agent.invoke(input)

    def my_func4(input: str) -> str:
        return global_agent.invoke(input)

    def my_func5() -> tuple[Callable[[str], str], RunnableLambda]:
        global_agent = RunnableLambda(lambda x: x * 3)

        def my_func6(input: str) -> str:
            return global_agent.invoke(input)

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
