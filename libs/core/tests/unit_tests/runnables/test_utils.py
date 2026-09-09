import functools
import gc
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    AddableDict,
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

    def my_func5() -> tuple[Callable[[str], str], RunnableLambda[str, str]]:
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


def test_get_function_nonlocals_bound_method_does_not_leak() -> None:
    """A bound method must not keep its owning instance alive forever.

    Regression test for https://github.com/langchain-ai/langchain/issues/30667:
    `get_function_nonlocals` used to be wrapped directly in `@lru_cache`, whose
    internal dict holds a strong reference to whatever is passed as the cache
    key. For a bound method, that key holds `__self__` alive via
    `__self__`/`__func__`, so the owning object (and anything it references)
    was kept alive for as long as it stayed in the 256-entry LRU cache -- even
    after every other reference to it was dropped.
    """

    class Owner:
        def __init__(self) -> None:
            self.payload = list(range(1000))

        def call(self, value: dict[str, Any]) -> dict[str, Any]:
            return value

    owner = Owner()
    ref = weakref.ref(owner)

    get_function_nonlocals(owner.call)
    del owner
    gc.collect()

    assert ref() is None


def test_get_function_nonlocals_bound_method_no_leak_via_config_specs() -> None:
    """Same as above, but through the realistic `RunnableLambda` trigger path.

    `get_function_nonlocals` is only reached through `RunnableLambda.deps`,
    which is consumed by `config_specs` and `get_graph` -- not by a plain
    `invoke`. Composite runnables (sequences, parallels, routers, ...) touch
    `config_specs` on their steps, so this is the realistic way the leak
    surfaces, e.g. when a chain containing a bound-method `RunnableLambda` is
    introspected or rendered.
    """

    class Owner:
        def __init__(self) -> None:
            self.payload = list(range(1000))

        def call(self, value: dict[str, Any]) -> dict[str, Any]:
            return value

    owner = Owner()
    ref = weakref.ref(owner)

    chain = RunnableLambda(owner.call) | RunnablePassthrough()
    chain.invoke({})
    assert chain.config_specs == []

    del owner
    del chain
    gc.collect()

    assert ref() is None


def test_get_function_nonlocals_callable_instance_does_not_leak() -> None:
    """A callable class instance passed directly must not leak either.

    The same strong-reference-as-cache-key problem applies to any callable
    that isn't a plain function -- not just bound methods.
    """

    class CallableOwner:
        def __init__(self) -> None:
            self.payload = list(range(1000))

        def __call__(self, value: dict[str, Any]) -> dict[str, Any]:
            return value

    instance = CallableOwner()
    ref = weakref.ref(instance)

    get_function_nonlocals(instance)
    del instance
    gc.collect()

    assert ref() is None


def test_get_function_nonlocals_partial_does_not_leak() -> None:
    """A `functools.partial` capturing a heavy argument must not leak it."""

    class Heavy:
        def __init__(self) -> None:
            self.payload = list(range(1000))

    def process(_heavy: Heavy, value: dict[str, Any]) -> dict[str, Any]:
        return value

    heavy = Heavy()
    ref = weakref.ref(heavy)
    partial_func = functools.partial(process, heavy)

    get_function_nonlocals(partial_func)
    del heavy
    del partial_func
    gc.collect()

    assert ref() is None


def test_addable_dict_add_incompatible_types_raises() -> None:
    left = AddableDict({"count": 1})
    right = AddableDict({"count": "some_string"})
    with pytest.raises(
        TypeError,
        match=r"Cannot add incompatible types for key 'count': 'int' and 'str'\.",
    ):
        left + right


def test_addable_dict_radd_incompatible_types_raises() -> None:
    left = AddableDict({"count": 1})
    right = AddableDict({"count": "some_string"})
    with pytest.raises(
        TypeError,
        match=r"Cannot add incompatible types for key 'count': 'int' and 'str'\.",
    ):
        right.__radd__(left)


def test_addable_dict_add_none_seeded_key_is_unaffected() -> None:
    left = AddableDict({"data": None})
    right = AddableDict({"data": {"a": 1}})
    assert (left + right) == AddableDict({"data": {"a": 1}})
