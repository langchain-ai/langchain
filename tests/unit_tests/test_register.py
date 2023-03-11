"""Test functionality for register functions"""

from typing import Any, Dict, List, Tuple

import pytest

from langchain.register import (
    _LLM_TOOLS,
    _TOOLS,
    AGENT_TO_CLASS,
    register,
    register_agent,
    register_llm_tool,
    register_tool,
)


def test_register() -> None:
    """Test that register works."""
    _registry: Dict[str, Tuple[Any, List[str]]] = {}

    @register("foo", _registry)
    def foo():
        pass

    assert _registry["foo"] == foo

    with pytest.raises(KeyError):

        @register("foo", _registry)
        def foo_new():
            pass


def test_register_agent() -> None:
    """Test that register_agent works."""

    @register_agent("foo")
    class Foo:
        pass

    assert AGENT_TO_CLASS["foo"] == Foo


def test_register_tool() -> None:
    """Test that register_tool works."""

    @register_tool("foo")
    class Foo:
        pass

    assert _TOOLS["foo"] == (Foo, [])

    _TOOLS.pop("foo")

    @register_tool("foo", ["bar"])
    class Foo:
        pass

    assert _TOOLS["foo"] == (Foo, ["bar"])


def test_register_llm_tool() -> None:
    """Test that register_llm_tool works."""

    @register_llm_tool("foo")
    class Foo:
        pass

    assert _LLM_TOOLS["foo"] == (Foo, [])

    _LLM_TOOLS.pop("foo")

    @register_llm_tool("foo", ["bar"])
    class Foo:
        pass

    assert _LLM_TOOLS["foo"] == (Foo, ["bar"])
