"""Unit tests for tool argument injection utilities."""

import pytest
from typing import Annotated

from langchain_core.tools import InjectedToolCallId
from langchain_core.tools.injection import (
    inject_runtime_args,
    validate_injected_args_present,
)


def test_inject_runtime_args_basic():
    """Test basic injection of tool_call_id."""
    def my_tool(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        return f"Result: {x}, Call ID: {tool_call_id}"

    injected_keys = frozenset(["tool_call_id"])
    kwargs = {"x": 42}

    result = inject_runtime_args(
        func=my_tool,
        injected_keys=injected_keys,
        kwargs=kwargs,
        tool_call_id="call_123",
    )

    assert result == {"x": 42, "tool_call_id": "call_123"}


def test_inject_runtime_args_missing_tool_call_id():
    """Test that missing tool_call_id raises an error."""
    def my_tool(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
        return f"Result: {x}"

    injected_keys = frozenset(["tool_call_id"])
    kwargs = {"x": 42}

    with pytest.raises(ValueError, match="requires 'tool_call_id' to be injected"):
        inject_runtime_args(
            func=my_tool,
            injected_keys=injected_keys,
            kwargs=kwargs,
            tool_call_id=None,  # Missing!
        )


def test_inject_runtime_args_already_present():
    """Test that injection doesn't override existing values in kwargs."""
    def my_tool(x: int, tool_call_id: str) -> str:
        return f"Result: {x}"

    injected_keys = frozenset(["tool_call_id"])
    # tool_call_id already in kwargs (edge case, shouldn't happen normally)
    kwargs = {"x": 42, "tool_call_id": "existing_id"}

    result = inject_runtime_args(
        func=my_tool,
        injected_keys=injected_keys,
        kwargs=kwargs,
        tool_call_id="new_id",
    )

    # Should NOT override existing value
    assert result == {"x": 42, "tool_call_id": "existing_id"}


def test_validate_injected_args_present_success():
    """Test validation passes when all injected args are present."""
    def my_tool(x: int, tool_call_id: str) -> str:
        return f"Result: {x}"

    injected_keys = frozenset(["tool_call_id"])
    kwargs = {"x": 42, "tool_call_id": "call_123"}

    # Should not raise
    validate_injected_args_present(
        func=my_tool,
        injected_keys=injected_keys,
        kwargs=kwargs,
    )


def test_validate_injected_args_present_missing():
    """Test validation fails when injected args are missing."""
    def my_tool(x: int, tool_call_id: str) -> str:
        return f"Result: {x}"

    injected_keys = frozenset(["tool_call_id"])
    kwargs = {"x": 42}  # Missing tool_call_id!

    with pytest.raises(RuntimeError, match="BUG.*missing from kwargs"):
        validate_injected_args_present(
            func=my_tool,
            injected_keys=injected_keys,
            kwargs=kwargs,
        )


def test_inject_runtime_args_multiple_injected_params():
    """Test injection with multiple injected parameters."""
    def my_tool(
        x: int,
        tool_call_id: Annotated[str, InjectedToolCallId],
        run_id: str,
    ) -> str:
        return f"Result: {x}"

    injected_keys = frozenset(["tool_call_id", "run_id"])
    kwargs = {"x": 42}

    result = inject_runtime_args(
        func=my_tool,
        injected_keys=injected_keys,
        kwargs=kwargs,
        tool_call_id="call_123",
        run_id="run_456",
    )

    assert result == {"x": 42, "tool_call_id": "call_123", "run_id": "run_456"}
