"""Unit tests for CompensationMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import tool

from langchain.agents.middleware.compensation import (
    CompensationMiddleware,
)
from langchain.agents.middleware.types import (
    ToolCallRequest,
)


def _fake_runtime(tools: dict[str, Any] | None = None) -> SimpleNamespace:
    """Create a fake runtime with a tools dict for testing."""
    return SimpleNamespace(tools=tools or {})


def _make_request(
    *,
    tool_name: str,
    state: dict[str, Any] | None = None,
    runtime: SimpleNamespace | None = None,
) -> ToolCallRequest:
    """Create a minimal ToolCallRequest for testing."""
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {},
            "id": "call_123",
        },
        tool=None,
        state=state or {"messages": []},
        runtime=runtime or _fake_runtime(),
    )


def test_successful_tool_call_adds_recovery_entry() -> None:
    """Test successful tool execution appends recovery log."""
    middleware = CompensationMiddleware(
        compensation_pairs={
            "charge_card": "refund_card",
        },
        compensation_schemas={
            "charge_card": lambda result: {
                "payment_id": result["payment_id"],
            }
        },
    )

    request = _make_request(
        tool_name="charge_card",
    )

    def mock_handler(
        req: ToolCallRequest,
    ) -> dict[str, Any]:
        return {
            "payment_id": "pay_123",
        }

    result = middleware.wrap_tool_call(
        request,
        mock_handler,
    )

    assert result["payment_id"] == "pay_123"

    recovery_log = request.state["recovery_log"]

    assert len(recovery_log) == 1

    assert recovery_log[0]["tool"] == "charge_card"

    assert recovery_log[0]["compensation_tool"] == "refund_card"

    assert recovery_log[0]["compensation_args"] == {"payment_id": "pay_123"}


def test_tool_without_compensation_mapping() -> None:
    """Test normal tool execution without compensation."""
    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    request = _make_request(
        tool_name="normal_tool",
    )

    def mock_handler(
        req: ToolCallRequest,
    ) -> dict[str, bool]:
        return {"ok": True}

    result = middleware.wrap_tool_call(
        request,
        mock_handler,
    )

    assert result == {"ok": True}

    assert request.state["recovery_log"] == []


def test_failure_triggers_compensation() -> None:
    """Test rollback executes on failure."""
    executed: list[str] = []

    @tool
    def refund_card(
        payment_id: str,
    ) -> str:
        """Refund a credit card charge."""
        executed.append(payment_id)
        return "refunded"

    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    state: dict[str, Any] = {
        "messages": [],
        "recovery_log": [
            {
                "tool": "charge_card",
                "compensation_tool": "refund_card",
                "compensation_args": {
                    "payment_id": "pay_1",
                },
            }
        ],
    }

    runtime = _fake_runtime(
        {
            "refund_card": refund_card,
        }
    )

    request = _make_request(
        tool_name="send_email",
        state=state,
        runtime=runtime,
    )

    def mock_handler(
        req: ToolCallRequest,
    ) -> Any:
        msg = "tool failure"
        raise RuntimeError(msg)

    with pytest.raises(
        RuntimeError,
        match="tool failure",
    ):
        middleware.wrap_tool_call(
            request,
            mock_handler,
        )

    assert executed == ["pay_1"]


def test_reverse_order_compensation() -> None:
    """Test rollback executes in reverse order."""
    execution_order: list[str] = []

    @tool
    def compensation_a() -> str:
        """Compensate action A."""
        execution_order.append("A")
        return "A"

    @tool
    def compensation_b() -> str:
        """Compensate action B."""
        execution_order.append("B")
        return "B"

    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    state: dict[str, Any] = {
        "messages": [],
        "recovery_log": [
            {
                "tool": "tool_a",
                "compensation_tool": "compensation_a",
                "compensation_args": {},
            },
            {
                "tool": "tool_b",
                "compensation_tool": "compensation_b",
                "compensation_args": {},
            },
        ],
    }

    runtime = _fake_runtime(
        {
            "compensation_a": compensation_a,
            "compensation_b": compensation_b,
        }
    )

    request = _make_request(
        tool_name="failing_tool",
        state=state,
        runtime=runtime,
    )

    def mock_handler(
        req: ToolCallRequest,
    ) -> Any:
        msg = "failure"
        raise RuntimeError(msg)

    with pytest.raises(
        RuntimeError,
        match="failure",
    ):
        middleware.wrap_tool_call(
            request,
            mock_handler,
        )

    assert execution_order == [
        "B",
        "A",
    ]


def test_compensation_failure_does_not_mask_original_error() -> None:
    """Test rollback failure preserves original exception."""

    @tool
    def failing_compensation_tool() -> str:
        """Compensation tool that always fails."""
        msg = "rollback failure"
        raise ValueError(msg)

    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    state: dict[str, Any] = {
        "messages": [],
        "recovery_log": [
            {
                "tool": "tool_a",
                "compensation_tool": "failing_compensation_tool",
                "compensation_args": {},
            }
        ],
    }

    runtime = _fake_runtime(
        {
            "failing_compensation_tool": failing_compensation_tool,
        }
    )

    request = _make_request(
        tool_name="main_tool",
        state=state,
        runtime=runtime,
    )

    def mock_handler(
        req: ToolCallRequest,
    ) -> Any:
        msg = "main failure"
        raise RuntimeError(msg)

    with pytest.raises(
        RuntimeError,
        match="main failure",
    ):
        middleware.wrap_tool_call(
            request,
            mock_handler,
        )


async def test_async_compensation_execution() -> None:
    """Test async rollback execution."""
    executed: list[str] = []

    @tool
    async def refund_card(
        payment_id: str,
    ) -> str:
        """Async refund a credit card charge."""
        executed.append(payment_id)
        return "refunded"

    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    state: dict[str, Any] = {
        "messages": [],
        "recovery_log": [
            {
                "tool": "charge_card",
                "compensation_tool": "refund_card",
                "compensation_args": {
                    "payment_id": "async_1",
                },
            }
        ],
    }

    runtime = _fake_runtime(
        {
            "refund_card": refund_card,
        }
    )

    request = _make_request(
        tool_name="failing_tool",
        state=state,
        runtime=runtime,
    )

    async def mock_handler(
        req: ToolCallRequest,
    ) -> Any:
        msg = "async failure"
        raise RuntimeError(msg)

    with pytest.raises(
        RuntimeError,
        match="async failure",
    ):
        await middleware.awrap_tool_call(
            request,
            mock_handler,
        )

    assert executed == ["async_1"]
