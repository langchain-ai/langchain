"""Unit tests for CompensationMiddleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import BaseTool, tool

from langchain.agents.middleware.compensation import (
    CompensationMiddleware,
)
from langchain.agents.middleware.types import (
    ToolCallRequest,
)


def _fake_runtime(tools: list[BaseTool] | None = None) -> SimpleNamespace:
    """Create a fake runtime with a tools list for testing.

    Args:
        tools: List of ``BaseTool`` instances to include in the runtime.

    Returns:
        A ``SimpleNamespace`` whose ``tools`` attribute is a list, matching
        the real ``ToolRuntime.tools: list[BaseTool]`` type.
    """
    return SimpleNamespace(tools=tools or [])


def _make_request(
    *,
    tool_name: str,
    state: dict[str, Any] | None = None,
    runtime: SimpleNamespace | None = None,
) -> ToolCallRequest:
    """Create a minimal ToolCallRequest for testing.

    Args:
        tool_name: The name of the tool being called.
        state: Optional agent state dict; defaults to ``{"messages": []}``.
        runtime: Optional fake runtime; defaults to an empty tool list.

    Returns:
        A ``ToolCallRequest`` suitable for passing to middleware methods.
    """
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

    runtime = _fake_runtime([refund_card])

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

    runtime = _fake_runtime([compensation_a, compensation_b])

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

    runtime = _fake_runtime([failing_compensation_tool])

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

    runtime = _fake_runtime([refund_card])

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


@pytest.mark.asyncio
async def test_async_success_adds_recovery_entry() -> None:
    """Test async successful tool execution appends a recovery entry."""
    middleware = CompensationMiddleware(
        compensation_pairs={
            "charge_card": "refund_card",
        },
        compensation_schemas={
            "charge_card": lambda result: {"payment_id": result["payment_id"]},
        },
    )

    request = _make_request(tool_name="charge_card")

    async def mock_handler(req: ToolCallRequest) -> dict[str, Any]:
        return {"payment_id": "async_pay_456"}

    result = await middleware.awrap_tool_call(request, mock_handler)

    assert result["payment_id"] == "async_pay_456"

    recovery_log = request.state["recovery_log"]
    assert len(recovery_log) == 1
    assert recovery_log[0]["tool"] == "charge_card"
    assert recovery_log[0]["compensation_tool"] == "refund_card"
    assert recovery_log[0]["compensation_args"] == {"payment_id": "async_pay_456"}


def test_missing_compensation_tool_is_skipped_during_rollback() -> None:
    """Test that a missing compensation tool in the runtime is silently skipped.

    If the recovery log references a tool that is not present in
    ``runtime.tools``, the rollback should continue with the remaining entries
    and the original exception must still be re-raised.
    """
    executed: list[str] = []

    @tool
    def present_tool() -> str:
        """A compensation tool that is present in the runtime."""
        executed.append("present")
        return "done"

    middleware = CompensationMiddleware(
        compensation_pairs={},
        compensation_schemas={},
    )

    state: dict[str, Any] = {
        "messages": [],
        "recovery_log": [
            {
                "tool": "tool_a",
                "compensation_tool": "ghost_tool",  # not in runtime
                "compensation_args": {},
            },
            {
                "tool": "tool_b",
                "compensation_tool": "present_tool",
                "compensation_args": {},
            },
        ],
    }

    # Only present_tool is in the runtime — ghost_tool is absent
    runtime = _fake_runtime([present_tool])

    request = _make_request(
        tool_name="failing_tool",
        state=state,
        runtime=runtime,
    )

    def mock_handler(req: ToolCallRequest) -> Any:
        msg = "original error"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="original error"):
        middleware.wrap_tool_call(request, mock_handler)

    # present_tool should still have been compensated
    assert executed == ["present"]


def test_multiple_successes_then_failure_rolls_back_all() -> None:
    """Test that multiple succeeded tool calls are all rolled back on failure.

    Simulates two tool calls that succeed (adding entries to the recovery log
    via wrap_tool_call) followed by a third call that fails.  All two
    compensation tools must be called in reverse order.
    """
    rolled_back: list[str] = []

    @tool
    def cancel_flight(booking_id: str) -> str:
        """Cancel a flight booking."""
        rolled_back.append(f"flight:{booking_id}")
        return "cancelled"

    @tool
    def cancel_hotel(reservation_id: str) -> str:
        """Cancel a hotel reservation."""
        rolled_back.append(f"hotel:{reservation_id}")
        return "cancelled"

    middleware = CompensationMiddleware(
        compensation_pairs={
            "book_flight": "cancel_flight",
            "book_hotel": "cancel_hotel",
        },
        compensation_schemas={
            "book_flight": lambda r: {"booking_id": r["booking_id"]},
            "book_hotel": lambda r: {"reservation_id": r["reservation_id"]},
        },
    )

    state: dict[str, Any] = {"messages": []}
    runtime = _fake_runtime([cancel_flight, cancel_hotel])

    # --- First call: book_flight succeeds ---
    request1 = _make_request(
        tool_name="book_flight",
        state=state,
        runtime=runtime,
    )
    middleware.wrap_tool_call(
        request1,
        lambda req: {"booking_id": "FL-001"},
    )

    # --- Second call: book_hotel succeeds ---
    request2 = _make_request(
        tool_name="book_hotel",
        state=state,
        runtime=runtime,
    )
    middleware.wrap_tool_call(
        request2,
        lambda req: {"reservation_id": "HT-001"},
    )

    assert len(state["recovery_log"]) == 2

    # --- Third call: rent_car fails → triggers rollback ---
    request3 = _make_request(
        tool_name="rent_car",
        state=state,
        runtime=runtime,
    )

    def failing_handler(req: ToolCallRequest) -> Any:
        msg = "no cars available"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="no cars available"):
        middleware.wrap_tool_call(request3, failing_handler)

    # Compensations must fire in reverse order: hotel first, then flight
    assert rolled_back == ["hotel:HT-001", "flight:FL-001"]
