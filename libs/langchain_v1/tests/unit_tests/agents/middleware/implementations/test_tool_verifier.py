"""Tests for ToolVerifierMiddleware functionality."""

from datetime import datetime, timezone

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from langchain.agents.factory import create_agent
from langchain.agents.middleware import ToolVerifierMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for something."""
    return f"Results for: {query}"


@tool
def calculator(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def _model_with_single_call(tool_name: str, tool_args: dict, call_id: str = "1"):
    return FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name=tool_name, args=tool_args, id=call_id)],
            [],
        ]
    )


def _model_with_multiple_calls():
    return FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="search", args={"query": "test1"}, id="1"),
                ToolCall(name="search", args={"query": "test2"}, id="2"),
            ],
            [],
        ]
    )


def test_allow_tool_executes() -> None:
    """Verifier returns allow=True -> tool executes."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"
    assert "Results for: test" in tool_messages[0].content


def test_deny_tool_does_not_execute() -> None:
    """Verifier returns allow=False -> synthetic ToolMessage returned."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": False,
            "reason": "Not allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied: Not allowed" in tool_messages[0].content
    assert "Results for: test" not in tool_messages[0].content


def test_malformed_verdict_blocked() -> None:
    """Malformed verdict -> blocked."""

    def verifier(request: ToolCallRequest) -> dict:
        return {"invalid": "structure"}

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "missing 'allow' field" in tool_messages[0].content


def test_missing_allow_blocked() -> None:
    """Verdict missing 'allow' field -> blocked."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "reason": "No allow field",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "missing 'allow' field" in tool_messages[0].content


def test_verifier_raises_blocked() -> None:
    """Verifier raises exception -> blocked."""

    def verifier(request: ToolCallRequest) -> dict:
        raise ValueError("Verifier error")

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "Verification failed" in tool_messages[0].content


def test_verifier_returns_none_blocked() -> None:
    """Verifier returns None -> blocked."""

    def verifier(request: ToolCallRequest) -> dict | None:
        return None

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "not a dict" in tool_messages[0].content


def test_verifier_returns_wrong_type_blocked() -> None:
    """Verifier returns wrong type -> blocked."""

    def verifier(request: ToolCallRequest) -> str:
        return "not a dict"

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "not a dict" in tool_messages[0].content


def test_tool_call_id_preserved() -> None:
    """Exact tool_call_id is preserved verbatim."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": False,
            "reason": "Denied",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}, call_id="custom-id-123"),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "custom-id-123"


def test_same_name_args_different_ids_independent() -> None:
    """Same tool/args with different IDs -> independently verified."""
    call_count = 0

    def verifier(request: ToolCallRequest) -> dict:
        nonlocal call_count
        call_count += 1
        # Allow first call, deny second
        return {
            "allow": call_count == 1,
            "reason": f"Call {call_count}",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    ToolCall(name="search", args={"query": "test"}, id="1"),
                    ToolCall(name="search", args={"query": "test"}, id="2"),
                ],
                [],
            ]
        ),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search twice")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    # First call allowed, second denied
    assert tool_messages[0].tool_call_id == "1"
    assert tool_messages[0].status != "error"
    assert tool_messages[1].tool_call_id == "2"
    assert tool_messages[1].status == "error"
    assert call_count == 2


def test_mixed_allow_deny_calls() -> None:
    """Multiple tool calls with mixed allow/deny -> correct results."""

    def verifier(request: ToolCallRequest) -> dict:
        # Allow calls with even IDs, deny odd
        call_id = request.tool_call["id"]
        return {
            "allow": int(call_id) % 2 == 0,
            "reason": f"ID {call_id}",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_multiple_calls(),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search twice")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    # ID 1 denied, ID 2 allowed
    assert tool_messages[0].tool_call_id == "1"
    assert tool_messages[0].status == "error"
    assert tool_messages[1].tool_call_id == "2"
    assert tool_messages[1].status != "error"


def test_ordering_preserved() -> None:
    """Tool call order preserved in results."""
    call_order = []

    def verifier(request: ToolCallRequest) -> dict:
        call_order.append(request.tool_call["id"])
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_multiple_calls(),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search twice")]})

    assert call_order == ["1", "2"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert tool_messages[0].tool_call_id == "1"
    assert tool_messages[1].tool_call_id == "2"


def test_unknown_tool_behavior() -> None:
    """Unknown tool (not in tools list) -> verifier called."""
    verifier_called = False

    def verifier(request: ToolCallRequest) -> dict:
        nonlocal verifier_called
        verifier_called = True
        # Verifier sees unknown tool
        assert request.tool is None
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("unknown_tool", {"arg": "value"}),
        tools=[search],  # unknown_tool not in list
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    # Tool execution may fail because tool is unknown, but verifier should be called
    result = agent.invoke({"messages": [HumanMessage("Use unknown tool")]})

    # Verify the verifier was invoked
    assert verifier_called


def test_sync_verifier() -> None:
    """Sync verifier works on sync path."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


async def test_async_verifier() -> None:
    """Async verifier works on async path."""

    async def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


async def test_sync_verifier_async_path() -> None:
    """Sync verifier works through async middleware path."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


async def test_async_verifier_exception_blocked() -> None:
    """Async verifier raises exception -> blocked."""

    async def verifier(request: ToolCallRequest) -> dict:
        raise ValueError("boom")

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "Access denied" in tool_messages[0].content
    assert "Verification failed" in tool_messages[0].content
    # Original exception message should not be exposed
    assert "boom" not in tool_messages[0].content


def test_async_only_verifier_sync_path_raises() -> None:
    """Async-only verifier on sync path raises RuntimeError."""

    async def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    with pytest.raises(RuntimeError, match="async verifier"):
        agent.invoke({"messages": [HumanMessage("Search for test")]})


def test_no_middleware_regression() -> None:
    """No middleware -> existing behavior unchanged."""
    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[],  # No verifier middleware
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"
    assert "Results for: test" in tool_messages[0].content


def test_expiry_with_expires_at() -> None:
    """Verdict with expires_at in past -> denied."""
    # Use realistic timestamps for pinned-clock checks
    # evaluatedAt: 1 hour ago
    # expires_at: 30 minutes ago (already expired)
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    evaluated_at = (now - timedelta(hours=1)).isoformat()
    expires_at = (now - timedelta(minutes=30)).isoformat()

    def verifier_with_past_expiry(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": evaluated_at,
            "expires_at": expires_at,
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier_with_past_expiry)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "expired" in tool_messages[0].content


def test_expiry_with_future_expires_at() -> None:
    """Verdict with expires_at in future -> allowed."""
    # Use realistic timestamps for pinned-clock checks
    # evaluatedAt: 1 hour ago (in the past)
    # expires_at: 1 hour from now (in the future)
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    evaluated_at = (now - timedelta(hours=1)).isoformat()
    expires_at = (now + timedelta(hours=1)).isoformat()

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": evaluated_at,
            "expires_at": expires_at,
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


def test_tool_definition_mutation_detected() -> None:
    """Tool definition hash matches -> allowed."""
    # Pre-compute the hash for the search tool
    middleware = ToolVerifierMiddleware(lambda r: {})
    # Create a mock request to compute the hash
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    mock_request = TCR(
        tool_call={"name": "search", "args": {"query": "test"}, "id": "1"},
        tool=search,
        state={},
        runtime=None,
    )
    stored_hash = middleware._compute_tool_definition_hash(mock_request)

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
            "toolDefinitionHash": stored_hash,
        }

    # First call - hash matches
    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


def test_tool_definition_mismatch_denied() -> None:
    """Tool definition hash mismatch -> denied."""

    def verifier_with_wrong_hash(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
            "toolDefinitionHash": "sha256:wronghashvalue",
        }

    agent = create_agent(
        model=_model_with_single_call("calculator", {"x": 1, "y": 2}),
        tools=[calculator],
        middleware=[ToolVerifierMiddleware(verifier_with_wrong_hash)],
    )

    result = agent.invoke({"messages": [HumanMessage("Calculate")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "mutated" in tool_messages[0].content


def test_no_hash_no_mutation_check() -> None:
    """No toolDefinitionHash in verdict -> no mutation check."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
            # No toolDefinitionHash
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"


def test_invalid_timestamp_format_denied() -> None:
    """Invalid evaluatedAt format -> denied."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "not-a-timestamp",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "ISO timestamp" in tool_messages[0].content


def test_allow_not_boolean_denied() -> None:
    """Allow field not boolean -> denied."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": "yes",  # String instead of bool
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "boolean" in tool_messages[0].content


def test_reason_not_string_denied() -> None:
    """Reason field not string -> denied."""

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": 123,  # Integer instead of string
            "evaluatedAt": "2026-09-22T10:00:00Z",
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "reason' must be a string" in tool_messages[0].content


def test_pinned_clock_expires_before_evaluated_denied() -> None:
    """Pinned-clock: expires_at before evaluatedAt -> denied (invalid verdict)."""
    evaluated_time = "2026-09-22T10:00:00Z"
    expires_before = "2026-09-22T09:00:00Z"  # Before evaluatedAt

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": evaluated_time,
            "expires_at": expires_before,  # Invalid: expires before evaluated
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "expires_at must be after evaluatedAt" in tool_messages[0].content


def test_pinned_clock_future_evaluated_at_denied() -> None:
    """Pinned-clock regression: future evaluatedAt with future expires_at is denied.

    This test distinguishes the pinned-clock coherence checks from simple
    dispatch-time expiry. A simple dispatch-time check would allow this verdict
    (since expires_at is in the future), but the pinned-clock check rejects it
    because evaluatedAt is in the future (incoherent verdict).

    This proves the implementation validates the verifier's evaluation clock,
    not just the current time against expiry.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    # Both timestamps are in the future - incoherent verdict
    future_evaluated = (now + timedelta(hours=1)).isoformat()
    future_expires = (now + timedelta(hours=2)).isoformat()

    def verifier(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": future_evaluated,
            "expires_at": future_expires,
        }

    agent = create_agent(
        model=_model_with_single_call("search", {"query": "test"}),
        tools=[search],
        middleware=[ToolVerifierMiddleware(verifier)],
    )

    result = agent.invoke({"messages": [HumanMessage("Search for test")]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "evaluatedAt must be in the past" in tool_messages[0].content


def test_schema_hash_identical_for_same_tool() -> None:
    """Identical name + description + schema → same hash."""
    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    # Create two requests for the same tool
    request1 = TCR(
        tool_call={"name": "search", "args": {"query": "test1"}, "id": "1"},
        tool=search,
        state={},
        runtime=None,
    )
    request2 = TCR(
        tool_call={"name": "search", "args": {"query": "test2"}, "id": "2"},
        tool=search,
        state={},
        runtime=None,
    )

    hash1 = middleware._compute_tool_definition_hash(request1)
    hash2 = middleware._compute_tool_definition_hash(request2)

    assert hash1 == hash2


def test_schema_hash_changes_with_schema_mutation() -> None:
    """Schema-only mutation: same name/description, different schema → hash changes."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    class SearchArgs(BaseModel):
        """Original search args."""

        query: str

    class SearchArgsV2(BaseModel):
        """Mutated search args with additional field."""

        query: str
        limit: int = 10

    # Create two tools with same name and description but different schemas
    def search_v1_func(query: str) -> str:
        return f"Results for: {query}"

    def search_v2_func(query: str, limit: int = 10) -> str:
        return f"Results for: {query} (limit {limit})"

    search_v1 = StructuredTool.from_function(
        func=search_v1_func,
        name="search_tool",
        description="Search for something",
        args_schema=SearchArgs,
    )

    search_v2 = StructuredTool.from_function(
        func=search_v2_func,
        name="search_tool",
        description="Search for something",
        args_schema=SearchArgsV2,
    )

    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    request1 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test"}, "id": "1"},
        tool=search_v1,
        state={},
        runtime=None,
    )
    request2 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test", "limit": 5}, "id": "2"},
        tool=search_v2,
        state={},
        runtime=None,
    )

    hash1 = middleware._compute_tool_definition_hash(request1)
    hash2 = middleware._compute_tool_definition_hash(request2)

    # Hashes should differ due to schema mutation (name and description are identical)
    assert hash1 != hash2


def test_schema_hash_changes_with_description() -> None:
    """Description-only mutation: same name/schema, different description → hash changes."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    class SearchArgs(BaseModel):
        """Search args."""

        query: str

    # Create two tools with same name and schema but different descriptions
    def search_func(query: str) -> str:
        return f"Results for: {query}"

    search_v1 = StructuredTool.from_function(
        func=search_func,
        name="search_tool",
        description="Original description",
        args_schema=SearchArgs,
    )

    search_v2 = StructuredTool.from_function(
        func=search_func,
        name="search_tool",
        description="Mutated description",
        args_schema=SearchArgs,
    )

    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    request1 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test"}, "id": "1"},
        tool=search_v1,
        state={},
        runtime=None,
    )
    request2 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test"}, "id": "2"},
        tool=search_v2,
        state={},
        runtime=None,
    )

    hash1 = middleware._compute_tool_definition_hash(request1)
    hash2 = middleware._compute_tool_definition_hash(request2)

    # Hashes should differ due to description mutation (name and schema are identical)
    assert hash1 != hash2


def test_schema_hash_changes_with_name() -> None:
    """Name-only mutation: same description/schema, different name → hash changes."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel

    class SearchArgs(BaseModel):
        """Search args."""

        query: str

    # Create two tools with same description and schema but different names
    def search_func(query: str) -> str:
        return f"Results for: {query}"

    search_v1 = StructuredTool.from_function(
        func=search_func,
        name="search_tool_v1",
        description="Search for something",
        args_schema=SearchArgs,
    )

    search_v2 = StructuredTool.from_function(
        func=search_func,
        name="search_tool_v2",
        description="Search for something",
        args_schema=SearchArgs,
    )

    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    request1 = TCR(
        tool_call={"name": "search_tool_v1", "args": {"query": "test"}, "id": "1"},
        tool=search_v1,
        state={},
        runtime=None,
    )
    request2 = TCR(
        tool_call={"name": "search_tool_v2", "args": {"query": "test"}, "id": "2"},
        tool=search_v2,
        state={},
        runtime=None,
    )

    hash1 = middleware._compute_tool_definition_hash(request1)
    hash2 = middleware._compute_tool_definition_hash(request2)

    # Hashes should differ due to name mutation (description and schema are identical)
    assert hash1 != hash2


def test_schema_hash_canonical_key_ordering() -> None:
    """Canonical key ordering: nested dict insertion order does not change hash."""
    from langchain_core.tools import StructuredTool

    # Create two dict schemas with equivalent content but different key ordering
    schema1 = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    schema2 = {"required": ["query"], "properties": {"query": {"type": "string"}}, "type": "object"}

    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    # Create tools with dict schemas (different key ordering)
    def search_func(query: str) -> str:
        return f"Results for: {query}"

    search1 = StructuredTool.from_function(
        func=search_func,
        name="search_tool",
        description="Search",
        args_schema=schema1,
    )

    search2 = StructuredTool.from_function(
        func=search_func,
        name="search_tool",
        description="Search",
        args_schema=schema2,
    )

    request1 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test"}, "id": "1"},
        tool=search1,
        state={},
        runtime=None,
    )
    request2 = TCR(
        tool_call={"name": "search_tool", "args": {"query": "test"}, "id": "2"},
        tool=search2,
        state={},
        runtime=None,
    )

    hash1 = middleware._compute_tool_definition_hash(request1)
    hash2 = middleware._compute_tool_definition_hash(request2)

    # Hashes should be identical because canonical JSON serialization normalizes key order
    assert hash1 == hash2


def test_hash_computation_failure_blocked() -> None:
    """Hash computation failure -> blocked with denial message."""
    from unittest.mock import MagicMock

    # Create a mock tool with a non-serializable args_schema
    mock_tool = MagicMock()
    mock_tool.name = "mock_tool"
    mock_tool.description = "Mock tool"
    mock_tool.args_schema = object()  # Non-serializable type

    middleware = ToolVerifierMiddleware(lambda r: {})
    from langgraph.prebuilt.tool_node import ToolCallRequest as TCR

    request = TCR(
        tool_call={"name": "mock_tool", "args": {}, "id": "1"},
        tool=mock_tool,
        state={},
        runtime=None,
    )

    # Hash computation should fail
    with pytest.raises(ValueError, match="Cannot deterministically serialize"):
        middleware._compute_tool_definition_hash(request)

    # Now test end-to-end: when hash computation fails during verdict validation,
    # the middleware should return a denial message
    def verifier_with_hash(request: ToolCallRequest) -> dict:
        return {
            "allow": True,
            "reason": "Allowed",
            "evaluatedAt": "2026-09-22T10:00:00Z",
            "toolDefinitionHash": "sha256:somehash",
        }

    # Use the mock tool directly in the middleware validation
    validation_result = middleware._validate_verdict(request, verifier_with_hash(request))

    # Should return a denial message
    assert validation_result is not None
    assert validation_result.status == "error"
    assert "Failed to compute tool definition hash" in validation_result.content
