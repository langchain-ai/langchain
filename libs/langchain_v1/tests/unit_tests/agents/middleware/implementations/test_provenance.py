"""Tests for ProvenanceMiddleware functionality."""

from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.factory import create_agent
from langchain.agents.middleware.provenance import ProvenanceMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Found record: id=abc-123, name={query}"


@tool(side_effects=True)
def delete_record(record_id: str) -> str:
    """Delete a record by ID."""
    return f"Deleted {record_id}"


@tool(side_effects=True)
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool
def lookup(key: str) -> str:
    """Look up a value by key."""
    return f"value-for-{key}"


def test_provenance_initialization_defaults() -> None:
    """Test ProvenanceMiddleware initialization with default values."""
    guard = ProvenanceMiddleware()
    assert guard.include_user_inputs is True
    assert guard.min_value_length == 3


def test_provenance_initialization_custom() -> None:
    """Test ProvenanceMiddleware initialization with custom values."""
    guard = ProvenanceMiddleware(include_user_inputs=False, min_value_length=5)
    assert guard.include_user_inputs is False
    assert guard.min_value_length == 5


def test_side_effecting_tool_blocked_without_provenance() -> None:
    """Side-effecting tool is blocked when args lack provenance."""
    model = FakeToolCallingModel(
        tool_calls=[
            # Agent immediately tries to delete with a hallucinated ID
            # (the ID does NOT appear in the user message)
            [ToolCall(name="delete_record", args={"record_id": "hallucinated-id-999"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Please delete that record")]},
        {"configurable": {"thread_id": "test-blocked"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1

    # The delete_record call should be blocked
    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    assert delete_msg.status == "error"
    assert "provenance" in delete_msg.content.lower()
    assert "record_id" in delete_msg.content


def test_side_effecting_tool_allowed_with_provenance() -> None:
    """Side-effecting tool succeeds when args come from prior tool output."""
    model = FakeToolCallingModel(
        tool_calls=[
            # Step 1: agent calls search tool (non-side-effecting, passes through)
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            # Step 2: agent uses the ID from search output to delete
            [ToolCall(name="delete_record", args={"record_id": "abc-123"}, id="2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search then delete")]},
        {"configurable": {"thread_id": "test-allowed"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    # search should succeed
    search_msg = next(m for m in tool_messages if m.name == "search")
    assert "abc-123" in search_msg.content

    # delete should succeed (abc-123 is in the search output)
    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    assert delete_msg.status != "error"
    assert "Deleted abc-123" in delete_msg.content


def test_non_side_effecting_tools_unaffected() -> None:
    """Non-side-effecting tools execute normally regardless of provenance."""
    model = FakeToolCallingModel(
        tool_calls=[
            # search is not side-effecting, should pass through even with no provenance
            [ToolCall(name="search", args={"query": "hallucinated-query-xyz"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search something")]},
        {"configurable": {"thread_id": "test-non-side-effect"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status != "error"
    assert "Found record" in tool_messages[0].content


def test_disabled_by_default_without_middleware() -> None:
    """Agent without ProvenanceMiddleware executes side-effecting tools freely."""
    model = FakeToolCallingModel(
        tool_calls=[
            # No prior search — directly delete with hallucinated ID
            [ToolCall(name="delete_record", args={"record_id": "hallucinated-id"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        # No middleware
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Delete something")]},
        {"configurable": {"thread_id": "test-no-middleware"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should succeed because no middleware is enforcing provenance
    assert tool_messages[0].status != "error"
    assert "Deleted hallucinated-id" in tool_messages[0].content


def test_user_input_as_provenance() -> None:
    """User message content counts as trusted provenance when include_user_inputs=True."""
    model = FakeToolCallingModel(
        tool_calls=[
            # Agent uses a value that the user provided in their message
            [ToolCall(name="delete_record", args={"record_id": "user-provided-id-42"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware(include_user_inputs=True)],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Please delete user-provided-id-42")]},
        {"configurable": {"thread_id": "test-user-provenance"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # Should succeed because the ID is in the user's message
    assert tool_messages[0].status != "error"
    assert "Deleted user-provided-id-42" in tool_messages[0].content


def test_user_input_provenance_disabled() -> None:
    """User messages do NOT count when include_user_inputs=False."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="delete_record", args={"record_id": "user-provided-id-42"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware(include_user_inputs=False)],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Please delete user-provided-id-42")]},
        {"configurable": {"thread_id": "test-no-user-provenance"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1

    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    # Should be blocked — user input is not trusted
    assert delete_msg.status == "error"
    assert "provenance" in delete_msg.content.lower()


def test_trivial_values_skip_provenance_check() -> None:
    """Boolean, None, and short string values are not provenance-checked."""

    @tool(side_effects=True)
    def update_setting(name: str, *, enabled: bool, value: str) -> str:
        """Update a setting."""
        return f"Updated {name}: enabled={enabled}, value={value}"

    model = FakeToolCallingModel(
        tool_calls=[
            # "name" is from user message, "enabled" is bool (skipped), "ok" is short (skipped)
            [
                ToolCall(
                    name="update_setting",
                    args={"name": "my-setting", "enabled": True, "value": "ok"},
                    id="1",
                )
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[update_setting],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Update my-setting")]},
        {"configurable": {"thread_id": "test-trivial"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    # "enabled" (bool) and "ok" (len < 3) are skipped
    # "my-setting" appears in user message
    assert tool_messages[0].status != "error"


def test_full_read_then_write_flow() -> None:
    """Full agent flow: read tool returns ID, write tool uses it."""
    model = FakeToolCallingModel(
        tool_calls=[
            # Step 1: lookup returns a value
            [ToolCall(name="lookup", args={"key": "project"}, id="1")],
            # Step 2: delete uses the looked-up value
            [ToolCall(name="delete_record", args={"record_id": "value-for-project"}, id="2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[lookup, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Lookup then delete")]},
        {"configurable": {"thread_id": "test-flow"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    lookup_msg = next(m for m in tool_messages if m.name == "lookup")
    assert "value-for-project" in lookup_msg.content

    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    assert delete_msg.status != "error"
    assert "Deleted value-for-project" in delete_msg.content


def test_multiple_args_partial_provenance_blocked() -> None:
    """If some args have provenance but others don't, execution is blocked."""
    model = FakeToolCallingModel(
        tool_calls=[
            # Search returns known data
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            # send_email: "to" is hallucinated, "body" references known content
            [
                ToolCall(
                    name="send_email",
                    args={"to": "hallucinated@evil.com", "body": "abc-123"},
                    id="2",
                )
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, send_email],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Search then email")]},
        {"configurable": {"thread_id": "test-partial"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    email_msg = next(m for m in tool_messages if m.name == "send_email")
    # Should be blocked because "to" arg is not in any trusted text
    assert email_msg.status == "error"
    assert "to=" in email_msg.content


async def test_provenance_async_blocked() -> None:
    """Async execution also blocks side-effecting tools without provenance."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="delete_record", args={"record_id": "async-hallucinated"}, id="1")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Please delete that record")]},
        {"configurable": {"thread_id": "test-async-blocked"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    assert delete_msg.status == "error"
    assert "provenance" in delete_msg.content.lower()


async def test_provenance_async_allowed() -> None:
    """Async execution allows side-effecting tools when provenance exists."""
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="1")],
            [ToolCall(name="delete_record", args={"record_id": "abc-123"}, id="2")],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[search, delete_record],
        middleware=[ProvenanceMiddleware()],
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Search then delete")]},
        {"configurable": {"thread_id": "test-async-allowed"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    delete_msg = next(m for m in tool_messages if m.name == "delete_record")
    assert delete_msg.status != "error"
    assert "Deleted abc-123" in delete_msg.content
