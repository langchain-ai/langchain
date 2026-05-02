"""Tests for AgentSession — interrupt/resume and time-travel/branching."""

import sys

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import AgentSession, create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)


def _make_echo_agent(
    tool_calls: list[list[dict]] | None = None,
    *,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> tuple:
    """Create a simple agent with a checkpointer for session tests."""

    @tool
    def echo(query: str) -> str:
        """Echo the input."""
        return f"echo: {query}"

    checkpointer = InMemorySaver()
    model = FakeToolCallingModel(
        tool_calls=tool_calls
        or [
            [{"args": {"query": "hi"}, "id": "t1", "name": "echo"}],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[echo],
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    return agent, checkpointer


def test_session_run_basic() -> None:
    """Session.run executes the agent and returns a result."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    result = session.run("hello")
    assert "messages" in result
    assert len(result["messages"]) >= 1


def test_session_run_string_input() -> None:
    """Session.run accepts a plain string and converts it to HumanMessage."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    result = session.run("hello")
    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hello"


def test_session_run_dict_input() -> None:
    """Session.run accepts a dict input directly."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    result = session.run({"messages": [HumanMessage("hello")]})
    assert "messages" in result


def test_session_run_list_input() -> None:
    """Session.run accepts a list of messages."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    result = session.run([HumanMessage("hello")])
    assert "messages" in result


def test_session_thread_id_auto_generated() -> None:
    """Session generates a thread_id if none is provided."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    assert session.thread_id is not None
    assert len(session.thread_id) > 0


def test_session_thread_id_custom() -> None:
    """Session uses a custom thread_id when provided."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent, thread_id="my-thread")

    assert session.thread_id == "my-thread"


def test_session_config() -> None:
    """Session.config returns the expected configurable dict."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent, thread_id="t1")

    assert session.config == {"configurable": {"thread_id": "t1"}}


def test_session_get_state() -> None:
    """Session.get_state returns the current state after running."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    state = session.get_state()

    assert state.values is not None
    assert "messages" in state.values


def test_session_get_history() -> None:
    """Session.get_history returns checkpoint snapshots."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    history = list(session.get_history())

    assert len(history) > 0
    for snapshot in history:
        assert snapshot.config is not None


def test_session_get_history_with_limit() -> None:
    """Session.get_history respects the limit parameter."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    history_limited = list(session.get_history(limit=1))

    assert len(history_limited) == 1


def test_session_interrupt_and_resume() -> None:
    """Session can interrupt before tools and resume execution."""
    agent, _ = _make_echo_agent(interrupt_before=["tools"])
    session = AgentSession(agent)

    result = session.run("hello")
    state = session.get_state()

    assert state.next == ("tools",)

    result = session.resume()
    assert "messages" in result


def test_session_branch() -> None:
    """Session.branch creates a new session from a checkpoint."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    history = list(session.get_history())

    checkpoint_id = history[-1].config["configurable"]["checkpoint_id"]
    branched = session.branch(checkpoint_id=checkpoint_id)

    assert branched.thread_id != session.thread_id


def test_session_branch_with_custom_thread_id() -> None:
    """Session.branch uses a custom thread ID when provided."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    history = list(session.get_history())
    checkpoint_id = history[-1].config["configurable"]["checkpoint_id"]

    branched = session.branch(
        checkpoint_id=checkpoint_id,
        new_thread_id="my-branch",
    )

    assert branched.thread_id == "my-branch"


def test_session_branch_preserves_original() -> None:
    """Branching does not modify the original session's state."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    session.run("hello")
    original_state = session.get_state()

    history = list(session.get_history())
    checkpoint_id = history[-1].config["configurable"]["checkpoint_id"]
    session.branch(checkpoint_id=checkpoint_id)

    after_state = session.get_state()
    assert len(original_state.values["messages"]) == len(after_state.values["messages"])


@pytest.mark.anyio
async def test_session_arun() -> None:
    """Async session.arun works correctly."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    result = await session.arun("hello")
    assert "messages" in result


@pytest.mark.anyio
async def test_session_aget_state() -> None:
    """Async session.aget_state returns state after running."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)

    await session.arun("hello")
    state = await session.aget_state()

    assert "messages" in state.values


@pytest.mark.anyio
async def test_session_aresume() -> None:
    """Async session.aresume works after interrupt."""
    agent, _ = _make_echo_agent(interrupt_before=["tools"])
    session = AgentSession(agent)

    await session.arun("hello")
    state = await session.aget_state()
    assert state.next == ("tools",)

    result = await session.aresume()
    assert "messages" in result


def test_session_agent_property() -> None:
    """Session.agent returns the underlying compiled graph."""
    agent, _ = _make_echo_agent()
    session = AgentSession(agent)
    assert session.agent is agent
