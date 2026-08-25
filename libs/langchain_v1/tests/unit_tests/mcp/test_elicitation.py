"""Unit tests for MCP elicitation interrupts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from mcp.types import TextContent
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.mcp import elicitation
from langchain.mcp import tools as mcp_tools
from langchain.mcp.callbacks import CallbackContext, Callbacks
from langchain.mcp.tools import convert_mcp_tool_to_langchain_tool, load_mcp_tools
from tests.unit_tests.agents.model import FakeToolCallingModel


class _ElicitationState(TypedDict):
    response: dict[str, object]


async def test_interrupt_elicitation_resumes_a_graph() -> None:
    """The bridge pauses and resumes an async LangGraph node with form content."""

    async def elicit(_state: _ElicitationState) -> _ElicitationState:
        result = elicitation.interrupt_for_elicitation(
            SimpleNamespace(
                mode="form",
                message="What date should the meeting occur?",
                requested_schema={"type": "object", "properties": {"date": {"type": "string"}}},
            ),
            CallbackContext(server_name="calendar"),
        )
        assert result.content is not None
        return {"response": result.content}

    graph = (
        StateGraph(_ElicitationState)
        .add_node("elicit", elicit)
        .add_edge(START, "elicit")
        .add_edge("elicit", END)
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": "elicitation"}}

    paused = await graph.ainvoke({"response": {}}, config)
    interrupt_value = paused["__interrupt__"][0].value
    assert interrupt_value["type"] == "mcp_elicitation"
    assert interrupt_value["mode"] == "form"

    resumed = await graph.ainvoke(
        Command(resume={"action": "accept", "content": {"date": "2026-12-24"}}),
        config,
    )
    assert resumed["response"] == {"date": "2026-12-24"}


async def test_create_agent_resumes_an_elicitation_enabled_mcp_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent pauses and resumes while executing a converted MCP tool."""

    @asynccontextmanager
    async def fake_create_session(_connection: object, *, mcp_callbacks: Any):
        class FakeSession:
            async def call_tool(
                self,
                _tool_name: str,
                _arguments: dict[str, Any],
                *,
                progress_callback: Any,
            ) -> Any:
                del progress_callback
                assert mcp_callbacks.elicitation_callback is not None
                response = await mcp_callbacks.elicitation_callback(
                    SimpleNamespace(),
                    SimpleNamespace(
                        mode="form",
                        message="Who should receive the invitation?",
                        requested_schema={
                            "type": "object",
                            "properties": {"email": {"type": "string"}},
                            "required": ["email"],
                        },
                    ),
                )
                assert response.action == "accept"
                return SimpleNamespace(
                    content=[
                        TextContent(type="text", text=f"Invited {response.content['email']}.")
                    ],
                    is_error=False,
                    structured_content=None,
                )

        yield FakeSession()

    monkeypatch.setattr(mcp_tools, "create_session", fake_create_session)
    mcp_tool = SimpleNamespace(
        name="create_invitation",
        description="Create a calendar invitation.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        annotations=None,
        meta=None,
    )
    tool = convert_mcp_tool_to_langchain_tool(
        None,
        mcp_tool,
        connection={},
        elicitation="interrupt",
        server_name="calendar",
    )
    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="create_invitation", args={"name": "Ada"}, id="call-1")],
            [],
        ]
    )
    agent = create_agent(model, tools=[tool], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "agent-elicitation"}}

    paused = await agent.ainvoke({"messages": [HumanMessage("Create an invitation")]}, config)
    interrupt_value = paused["__interrupt__"][0].value
    assert interrupt_value == {
        "type": "mcp_elicitation",
        "mode": "form",
        "server": "calendar",
        "message": "Who should receive the invitation?",
        "response_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    }

    completed = await agent.ainvoke(
        Command(resume={"action": "accept", "content": {"email": "ada@example.com"}}),
        config,
    )
    tool_messages = [
        message for message in completed["messages"] if isinstance(message, ToolMessage)
    ]
    assert tool_messages[-1].content[0]["text"] == "Invited ada@example.com."


async def test_interrupt_elicitation_callback_returns_accepted_form_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The callback turns a form request into an interrupt and MCP result."""
    seen: list[dict[str, Any]] = []

    def fake_interrupt(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload)
        return {"action": "accept", "content": {"email": "ada@example.com"}}

    monkeypatch.setattr(elicitation, "interrupt", fake_interrupt)
    mcp_callbacks = Callbacks().to_mcp_format(
        context=CallbackContext(server_name="calendar", tool_name="create_event"),
        elicitation="interrupt",
    )
    assert mcp_callbacks.elicitation_callback is not None

    result = await mcp_callbacks.elicitation_callback(
        SimpleNamespace(),
        SimpleNamespace(
            mode="form",
            message="Who should receive the invitation?",
            requested_schema={"type": "object", "properties": {"email": {"type": "string"}}},
        ),
    )

    assert result.action == "accept"
    assert result.content == {"email": "ada@example.com"}
    assert seen == [
        {
            "type": "mcp_elicitation",
            "mode": "form",
            "server": "calendar",
            "message": "Who should receive the invitation?",
            "response_schema": {"type": "object", "properties": {"email": {"type": "string"}}},
        }
    ]


async def test_interrupt_elicitation_callback_handles_url_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL-mode acceptance never returns user-supplied content to the server."""
    monkeypatch.setattr(elicitation, "interrupt", lambda _payload: {"action": "accept"})
    mcp_callbacks = Callbacks().to_mcp_format(
        context=CallbackContext(server_name="calendar"),
        elicitation="interrupt",
    )
    assert mcp_callbacks.elicitation_callback is not None

    result = await mcp_callbacks.elicitation_callback(
        SimpleNamespace(),
        SimpleNamespace(
            mode="url",
            message="Connect your calendar account.",
            url="https://calendar.example.com/connect",
        ),
    )

    assert result.action == "accept"
    assert result.content is None


def test_interrupt_elicitation_rejects_invalid_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed resume value cannot be returned as an MCP elicitation result."""
    monkeypatch.setattr(elicitation, "interrupt", lambda _payload: {"action": "accept"})

    with pytest.raises(ToolException, match="requires object content"):
        elicitation.interrupt_for_elicitation(
            SimpleNamespace(
                mode="form",
                message="Who should receive the invitation?",
                requested_schema={"type": "object"},
            ),
            CallbackContext(server_name="calendar"),
        )


async def test_interrupt_elicitation_rejects_nested_form_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Form content uses MCP's flat primitive value subset."""
    monkeypatch.setattr(
        elicitation,
        "interrupt",
        lambda _payload: {"action": "accept", "content": {"details": {"nested": "value"}}},
    )

    with pytest.raises(ToolException, match="unsupported value"):
        elicitation.interrupt_for_elicitation(
            SimpleNamespace(
                mode="form",
                message="Provide details.",
                requested_schema={"type": "object"},
            ),
            CallbackContext(server_name="profile"),
        )


async def test_existing_session_rejects_interrupt_elicitation() -> None:
    """An existing SDK session cannot be retrofitted with an interrupt callback."""
    with pytest.raises(ValueError, match="connection-backed tool"):
        await load_mcp_tools(object(), elicitation="interrupt")  # type: ignore[arg-type]


def test_custom_and_interrupt_elicitation_handlers_conflict() -> None:
    """A caller must choose one elicitation handling strategy."""

    async def custom_elicitation(*args: object) -> Any:
        return None

    callbacks = Callbacks(on_elicitation=custom_elicitation)
    with pytest.raises(ValueError, match="either `on_elicitation`"):
        callbacks.to_mcp_format(
            context=CallbackContext(server_name="calendar"),
            elicitation="interrupt",
        )
