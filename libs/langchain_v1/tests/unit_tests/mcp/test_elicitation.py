"""Tests for answering MCP elicitation requests with a LangGraph interrupt.

This module deliberately omits `from __future__ import annotations`: the MCP SDK
evaluates a tool's annotations to discover its `Resolve` markers, and stringized
annotations declared inside a function cannot be resolved.
"""

import json
from typing import Annotated, Any, cast

import pytest
from fastmcp import Client, Context, FastMCP
from fastmcp.client.group import ClientGroup
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.server.mcpserver import Elicit, MCPServer, Resolve
from mcp.types import (
    CallToolResult,
    CreateMessageRequest,
    CreateMessageRequestParams,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    SamplingMessage,
    TextContent,
)
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain.mcp.elicitation import (
    _arm_for_interrupts,
    _call_tool_with_interrupts,
    _drives_interrupts,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

_HANDSHAKE_ERA = "2025-11-25"
"""Latest protocol version that negotiates with the legacy `initialize` handshake."""

_MODERN_ERA = "2026-07-28"
"""Modern protocol version that carries the `InputRequiredResult` elicitation path."""


class PartySize(BaseModel):
    """The data the server wants filled in."""

    guests: int


def _restaurant_server(calls: dict[str, int]) -> MCPServer:
    """A server whose tool cannot run until a human supplies a party size."""
    server = MCPServer("restaurant")

    def ask_party_size() -> Elicit[PartySize]:
        calls["resolver"] += 1
        return Elicit("How many guests are dining?", PartySize)

    @server.tool()
    def book_table(party: Annotated[PartySize, Resolve(ask_party_size)]) -> str:
        """Book a table."""
        calls["body"] += 1
        return f"Booked a table for {party.guests}."

    return server


def _plain_server(name: str) -> MCPServer:
    """A legacy-era server whose one tool needs no input."""
    server = MCPServer(name)

    @server.tool()
    def whoami() -> str:
        """Report the server name."""
        return name

    return server


def _agent(tools: list[Any]) -> Any:
    return create_agent(
        FakeToolCallingModel(tool_calls=[[{"name": "book_table", "args": {}, "id": "c1"}], []]),
        tools,
        checkpointer=InMemorySaver(),
    )


@pytest.mark.asyncio
async def test_interrupt_carries_the_question_and_resume_completes_the_call() -> None:
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls)).list_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)

    [pause] = paused["__interrupt__"]
    request = pause.value
    assert request["type"] == "mcp_elicitation"
    assert request["tool_name"] == "book_table"
    [question] = request["requests"]
    assert question["message"] == "How many guests are dining?"
    assert question["mode"] == "form"
    assert question["requested_schema"]["required"] == ["guests"]
    # The tool body must not have run while the question was outstanding.
    assert calls["body"] == 0

    resumed = await agent.ainvoke(
        Command(
            resume={"responses": {question["key"]: {"action": "accept", "content": {"guests": 4}}}}
        ),
        config,
    )

    tool_message = next(message for message in resumed["messages"] if message.type == "tool")
    assert tool_message.content[0]["text"] == "Booked a table for 4."
    assert tool_message.status == "success"


@pytest.mark.asyncio
async def test_the_tool_body_runs_once_despite_the_replay() -> None:
    """Resuming re-issues the call, but only the answered round reaches the body."""
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls)).list_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    [question] = pause.value["requests"]

    await agent.ainvoke(
        Command(
            resume={"responses": {question["key"]: {"action": "accept", "content": {"guests": 2}}}}
        ),
        config,
    )

    assert calls["body"] == 1
    # The question is re-declared on each replayed round; that is the cost.
    assert calls["resolver"] > 1


@pytest.mark.asyncio
async def test_declining_leaves_the_tool_unrun() -> None:
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls)).list_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    [question] = pause.value["requests"]

    resumed = await agent.ainvoke(
        Command(resume={"responses": {question["key"]: {"action": "decline"}}}),
        config,
    )

    tool_message = next(message for message in resumed["messages"] if message.type == "tool")
    assert tool_message.status == "error"
    assert calls["body"] == 0


@pytest.mark.asyncio
async def test_elicitation_through_a_group_resolves_the_member_session() -> None:
    """A group namespaces the tool, so the loop must drive the member session.

    The interrupt loop reads the raw `InputRequiredResult` from a client session,
    which a group does not expose directly; it must resolve the namespaced name to
    the member client that serves it. Covers that resolution end to end.
    """
    calls = {"resolver": 0, "body": 0}
    group = ClientGroup({"dining": Client(_restaurant_server(calls))})
    tools = await MCPAdapter(group).list_tools()
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[{"name": "dining_book_table", "args": {}, "id": "c1"}], []]
        ),
        tools,
        checkpointer=InMemorySaver(),
    )
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    assert pause.value["tool_name"] == "dining_book_table"
    [question] = pause.value["requests"]
    assert calls["body"] == 0

    resumed = await agent.ainvoke(
        Command(
            resume={"responses": {question["key"]: {"action": "accept", "content": {"guests": 4}}}}
        ),
        config,
    )

    tool_message = next(message for message in resumed["messages"] if message.type == "tool")
    assert tool_message.content[0]["text"] == "Booked a table for 4."
    assert tool_message.status == "success"


@pytest.mark.asyncio
async def test_interrupt_fires_on_the_modern_member_of_a_mixed_era_group() -> None:
    """A group can mix eras; the interrupt drives only the modern member.

    Elicitation is a modern-era feature, so in a group holding one legacy and
    one modern server the loop must resolve to the modern member and interrupt
    there, while the legacy member coexists untouched. Covers the group +
    mixed-era + interrupt path together.
    """
    calls = {"resolver": 0, "body": 0}
    group = ClientGroup(
        {
            "info": Client(_plain_server("info-server"), mode="legacy"),
            "dining": Client(_restaurant_server(calls), mode="auto"),
        }
    )
    tools = {tool.name: tool for tool in await MCPAdapter(group).list_tools()}
    assert sorted(tools) == ["dining_book_table", "info_whoami"]

    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[{"name": "dining_book_table", "args": {}, "id": "c1"}], []]
        ),
        list(tools.values()),
        checkpointer=InMemorySaver(),
    )
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    assert pause.value["tool_name"] == "dining_book_table"
    [question] = pause.value["requests"]
    assert calls["body"] == 0

    resumed = await agent.ainvoke(
        Command(
            resume={"responses": {question["key"]: {"action": "accept", "content": {"guests": 4}}}}
        ),
        config,
    )

    tool_message = next(message for message in resumed["messages"] if message.type == "tool")
    assert tool_message.content[0]["text"] == "Booked a table for 4."
    assert tool_message.status == "success"


@pytest.mark.asyncio
async def test_the_adapter_declares_the_capability_on_the_wire() -> None:
    """Pin the capability declaration, which rides on a sentinel handler.

    FastMCP declares `elicitation` only when the client's callback differs by
    identity from the SDK's default, so the adapter installs a sentinel purely
    to trip that comparison. Assert the negotiated capability directly: if
    FastMCP ever changes how it decides, servers would quietly stop asking, and
    every other test here would still pass.
    """
    adapter = MCPAdapter(_restaurant_server({"resolver": 0, "body": 0}))

    async with adapter:
        capabilities = cast("Client[Any]", adapter.client).session._build_capabilities("2026-07-28")

    assert capabilities.elicitation is not None
    assert capabilities.elicitation.form is not None


@pytest.mark.asyncio
async def test_prebuilt_client_declares_elicitation_without_mutating_the_original() -> None:
    """The adapter arms a clone of a pre-built client that has no handler."""
    client = Client(_restaurant_server({"resolver": 0, "body": 0}))
    adapter = MCPAdapter(client)

    assert adapter.client is not client
    async with adapter:
        adapter_capabilities = cast("Client[Any]", adapter.client).session._build_capabilities(
            "2026-07-28"
        )
    async with client:
        original_capabilities = client.session._build_capabilities("2026-07-28")

    assert adapter_capabilities.elicitation is not None
    assert original_capabilities.elicitation is None


@pytest.mark.asyncio
async def test_a_prebuilt_clients_own_handler_is_honored_not_overridden() -> None:
    """A caller's own elicitation handler is respected: the client is left as-is.

    The adapter arms only a client that has no handler. One the caller already
    built with a handler answers elicitation its own way, so the adapter uses it
    untouched rather than cloning it or replacing the handler.
    """

    async def own_handler(*_: Any) -> Any:
        return None

    client = Client(_restaurant_server({"resolver": 0, "body": 0}), elicitation_handler=own_handler)
    adapter = MCPAdapter(client)

    assert adapter.client is client


@pytest.mark.asyncio
async def test_a_modern_server_drives_interrupts() -> None:
    """An armed client on a modern-era connection routes through the loop."""
    client = Client(_restaurant_server({"resolver": 0, "body": 0}))
    _arm_for_interrupts(client)
    async with client:
        assert client.protocol_version == _MODERN_ERA
        assert _drives_interrupts(cast("Client[Any]", client))


@pytest.mark.asyncio
async def test_a_legacy_server_does_not_drive_interrupts_despite_arming() -> None:
    """The interrupt loop answers an `InputRequiredResult`, a modern-era feature.

    Even when armed, a legacy-era connection falls back to the plain call.
    """
    server: FastMCP[None] = FastMCP("legacy")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = Client(server, mode="legacy")
    _arm_for_interrupts(client)
    async with client:
        assert client.protocol_version == _HANDSHAKE_ERA
        assert not _drives_interrupts(cast("Client[Any]", client))


class _FakeSession:
    """A session that returns a scripted sequence of results."""

    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append({"name": name, "arguments": arguments, **kwargs})
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


def _unanswerable_server() -> FastMCP[None]:
    """A server whose tools ask for things interrupt-based elicitation refuses."""
    server: FastMCP[None] = FastMCP("unanswerable")

    @server.tool
    async def summarize() -> InputRequiredResult:
        """Ask for sampling, which only FastMCP's own callbacks can answer."""
        return InputRequiredResult(
            input_requests={
                "sample": CreateMessageRequest(
                    method="sampling/createMessage",
                    params=CreateMessageRequestParams(
                        messages=[
                            SamplingMessage(
                                role="user", content=TextContent(type="text", text="hi")
                            )
                        ],
                        maxTokens=16,
                    ),
                )
            },
            request_state="state-1",
        )

    @server.tool
    async def slow() -> InputRequiredResult:
        """Return a continuation round: state to come back with, nothing to ask."""
        return InputRequiredResult(request_state="still-working")

    return server


@pytest.mark.asyncio
async def test_sampling_requests_are_rejected_rather_than_mishandled() -> None:
    """Driving the loop by hand bypasses the callbacks that answer sampling."""
    async with MCPAdapter(_unanswerable_server()) as adapter:
        client = cast("Client[Any]", adapter.client)
        async with client:
            with pytest.raises(NotImplementedError, match="sampling/createMessage"):
                await _call_tool_with_interrupts(client, "summarize", {})


@pytest.mark.asyncio
async def test_a_client_without_the_session_guard_warns() -> None:
    """The guard is FastMCP-private, so losing it must not fail silently.

    Without it a transport failure mid-elicitation hangs instead of raising, and
    a hang is the worst way to discover a renamed helper.
    """
    done = CallToolResult(content=[TextContent(type="text", text="done")])

    class _UnguardedClient:
        """A client from a FastMCP that no longer exposes the guard."""

        def __init__(self, session: _FakeSession) -> None:
            self.session = session

    client = _UnguardedClient(_FakeSession([done]))

    with pytest.warns(RuntimeWarning, match="_await_with_session_monitoring"):
        await _call_tool_with_interrupts(client, "greet", {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_continuation_round_is_refused_rather_than_polled() -> None:
    """A round with state but no questions is long-running work, not elicitation."""
    async with MCPAdapter(_unanswerable_server()) as adapter:
        client = cast("Client[Any]", adapter.client)
        async with client:
            with pytest.raises(NotImplementedError, match="continuation round"):
                await _call_tool_with_interrupts(client, "slow", {})


def _multi_question_server(calls: dict[str, int]) -> FastMCP[None]:
    """A server that asks two things at once, then a third on the next round.

    Written with the guard pattern: the tool inspects `ctx.input_responses` and
    returns an `InputRequiredResult` for whatever it still needs. Each retry
    carries only the answers to *that* round, so remembering earlier answers is
    the server's job — it threads them through the opaque `request_state`, which
    is exactly what that field is for.
    """
    server: FastMCP[None] = FastMCP("survey")

    def _ask(known: dict[str, str], **questions: str) -> InputRequiredResult:
        return InputRequiredResult(
            input_requests={
                key: ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        mode="form",
                        message=message,
                        requestedSchema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                )
                for key, message in questions.items()
            },
            request_state=json.dumps(known, sort_keys=True),
        )

    @server.tool
    async def plan_trip(ctx: Context) -> list[TextContent] | InputRequiredResult:
        """Plan a trip, asking for whatever is still unknown."""
        known: dict[str, str] = json.loads(ctx.request_state) if ctx.request_state else {}
        for key, answer in (ctx.input_responses or {}).items():
            if isinstance(answer, ElicitResult) and answer.action == "accept" and answer.content:
                known[key] = str(answer.content["answer"])

        # Round one: two questions at the same time.
        if "city" not in known or "month" not in known:
            calls["round_1"] += 1
            return _ask(known, city="Which city?", month="Which month?")
        # Round two: a question that only makes sense once round one is in.
        if "hotel" not in known:
            calls["round_2"] += 1
            return _ask(known, hotel="Which hotel?")

        calls["body"] += 1
        return [
            TextContent(
                type="text",
                text=f"{known['city']} in {known['month']}, staying at {known['hotel']}.",
            )
        ]

    return server


def _trip_agent(tools: list[Any]) -> Any:
    return create_agent(
        FakeToolCallingModel(tool_calls=[[{"name": "plan_trip", "args": {}, "id": "c1"}], []]),
        tools,
        checkpointer=InMemorySaver(),
    )


def _accept_all(request: dict[str, Any], answer: str) -> dict[str, Any]:
    """Answer every question in one interrupt, keyed the way the server keyed it."""
    return {
        "responses": {
            question["key"]: {"action": "accept", "content": {"answer": answer}}
            for question in request["requests"]
        }
    }


@pytest.mark.asyncio
async def test_several_requests_in_one_round_share_a_single_interrupt() -> None:
    """Parallel questions arrive together, so one resume answers them all."""
    calls = {"round_1": 0, "round_2": 0, "body": 0}
    tools = await MCPAdapter(_multi_question_server(calls)).list_tools()
    agent = _trip_agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "plan"}]}, config)

    [pause] = paused["__interrupt__"]
    request = pause.value
    assert [question["key"] for question in request["requests"]] == ["city", "month"]
    assert [question["message"] for question in request["requests"]] == [
        "Which city?",
        "Which month?",
    ]


@pytest.mark.asyncio
async def test_sequential_rounds_interrupt_once_each_and_resume_in_order() -> None:
    """Two rounds need two resumes, correlated by request key rather than by id."""
    calls = {"round_1": 0, "round_2": 0, "body": 0}
    tools = await MCPAdapter(_multi_question_server(calls)).list_tools()
    agent = _trip_agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    first = await agent.ainvoke({"messages": [{"role": "user", "content": "plan"}]}, config)
    [pause] = first["__interrupt__"]
    assert sorted(question["key"] for question in pause.value["requests"]) == ["city", "month"]

    second = await agent.ainvoke(Command(resume=_accept_all(pause.value, "Lisbon")), config)

    # Round one's answers are replayed from the scratchpad, so the run reaches
    # round two and stops there — a second interrupt, not a repeat of the first.
    [pause] = second["__interrupt__"]
    assert [question["key"] for question in pause.value["requests"]] == ["hotel"]

    final = await agent.ainvoke(Command(resume=_accept_all(pause.value, "Tivoli")), config)

    tool_message = next(message for message in final["messages"] if message.type == "tool")
    assert tool_message.content[0]["text"] == "Lisbon in Lisbon, staying at Tivoli."
    assert calls["body"] == 1


@pytest.mark.asyncio
async def test_answering_only_some_of_a_round_is_rejected() -> None:
    """A partial answer must fail loudly rather than reach the server."""
    calls = {"round_1": 0, "round_2": 0, "body": 0}
    tools = await MCPAdapter(_multi_question_server(calls)).list_tools()
    agent = _trip_agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "plan"}]}, config)
    assert len(paused["__interrupt__"]) == 1

    partial = {"responses": {"city": {"action": "accept", "content": {"answer": "Lisbon"}}}}
    with pytest.raises(ValueError, match="had none: month"):
        await agent.ainvoke(Command(resume=partial), config)

    assert calls["body"] == 0


@pytest.mark.asyncio
async def test_an_unknown_action_is_rejected() -> None:
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls)).list_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    [question] = pause.value["requests"]

    bogus = {"responses": {question["key"]: {"action": "maybe"}}}
    with pytest.raises(ValueError, match="expected 'accept', 'decline', or 'cancel'"):
        await agent.ainvoke(Command(resume=bogus), config)
