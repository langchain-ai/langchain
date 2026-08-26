"""Tests for answering MCP elicitation requests with a LangGraph interrupt.

This module deliberately omits `from __future__ import annotations`: the MCP SDK
evaluates a tool's annotations to discover its `Resolve` markers, and stringized
annotations declared inside a function cannot be resolved.
"""

import json
from typing import Annotated, Any

import pytest
from fastmcp import Context, FastMCP
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.server.mcpserver import Elicit, MCPServer, Resolve
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolResult,
    CreateMessageRequest,
    CreateMessageRequestParams,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    SamplingMessage,
    TextContent,
)
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain.mcp.elicitation import _call_tool_with_interrupts
from tests.unit_tests.agents.model import FakeToolCallingModel


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


def _agent(tools: list[Any]) -> Any:
    return create_agent(
        FakeToolCallingModel(tool_calls=[[{"name": "book_table", "args": {}, "id": "c1"}], []]),
        tools,
        checkpointer=InMemorySaver(),
    )


@pytest.mark.asyncio
async def test_interrupt_carries_the_question_and_resume_completes_the_call() -> None:
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls), elicitation="interrupt").get_tools()
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
    tools = await MCPAdapter(_restaurant_server(calls), elicitation="interrupt").get_tools()
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
    tools = await MCPAdapter(_restaurant_server(calls), elicitation="interrupt").get_tools()
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
async def test_without_the_opt_in_the_capability_is_never_declared() -> None:
    """The default stays non-interactive, and a server needing input says so.

    Elicitation is only offered to a client that advertises the capability, and
    only `elicitation='interrupt'` makes the adapter advertise it. A server whose
    tool *requires* an answer therefore refuses the call outright rather than
    silently running without one.
    """
    calls = {"resolver": 0, "body": 0}
    tools = await MCPAdapter(_restaurant_server(calls)).get_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    with pytest.raises(MCPError, match="did not declare the form elicitation capability"):
        await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)

    assert calls["body"] == 0


class _FakeSession:
    """A session that returns a scripted sequence of results."""

    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append({"name": name, "arguments": arguments, **kwargs})
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


class _FakeClient:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session


def _elicit_round(key: str = "ask") -> InputRequiredResult:
    return InputRequiredResult(
        input_requests={
            key: ElicitRequest(
                method="elicitation/create",
                params=ElicitRequestFormParams(
                    mode="form",
                    message="How many?",
                    requestedSchema={"type": "object", "properties": {}},
                ),
            )
        },
        request_state="state-1",
    )


@pytest.mark.asyncio
async def test_sampling_requests_are_rejected_rather_than_mishandled() -> None:
    """Driving the loop by hand bypasses the callbacks that answer sampling."""
    sampling = InputRequiredResult(
        input_requests={
            "sample": CreateMessageRequest(
                method="sampling/createMessage",
                params=CreateMessageRequestParams(
                    messages=[
                        SamplingMessage(role="user", content=TextContent(type="text", text="hi"))
                    ],
                    maxTokens=16,
                ),
            )
        },
        request_state="state-1",
    )
    client = _FakeClient(_FakeSession([sampling]))

    with pytest.raises(NotImplementedError, match="sampling/createMessage"):
        await _call_tool_with_interrupts(client, "summarize", {})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_state_only_round_is_retried_without_asking_anyone() -> None:
    """A server still working sends state with no requests; back off, do not interrupt."""
    working = InputRequiredResult(input_requests=None, request_state="state-1")
    done = CallToolResult(content=[TextContent(type="text", text="done")])
    session = _FakeSession([working, done])
    client = _FakeClient(_FakeSession([working, done]))
    client.session = session

    result = await _call_tool_with_interrupts(client, "slow", {})  # type: ignore[arg-type]

    assert result is done
    assert session.calls[1]["request_state"] == "state-1"
    assert session.calls[1]["input_responses"] is None


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
            if answer.action == "accept" and answer.content:
                known[key] = answer.content["answer"]

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
    tools = await MCPAdapter(_multi_question_server(calls), elicitation="interrupt").get_tools()
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
    tools = await MCPAdapter(_multi_question_server(calls), elicitation="interrupt").get_tools()
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
    tools = await MCPAdapter(_multi_question_server(calls), elicitation="interrupt").get_tools()
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
    tools = await MCPAdapter(_restaurant_server(calls), elicitation="interrupt").get_tools()
    agent = _agent(tools)
    config: Any = {"configurable": {"thread_id": "t"}}

    paused = await agent.ainvoke({"messages": [{"role": "user", "content": "book"}]}, config)
    [pause] = paused["__interrupt__"]
    [question] = pause.value["requests"]

    bogus = {"responses": {question["key"]: {"action": "maybe"}}}
    with pytest.raises(ValueError, match="expected 'accept', 'decline', or 'cancel'"):
        await agent.ainvoke(Command(resume=bogus), config)
