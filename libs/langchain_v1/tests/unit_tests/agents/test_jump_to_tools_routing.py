"""Routing tests for middleware that returns `jump_to="tools"`.

A jump to `tools` must dispatch exactly what the default model-to-tools edge dispatches:
one task per tool call on the last `AIMessage` that has no `ToolMessage` yet, and nothing
at all when every call is answered, in which case the edge's default routing applies.
Before this was enforced, the jump sent the whole state to the tools node, which re-ran
every tool call on the message, including calls that were already answered, in a single
task.

Regression coverage for https://github.com/langchain-ai/langchain/issues/40492.
"""

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import AgentState
from langchain.agents.structured_output import ToolStrategy
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _last_ai_has_tool_calls(state: AgentState[Any]) -> bool:
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    return last_ai is not None and bool(last_ai.tool_calls)


class JumpToToolsMiddleware(AgentMiddleware):
    """Jump straight to the tools node whenever the model asked for tools."""

    @hook_config(can_jump_to=["tools"])
    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        return {"jump_to": "tools"} if _last_ai_has_tool_calls(state) else None


class NoopAfterModelMiddleware(AgentMiddleware):
    """Another `after_model` hook, so a preceding jump takes the middleware chain edge."""

    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
        return None


def _with_jumper(jumper: AgentMiddleware, *, chained: bool) -> list[AgentMiddleware]:
    """Place `jumper` so its jump leaves from the loop exit edge or from a chain edge.

    `after_model` hooks run in reverse middleware order. With `chained=True` the jumper
    runs first and its node is wired through the middleware-to-middleware edge; with
    `chained=False` it is the only hook, so its node is the loop exit and its jump is
    resolved by the model-to-tools edge instead. Both edges must behave the same.
    """
    return [NoopAfterModelMiddleware(), jumper] if chained else [jumper]


jump_edge_positions = pytest.mark.parametrize(
    "chained", [False, True], ids=["loop_exit_edge", "middleware_chain_edge"]
)


@jump_edge_positions
def test_jump_to_tools_runs_each_pending_call_as_its_own_task(*, chained: bool) -> None:
    """Each pending tool call gets its own task, so interrupts raised inside tools stay apart."""

    @tool
    def ask_a() -> str:
        """Ask the human about A."""
        return str(interrupt({"tool": "ask_a"}))

    @tool
    def ask_b() -> str:
        """Ask the human about B."""
        return str(interrupt({"tool": "ask_b"}))

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="ask_a", args={}, id="a"), ToolCall(name="ask_b", args={}, id="b")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[ask_a, ask_b],
        middleware=_with_jumper(JumpToToolsMiddleware(), chained=chained),
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "one-task-per-call"}}

    interrupted = agent.invoke({"messages": [HumanMessage("go")]}, config)

    interrupts = interrupted["__interrupt__"]
    assert sorted(i.value["tool"] for i in interrupts) == ["ask_a", "ask_b"]
    assert len({i.id for i in interrupts}) == 2

    id_by_tool = {i.value["tool"]: i.id for i in interrupts}
    final = agent.invoke(
        Command(resume={id_by_tool["ask_a"]: "answer A", id_by_tool["ask_b"]: "answer B"}),
        config,
    )

    assert "__interrupt__" not in final
    results = {m.tool_call_id: m.content for m in final["messages"] if isinstance(m, ToolMessage)}
    assert results == {"a": "answer A", "b": "answer B"}


@jump_edge_positions
def test_jump_to_tools_skips_calls_that_already_have_a_tool_message(*, chained: bool) -> None:
    """A call that was answered before the jump must not run again."""
    ran: list[str] = []

    @tool
    def tool_a() -> str:
        """Tool A."""
        ran.append("a")
        return "a"

    @tool
    def tool_b() -> str:
        """Tool B."""
        ran.append("b")
        return "b"

    class AnswerAThenJump(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            if not _last_ai_has_tool_calls(state):
                return None
            answer = ToolMessage("answered by middleware", tool_call_id="a", name="tool_a")
            return {"messages": [answer], "jump_to": "tools"}

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="tool_a", args={}, id="a"), ToolCall(name="tool_b", args={}, id="b")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[tool_a, tool_b],
        middleware=_with_jumper(AnswerAThenJump(), chained=chained),
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert ran == ["b"]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["a", "b"]
    assert tool_messages[0].content == "answered by middleware"


@jump_edge_positions
def test_jump_to_tools_with_nothing_pending_continues_to_the_model(*, chained: bool) -> None:
    """With every call answered, the jump is a no-op and the loop continues to the model."""
    ran: list[str] = []

    @tool
    def tool_a() -> str:
        """Tool A."""
        ran.append("a")
        return "a"

    class AnswerAllThenJump(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            if not _last_ai_has_tool_calls(state):
                return None
            answer = ToolMessage("answered by middleware", tool_call_id="a", name="tool_a")
            return {"messages": [answer], "jump_to": "tools"}

    model = FakeToolCallingModel(tool_calls=[[ToolCall(name="tool_a", args={}, id="a")], []])
    agent = create_agent(
        model=model,
        tools=[tool_a],
        middleware=_with_jumper(AnswerAllThenJump(), chained=chained),
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert ran == []
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    assert not ai_messages[-1].tool_calls


@jump_edge_positions
def test_jump_to_tools_exits_once_a_structured_response_exists(*, chained: bool) -> None:
    """A structured output tool call is never dispatched, and the loop exits with the response."""

    class Weather(BaseModel):
        temperature: float
        condition: str

    @tool
    def get_weather() -> str:
        """Get the weather."""
        msg = "get_weather must not run"
        raise AssertionError(msg)

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="Weather", args={"temperature": 72.0, "condition": "sunny"}, id="s")]
        ]
    )
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=_with_jumper(JumpToToolsMiddleware(), chained=chained),
        response_format=ToolStrategy(Weather),
    )

    # A regression that loops back to the model would otherwise run until the agent's
    # default recursion limit of 9_999.
    result = agent.invoke({"messages": [HumanMessage("weather?")]}, {"recursion_limit": 10})

    assert result["structured_response"] == Weather(temperature=72.0, condition="sunny")
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 1
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["s"]
    assert tool_messages[0].status != "error"


def test_jump_to_tools_without_a_tools_node_exits_with_the_structured_response() -> None:
    """With no tools node, a jump is a no-op and the structured response ends the run.

    An agent with structured output but no tools routes the loop exit through the
    model-to-model edge. Before the fix a jump there resolved to the undeclared `tools`
    destination and raised `KeyError`.
    """

    class Weather(BaseModel):
        temperature: float
        condition: str

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="Weather", args={"temperature": 72.0, "condition": "sunny"}, id="s")]
        ]
    )
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[JumpToToolsMiddleware()],
        response_format=ToolStrategy(Weather),
    )

    result = agent.invoke({"messages": [HumanMessage("weather?")]}, {"recursion_limit": 10})

    assert result["structured_response"] == Weather(temperature=72.0, condition="sunny")
    assert len([m for m in result["messages"] if isinstance(m, AIMessage)]) == 1


def test_jump_to_tools_from_before_model_with_nothing_pending_runs_the_model() -> None:
    """A jump before any `AIMessage` exists has nothing to dispatch and continues to the model.

    The jump is a no-op, so the edge takes its default destination and the hook does not
    run again. Before the fix the tools node raised `No AIMessage found in input`.
    """
    ran: list[str] = []
    hook_runs: list[int] = []

    @tool
    def tool_a() -> str:
        """Tool A."""
        ran.append("a")
        return "a"

    class JumpToToolsOnce(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            hook_runs.append(len(state["messages"]))
            return {"jump_to": "tools"} if len(hook_runs) == 1 else None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]), tools=[tool_a], middleware=[JumpToToolsOnce()]
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert hook_runs == [1]
    assert ran == []
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]


def test_jump_to_tools_from_before_model_on_a_later_turn_still_answers_the_new_message() -> None:
    """A no-op jump on a follow-up turn must not end the run on the previous turn's response.

    `structured_response` persists in a checkpointed thread until the model runs again, so a
    jump that consulted it before the model ran would exit without answering the new message.
    """

    class Weather(BaseModel):
        temperature: float
        condition: str

    @tool
    def get_weather() -> str:
        """Get the weather."""
        msg = "get_weather must not run"
        raise AssertionError(msg)

    class JumpToToolsEachTurn(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            return {"jump_to": "tools"} if isinstance(state["messages"][-1], HumanMessage) else None

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="Weather", args={"temperature": 1.0, "condition": "turn 1"}, id="s1")],
            [ToolCall(name="Weather", args={"temperature": 2.0, "condition": "turn 2"}, id="s2")],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[JumpToToolsEachTurn()],
        response_format=ToolStrategy(Weather),
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "later-turn"}}

    first = agent.invoke({"messages": [HumanMessage("turn 1")]}, config)
    second = agent.invoke({"messages": [HumanMessage("turn 2")]}, config)

    assert first["structured_response"] == Weather(temperature=1.0, condition="turn 1")
    assert second["structured_response"] == Weather(temperature=2.0, condition="turn 2")
    assert len([m for m in second["messages"] if isinstance(m, AIMessage)]) == 2


@jump_edge_positions
def test_jump_to_tools_after_a_final_answer_ends_the_run(*, chained: bool) -> None:
    """A jump when the model called no tools is a no-op, so the run ends instead of looping."""

    @tool
    def tool_a() -> str:
        """Tool A."""
        msg = "tool_a must not run"
        raise AssertionError(msg)

    class AlwaysJumpToTools(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any]:
            return {"jump_to": "tools"}

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[tool_a],
        middleware=_with_jumper(AlwaysJumpToTools(), chained=chained),
    )

    result = agent.invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 10})

    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]


async def test_jump_to_tools_runs_each_pending_call_as_its_own_task_async() -> None:
    """The async path also gives each pending call its own task and interrupt."""

    @tool
    def ask_a() -> str:
        """Ask the human about A."""
        return str(interrupt({"tool": "ask_a"}))

    @tool
    def ask_b() -> str:
        """Ask the human about B."""
        return str(interrupt({"tool": "ask_b"}))

    class JumpToToolsAsync(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        async def aafter_model(
            self, state: AgentState[Any], runtime: Runtime
        ) -> dict[str, Any] | None:
            return {"jump_to": "tools"} if _last_ai_has_tool_calls(state) else None

    model = FakeToolCallingModel(
        tool_calls=[
            [ToolCall(name="ask_a", args={}, id="a"), ToolCall(name="ask_b", args={}, id="b")],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[ask_a, ask_b],
        middleware=[JumpToToolsAsync()],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "one-task-per-call-async"}}

    interrupted = await agent.ainvoke({"messages": [HumanMessage("go")]}, config)

    interrupts = interrupted["__interrupt__"]
    assert sorted(i.value["tool"] for i in interrupts) == ["ask_a", "ask_b"]
    assert len({i.id for i in interrupts}) == 2

    id_by_tool = {i.value["tool"]: i.id for i in interrupts}
    final = await agent.ainvoke(
        Command(resume={id_by_tool["ask_a"]: "answer A", id_by_tool["ask_b"]: "answer B"}),
        config,
    )

    assert "__interrupt__" not in final
    results = {m.tool_call_id: m.content for m in final["messages"] if isinstance(m, ToolMessage)}
    assert results == {"a": "answer A", "b": "answer B"}
