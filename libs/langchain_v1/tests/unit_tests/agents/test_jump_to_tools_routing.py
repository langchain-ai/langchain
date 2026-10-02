"""`jump_to="tools"` must dispatch one task per pending tool call."""

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def _last_ai_has_tool_calls(state: AgentState[Any]) -> bool:
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


class JumpToTools(AgentMiddleware):
    """Jump straight to tools whenever the model asked for any."""

    @hook_config(can_jump_to=["tools"])
    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        return {"jump_to": "tools"} if _last_ai_has_tool_calls(state) else None


class NoopAfterModel(AgentMiddleware):
    """A second `after_model` hook, so the jump takes the middleware-to-middleware edge."""

    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> None:
        return None


def _middleware(jumper: AgentMiddleware, *, chained: bool) -> list[AgentMiddleware]:
    # `after_model` hooks run last to first, so `jumper` runs first when chained.
    return [NoopAfterModel(), jumper] if chained else [jumper]


chained_or_not = pytest.mark.parametrize(
    "chained", [False, True], ids=["last_after_model", "chained_after_model"]
)


@chained_or_not
def test_jump_to_tools_gives_each_tool_call_its_own_interrupt(*, chained: bool) -> None:
    @tool
    def ask_a() -> str:
        """Ask A."""
        return str(interrupt({"tool": "ask_a"}))

    @tool
    def ask_b() -> str:
        """Ask B."""
        return str(interrupt({"tool": "ask_b"}))

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="ask_a", args={}, id="a"),
                ToolCall(name="ask_b", args={}, id="b"),
            ],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[ask_a, ask_b],
        middleware=_middleware(JumpToTools(), chained=chained),
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "jump-distinct"}}

    result = agent.invoke({"messages": [HumanMessage("go")]}, config)

    interrupts = result["__interrupt__"]
    assert sorted(i.value["tool"] for i in interrupts) == ["ask_a", "ask_b"]
    assert len({i.id for i in interrupts}) == 2

    by_tool = {i.value["tool"]: i.id for i in interrupts}
    final = agent.invoke(Command(resume={by_tool["ask_a"]: "A", by_tool["ask_b"]: "B"}), config)
    contents = {m.tool_call_id: m.content for m in final["messages"] if isinstance(m, ToolMessage)}
    assert contents == {"a": "A", "b": "B"}


@chained_or_not
def test_jump_to_tools_skips_calls_that_already_have_a_tool_message(*, chained: bool) -> None:
    ran: list[str] = []

    @tool
    def tool_a() -> str:
        """A."""
        ran.append("a")
        return "a"

    @tool
    def tool_b() -> str:
        """B."""
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
            [
                ToolCall(name="tool_a", args={}, id="a"),
                ToolCall(name="tool_b", args={}, id="b"),
            ],
            [],
        ]
    )
    agent = create_agent(
        model=model,
        tools=[tool_a, tool_b],
        middleware=_middleware(AnswerAThenJump(), chained=chained),
    )

    agent.invoke({"messages": [HumanMessage("go")]})

    assert ran == ["b"]


@chained_or_not
def test_jump_to_tools_with_nothing_pending_goes_back_to_the_model(*, chained: bool) -> None:
    @tool
    def tool_a() -> str:
        """A."""
        msg = "tool_a must not run"
        raise AssertionError(msg)

    class AnswerAllThenJump(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            if not _last_ai_has_tool_calls(state):
                return None
            answer = ToolMessage("done", tool_call_id="a", name="tool_a")
            return {"messages": [answer], "jump_to": "tools"}

    model = FakeToolCallingModel(tool_calls=[[ToolCall(name="tool_a", args={}, id="a")], []])
    agent = create_agent(
        model=model,
        tools=[tool_a],
        middleware=_middleware(AnswerAllThenJump(), chained=chained),
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls


def test_jump_to_tools_from_before_model_with_nothing_pending_runs_the_model() -> None:
    @tool
    def tool_a() -> str:
        """A."""
        return "a"

    jumped: list[bool] = []

    class JumpToToolsOnce(AgentMiddleware):
        @hook_config(can_jump_to=["tools"])
        @override
        def before_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
            if jumped:
                return None
            jumped.append(True)
            return {"jump_to": "tools"}

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]), tools=[tool_a], middleware=[JumpToToolsOnce()]
    )

    result = agent.invoke({"messages": [HumanMessage("go")]})

    assert jumped == [True]
    assert isinstance(result["messages"][-1], AIMessage)
