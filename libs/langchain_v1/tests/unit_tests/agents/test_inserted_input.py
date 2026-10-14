"""Input inserted into a running agent lands before its next model call."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import RunControl, Runtime

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_inserted_input_lands_after_every_tool_result() -> None:
    control = RunControl()

    @tool
    def first(x: int) -> str:
        """First tool."""
        assert x == 1
        control.submit_input("insert-1", {"messages": [HumanMessage("also check B")]})
        return "first done"

    @tool
    def second(x: int) -> str:
        """Second tool."""
        assert x == 2
        return "second done"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 1}, "id": "call_1", "name": "first"},
                    {"args": {"x": 2}, "id": "call_2", "name": "second"},
                ],
                [],
            ]
        ),
        tools=[first, second],
    )

    result = agent.invoke({"messages": [HumanMessage("check A")]}, control=control)

    kinds = [type(message) for message in result["messages"]]
    assert kinds == [HumanMessage, AIMessage, ToolMessage, ToolMessage, HumanMessage, AIMessage]
    assert result["messages"][4].content == "also check B"
    assert not control.has_inputs


def test_agent_without_input_runs_as_before() -> None:
    agent = create_agent(model=FakeToolCallingModel(), tools=[])

    result = agent.invoke({"messages": [HumanMessage("hi")]}, control=RunControl())

    assert [type(message) for message in result["messages"]] == [HumanMessage, AIMessage]


def test_inserted_input_reaches_before_model_middleware() -> None:
    control = RunControl()
    seen: list[list[str]] = []

    @tool
    def work(x: int) -> str:
        """Do work."""
        assert x == 1
        control.submit_input("insert-1", {"messages": [HumanMessage("also check B")]})
        return "done"

    @before_model
    def record(state: AgentState, runtime: Runtime) -> None:  # noqa: ARG001  # hook signature
        seen.append([str(m.content) for m in state["messages"] if m.type == "human"])

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "work"}], []]
        ),
        tools=[work],
        middleware=[record],
    )

    agent.invoke({"messages": [HumanMessage("check A")]}, control=control)

    # the second model call's middleware already sees the inserted message
    assert seen == [["check A"], ["check A", "also check B"]]
