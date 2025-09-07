from typing import Any
from langchain_core.messages import AIMessage, ToolCall
from syrupy import SnapshotAssertion

from langchain_core.tools import tool
from langchain.agents.middleware_agent import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from .model import FakeToolCallingModel
from .messages import _AnyIdToolMessage, _AnyIdHumanMessage
from .model import FakeToolCallingModel


def test_create_agent_diagram(
    snapshot: SnapshotAssertion,
):
    class NoopOne(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopTwo(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopThree(AgentMiddleware):
        def before_model(self, state):
            pass

    class NoopFour(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopFive(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopSix(AgentMiddleware):
        def after_model(self, state):
            pass

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopEight(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopNine(AgentMiddleware):
        def before_model(self, state):
            pass

        def after_model(self, state):
            pass

    class NoopTen(AgentMiddleware):
        def before_model(self, state):
            pass

        def modify_model_request(self, request, state):
            pass

        def after_model(self, state):
            pass

    class NoopEleven(AgentMiddleware):
        def before_model(self, state):
            pass

        def modify_model_request(self, request, state):
            pass

        def after_model(self, state):
            pass

    agent_zero = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_zero.compile().get_graph().draw_mermaid() == snapshot

    agent_one = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne()],
    )

    assert agent_one.compile().get_graph().draw_mermaid() == snapshot

    agent_two = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo()],
    )

    assert agent_two.compile().get_graph().draw_mermaid() == snapshot

    agent_three = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo(), NoopThree()],
    )

    assert agent_three.compile().get_graph().draw_mermaid() == snapshot

    agent_four = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour()],
    )

    assert agent_four.compile().get_graph().draw_mermaid() == snapshot

    agent_five = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive()],
    )

    assert agent_five.compile().get_graph().draw_mermaid() == snapshot

    agent_six = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive(), NoopSix()],
    )

    assert agent_six.compile().get_graph().draw_mermaid() == snapshot

    agent_seven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven()],
    )

    assert agent_seven.compile().get_graph().draw_mermaid() == snapshot

    agent_eight = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    )

    assert agent_eight.compile().get_graph().draw_mermaid() == snapshot

    agent_nine = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight(), NoopNine()],
    )

    assert agent_nine.compile().get_graph().draw_mermaid() == snapshot

    agent_ten = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen()],
    )

    assert agent_ten.compile().get_graph().draw_mermaid() == snapshot

    agent_eleven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen(), NoopEleven()],
    )

    assert agent_eleven.compile().get_graph().draw_mermaid() == snapshot


def test_create_agent_invoke(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopSeven.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopSeven.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopEight.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopEight.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"input": "yo"}, "id": "1", "name": "my_tool"},
                ],
                [],
            ]
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    ).compile(checkpointer=sync_checkpointer)

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": ["hello"]}, thread1) == {
        "messages": [
            _AnyIdHumanMessage(content="hello"),
            AIMessage(
                content="You are a helpful assistant.-hello",
                additional_kwargs={},
                response_metadata={},
                id="0",
                tool_calls=[
                    {
                        "name": "my_tool",
                        "args": {"input": "yo"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            _AnyIdToolMessage(content="YO", name="my_tool", tool_call_id="1"),
            AIMessage(
                content="You are a helpful assistant.-hello-You are a helpful assistant.-hello-YO",
                additional_kwargs={},
                response_metadata={},
                id="1",
            ),
        ],
    }
    assert calls == [
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.modify_model_request",
        "NoopEight.modify_model_request",
        "NoopEight.after_model",
        "NoopSeven.after_model",
        "my_tool",
        "NoopSeven.before_model",
        "NoopEight.before_model",
        "NoopSeven.modify_model_request",
        "NoopEight.modify_model_request",
        "NoopEight.after_model",
        "NoopSeven.after_model",
    ]


def test_create_agent_jump(
    snapshot: SnapshotAssertion,
    sync_checkpointer: BaseCheckpointSaver,
):
    calls = []

    class NoopSeven(AgentMiddleware):
        def before_model(self, state):
            calls.append("NoopSeven.before_model")

        def modify_model_request(self, request, state):
            calls.append("NoopSeven.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopSeven.after_model")

    class NoopEight(AgentMiddleware):
        def before_model(self, state) -> dict[str, Any]:
            calls.append("NoopEight.before_model")
            return {"jump_to": END}

        def modify_model_request(self, request, state) -> ModelRequest:
            calls.append("NoopEight.modify_model_request")
            return request

        def after_model(self, state):
            calls.append("NoopEight.after_model")

    @tool
    def my_tool(input: str) -> str:
        """A great tool"""
        calls.append("my_tool")
        return input.upper()

    agent_one = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[ToolCall(id="1", name="my_tool", args={"input": "yo"})]],
        ),
        tools=[my_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    ).compile(checkpointer=sync_checkpointer)

    if isinstance(sync_checkpointer, InMemorySaver):
        assert agent_one.get_graph().draw_mermaid() == snapshot

    thread1 = {"configurable": {"thread_id": "1"}}
    assert agent_one.invoke({"messages": []}, thread1) == {"messages": []}
    assert calls == ["NoopSeven.before_model", "NoopEight.before_model"]
