from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime
from syrupy.assertion import SnapshotAssertion

from langchain.agents import AgentState
from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_create_agent_diagram(
    snapshot: SnapshotAssertion,
) -> None:
    class NoopOne(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopTwo(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopThree(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopFour(AgentMiddleware):
        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopFive(AgentMiddleware):
        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopSix(AgentMiddleware):
        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopSeven(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopEight(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopNine(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopTen(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(request)

        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    class NoopEleven(AgentMiddleware):
        def before_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(request)

        def after_model(self, state: AgentState[Any], runtime: Runtime[None]) -> None:
            pass

    agent_zero = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
    )

    assert agent_zero.get_graph().draw_mermaid() == snapshot

    agent_one = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne()],
    )

    assert agent_one.get_graph().draw_mermaid() == snapshot

    agent_two = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo()],
    )

    assert agent_two.get_graph().draw_mermaid() == snapshot

    agent_three = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopOne(), NoopTwo(), NoopThree()],
    )

    assert agent_three.get_graph().draw_mermaid() == snapshot

    agent_four = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour()],
    )

    assert agent_four.get_graph().draw_mermaid() == snapshot

    agent_five = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive()],
    )

    assert agent_five.get_graph().draw_mermaid() == snapshot

    agent_six = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopFour(), NoopFive(), NoopSix()],
    )

    assert agent_six.get_graph().draw_mermaid() == snapshot

    agent_seven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven()],
    )

    assert agent_seven.get_graph().draw_mermaid() == snapshot

    agent_eight = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight()],
    )

    assert agent_eight.get_graph().draw_mermaid() == snapshot

    agent_nine = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopSeven(), NoopEight(), NoopNine()],
    )

    assert agent_nine.get_graph().draw_mermaid() == snapshot

    agent_ten = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen()],
    )

    assert agent_ten.get_graph().draw_mermaid() == snapshot

    agent_eleven = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt="You are a helpful assistant.",
        middleware=[NoopTen(), NoopEleven()],
    )

    assert agent_eleven.get_graph().draw_mermaid() == snapshot
