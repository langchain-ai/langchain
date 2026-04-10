import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pytest_benchmark.fixture import BenchmarkFixture

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)


@pytest.mark.benchmark
def test_create_agent_instantiation(benchmark: BenchmarkFixture) -> None:
    def instantiate_agent() -> None:
        create_agent(model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])))

    benchmark(instantiate_agent)


@pytest.mark.benchmark
def test_create_agent_instantiation_with_middleware(
    benchmark: BenchmarkFixture,
) -> None:
    def instantiate_agent() -> None:
        create_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
            middleware=[
                TodoListMiddleware(),
                ToolRetryMiddleware(),
                ModelRetryMiddleware(),
            ],
        )

    benchmark(instantiate_agent)
