"""Live integration tests for `OpenAIModelRouterMiddleware`."""

from __future__ import annotations

import itertools

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai.middleware import ModelChoice, OpenAIModelRouterMiddleware

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.LangChainBetaWarning"
)


def _fake(name: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(name)]))


@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize(
    ("task", "expected_route"),
    [
        ("What is 2 + 2?", "fast"),
        (
            "Design a Byzantine fault tolerant consensus protocol and prove its "
            "safety and liveness properties.",
            "powerful",
        ),
    ],
)
async def test_live_routing(task: str, expected_route: str, *, async_: bool) -> None:
    router = OpenAIModelRouterMiddleware(
        choices={
            "fast": ModelChoice(model=_fake("fast"), criteria="Simple, quick tasks."),
            "powerful": ModelChoice(
                model=_fake("powerful"),
                criteria="Complex tasks requiring deep reasoning.",
            ),
        },
        instructions="Choose the least costly model suited to the task.",
        model="gpt-6-luna",
    )
    agent = create_agent(_fake("default"), middleware=[router])
    state = InputAgentState(messages=[HumanMessage(task)])

    result = await agent.ainvoke(state) if async_ else agent.invoke(state)

    assert result["model_route"].choice == expected_route
    assert result["messages"][-1].content == expected_route
