"""Live integration tests for `ModelRouterMiddleware`."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)


def _middleware(
    fast_model: GenericFakeChatModel,
    powerful_model: GenericFakeChatModel,
) -> ModelRouterMiddleware:
    """Create a router with criteria that make the expected route explicit."""
    return ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=fast_model,
                criteria="The request contains the exact marker `ROUTE: fast`.",
            ),
            "powerful": ModelChoice(
                model=powerful_model,
                criteria="The request contains the exact marker `ROUTE: powerful`.",
            ),
        },
        instructions=(
            "Select the route named by the exact `ROUTE: <name>` marker in the "
            "request. Do not infer a different route."
        ),
    )


@pytest.mark.parametrize("route", ["fast", "powerful"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_model_router_live_classification(
    route: str,
    *,
    asynchronous: bool,
) -> None:
    """Route both criteria through live synchronous and asynchronous paths."""
    fast_model = GenericFakeChatModel(messages=iter([AIMessage("fast model")]))
    powerful_model = GenericFakeChatModel(messages=iter([AIMessage("powerful model")]))
    middleware = _middleware(fast_model, powerful_model)
    agent = create_agent(fast_model, middleware=[middleware])
    task = HumanMessage(f"ROUTE: {route}. Follow the explicitly marked route.")

    try:
        if asynchronous:
            result = await agent.ainvoke({"messages": [task]})
        else:
            result = agent.invoke({"messages": [task]})
        assert result["messages"][-1].text == f"{route} model"
    finally:
        if middleware.classifier.async_client is not None:
            await middleware.classifier.async_client.aclose()
        if middleware.classifier.client is not None:
            middleware.classifier.client.close()
