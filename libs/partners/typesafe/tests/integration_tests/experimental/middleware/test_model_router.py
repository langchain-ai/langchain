"""Live integration tests for `ModelRouterMiddleware`."""

from __future__ import annotations

import asyncio

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
        default_route="powerful",
    )


def test_model_router_live_sync_classification() -> None:
    """Route a synchronous agent run using a live TypeSafe classification."""
    fast_model = GenericFakeChatModel(messages=iter([AIMessage("fast model")]))
    powerful_model = GenericFakeChatModel(messages=iter([AIMessage("powerful model")]))
    middleware = _middleware(fast_model, powerful_model)
    agent = create_agent(powerful_model, middleware=[middleware])

    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage("ROUTE: fast. Update one line of documentation.")
                ]
            }
        )
        assert result["messages"][-1].text == "fast model"
    finally:
        if middleware.classifier.client is not None:
            middleware.classifier.client.close()
        if middleware.classifier.async_client is not None:
            asyncio.run(middleware.classifier.async_client.aclose())


async def test_model_router_live_async_classification() -> None:
    """Route an asynchronous agent run using a live TypeSafe classification."""
    fast_model = GenericFakeChatModel(messages=iter([AIMessage("fast model")]))
    powerful_model = GenericFakeChatModel(messages=iter([AIMessage("powerful model")]))
    middleware = _middleware(fast_model, powerful_model)
    agent = create_agent(fast_model, middleware=[middleware])

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "ROUTE: powerful. Design a multi-service migration strategy."
                    )
                ]
            }
        )
        assert result["messages"][-1].text == "powerful model"
    finally:
        if middleware.classifier.async_client is not None:
            await middleware.classifier.async_client.aclose()
        if middleware.classifier.client is not None:
            middleware.classifier.client.close()
