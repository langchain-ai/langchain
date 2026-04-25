"""Unit tests for IntentAwareModelRouterMiddleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents.middleware.model_routing import IntentAwareModelRouterMiddleware

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun


def test_routes_short_simple_task_to_fast_model() -> None:
    """Simple extraction requests should use fast model."""
    router = IntentAwareModelRouterMiddleware(
        fast_model=GenericFakeChatModel(messages=iter([AIMessage(content="fast")])),
        balanced_model=GenericFakeChatModel(messages=iter([AIMessage(content="balanced")])),
        quality_model=GenericFakeChatModel(messages=iter([AIMessage(content="quality")])),
    )

    agent = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="primary")])),
        middleware=[router],
    )

    result = agent.invoke({"messages": [HumanMessage("Summarize this short note.")]})

    assert result["messages"][-1].content == "fast"


def test_routes_high_stakes_request_to_quality_model() -> None:
    """High-stakes prompts should use quality model."""
    router = IntentAwareModelRouterMiddleware(
        fast_model=GenericFakeChatModel(messages=iter([AIMessage(content="fast")])),
        balanced_model=GenericFakeChatModel(messages=iter([AIMessage(content="balanced")])),
        quality_model=GenericFakeChatModel(messages=iter([AIMessage(content="quality")])),
    )

    agent = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="primary")])),
        middleware=[router],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Give legal guidance for this contract clause.")]}
    )

    assert result["messages"][-1].content == "quality"


def test_routes_default_to_balanced_model() -> None:
    """Requests that are not simple/high-stakes should use balanced model."""
    router = IntentAwareModelRouterMiddleware(
        fast_model=GenericFakeChatModel(messages=iter([AIMessage(content="fast")])),
        balanced_model=GenericFakeChatModel(messages=iter([AIMessage(content="balanced")])),
        quality_model=GenericFakeChatModel(messages=iter([AIMessage(content="quality")])),
    )

    agent = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="primary")])),
        middleware=[router],
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Propose three architecture tradeoffs for this service.")]}
    )

    assert result["messages"][-1].content == "balanced"


def test_routing_middleware_adds_trace_metadata() -> None:
    """Routing decisions should be exposed in response metadata."""
    router = IntentAwareModelRouterMiddleware(
        fast_model=GenericFakeChatModel(messages=iter([AIMessage(content="fast")])),
        balanced_model=GenericFakeChatModel(messages=iter([AIMessage(content="balanced")])),
        quality_model=GenericFakeChatModel(messages=iter([AIMessage(content="quality")])),
    )

    agent = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="primary")])),
        middleware=[router],
    )

    result = agent.invoke({"messages": [HumanMessage("Extract key fields from this receipt.")]})
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    routing_meta = last.response_metadata.get("routing")
    assert isinstance(routing_meta, dict)
    assert routing_meta.get("policy") == "intent_aware_router"
    assert routing_meta.get("route_reason") == "simple_short_task"


def test_routing_and_fallback_middleware_compose() -> None:
    """Selected routed model can fail and fallback middleware recovers."""

    class AlwaysFailModel(BaseChatModel):
        """Model that always errors."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "routed model failed"
            raise ValueError(msg)

        @property
        def _llm_type(self) -> str:
            return "always_fail"

    router = IntentAwareModelRouterMiddleware(
        fast_model=AlwaysFailModel(),
        balanced_model=GenericFakeChatModel(messages=iter([AIMessage(content="balanced")])),
        quality_model=GenericFakeChatModel(messages=iter([AIMessage(content="quality")])),
    )
    fallback = ModelFallbackMiddleware(
        GenericFakeChatModel(messages=iter([AIMessage(content="fallback-success")]))
    )
    agent = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="primary")])),
        middleware=[router, fallback],
    )

    result = agent.invoke({"messages": [HumanMessage("Summarize this short note.")]})
    assert result["messages"][-1].content == "fallback-success"

