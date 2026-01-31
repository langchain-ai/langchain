"""Integration tests for AsyncAnonymizerMiddleware with create_agent."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent
from langchain.agents.middleware.pii import AsyncAnonymizerMiddleware

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from langchain.agents.middleware.types import _InputAgentState


def _get_model(provider: str) -> Any:
    """Get chat model for the specified provider."""
    if provider == "anthropic":
        return pytest.importorskip("langchain_anthropic").ChatAnthropic(
            model="claude-sonnet-4-5-20250929"
        )
    if provider == "openai":
        return pytest.importorskip("langchain_openai").ChatOpenAI(model="gpt-4o-mini")
    msg = f"Unknown provider: {provider}"
    raise ValueError(msg)


async def simple_anonymizer(content: str) -> str:
    """A simple anonymizer that redacts email-like patterns."""
    return re.sub(r"\S+@\S+\.\S+", "[EMAIL_REDACTED]", content)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.asyncio
async def test_async_anonymizer_with_agent(provider: str) -> None:
    """Test AsyncAnonymizerMiddleware integrated with create_agent."""
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        middleware=[AsyncAnonymizerMiddleware(simple_anonymizer)],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Please help me contact john.doe@example.com")]}
    )

    messages = result["messages"]
    first_message_content = str(messages[0].content)

    assert first_message_content == "Please help me contact [EMAIL_REDACTED]"


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.asyncio
async def test_async_anonymizer_preserves_conversation_flow(provider: str) -> None:
    """Test that anonymization doesn't break conversation flow."""
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        middleware=[AsyncAnonymizerMiddleware(simple_anonymizer)],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage("What is 2 + 2? (FYI my email is test@test.com)")]}
    )

    messages = result["messages"]

    assert str(messages[0].content) == "What is 2 + 2? (FYI my email is [EMAIL_REDACTED])"

    assert len(messages) >= 2


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.asyncio
async def test_async_anonymizer_with_message_type_filter(provider: str) -> None:
    """Test AsyncAnonymizerMiddleware with message type filtering."""
    agent: CompiledStateGraph[Any, Any, _InputAgentState, Any] = create_agent(
        model=_get_model(provider),
        middleware=[AsyncAnonymizerMiddleware(simple_anonymizer, message_types=["human"])],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Contact me at user@domain.com")]})

    messages = result["messages"]

    human_msg = messages[0]
    assert str(human_msg.content) == "Contact me at [EMAIL_REDACTED]"
