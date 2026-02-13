"""Integration tests for `ChatOpenRouter` chat model."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from pydantic import BaseModel, Field

from langchain_openrouter.chat_models import ChatOpenRouter


def test_basic_invoke() -> None:
    """Test basic invocation."""
    model = ChatOpenRouter(model="openai/gpt-4o-mini", temperature=0)
    response = model.invoke("Say 'hello' and nothing else.")
    assert response.content
    assert response.response_metadata.get("model_provider") == "openrouter"


def test_streaming() -> None:
    """Test streaming."""
    model = ChatOpenRouter(model="openai/gpt-4o-mini", temperature=0)
    full: BaseMessageChunk | None = None
    for chunk in model.stream("Say 'hello' and nothing else."):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content


def test_tool_calling() -> None:
    """Test tool calling via OpenRouter."""

    class GetWeather(BaseModel):
        """Get the current weather in a given location."""

        location: str = Field(description="The city and state")

    model = ChatOpenRouter(model="openai/gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools([GetWeather])
    response = model_with_tools.invoke("What's the weather in San Francisco?")
    assert response.tool_calls


def test_structured_output() -> None:
    """Test structured output via OpenRouter."""

    class Joke(BaseModel):
        """A joke."""

        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline of the joke")

    model = ChatOpenRouter(model="openai/gpt-4o-mini", temperature=0)
    structured = model.with_structured_output(Joke)
    result = structured.invoke("Tell me a joke about programming")
    assert isinstance(result, Joke)
    assert result.setup
    assert result.punchline


@pytest.mark.xfail(reason="Depends on reasoning model availability on OpenRouter.")
def test_reasoning_content() -> None:
    """Test reasoning content from a reasoning model."""
    model = ChatOpenRouter(
        model="openai/o3-mini",
        openrouter_reasoning={"effort": "low"},
    )
    response = model.invoke("What is 2 + 2?")
    assert response.content
