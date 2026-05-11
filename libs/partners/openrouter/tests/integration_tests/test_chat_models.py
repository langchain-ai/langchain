"""Integration tests for `ChatOpenRouter` chat model."""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)
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
        reasoning={"effort": "low"},
    )
    response = model.invoke("What is 2 + 2?")
    assert response.content


def test_streaming_reasoning_multi_turn() -> None:
    """Multi-turn streaming with reasoning preserves the thinking signature.

    Regression test for #36400. During streaming, `reasoning_details` is
    fragmented into multiple list entries by `AIMessageChunk.__add__` (because
    `index` is a float and `langchain_core.utils._merge.merge_lists` only
    auto-merges int-indexed dicts). When sent back on the next turn, the
    fragmented entries cause Anthropic via OpenRouter to reject the request
    with `"Invalid signature in thinking block"`. The fix in
    `_convert_message_to_dict` merges fragments before serialization.
    """
    model = ChatOpenRouter(
        model="anthropic/claude-haiku-4.5",
        reasoning={"effort": "low"},
    )

    messages: list = [HumanMessage(content="What is 2+2? Think briefly.")]

    full: BaseMessageChunk | None = None
    for chunk in model.stream(messages):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs.get("reasoning_details"), (
        "expected reasoning_details on the streamed chunk"
    )

    # Hand-build the AIMessage from the accumulated chunk and continue the
    # conversation. Pre-fix, this raises a 400 from the provider.
    assistant_msg = AIMessage(
        content=full.content,
        additional_kwargs=full.additional_kwargs,
        response_metadata=full.response_metadata,
    )
    messages.append(assistant_msg)
    messages.append(HumanMessage(content="Now what is 3+3?"))

    response = model.invoke(messages)
    assert response.content
