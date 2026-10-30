"""Tests for opt-in refusal handling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import (
    InputAgentState,
    ModelFallbackMiddleware,
    ModelRefusalMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.outputs import ChatGenerationChunk


class RefusalStreamingModel(GenericFakeChatModel):
    @override
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        for chunk in super()._stream(*args, **kwargs):
            chunk.message.response_metadata["model_provider"] = "openai"
            yield chunk


def _model(message: AIMessage) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([message]))


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_refusal_is_opt_in(*, asynchronous: bool) -> None:
    message = AIMessage(
        "I cannot help.",
        response_metadata={"stop_reason": "refusal", "model_provider": "anthropic"},
    )
    agent = create_agent(_model(message))
    inputs: InputAgentState = {"messages": [HumanMessage("request")]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert result["messages"][-1].content == message.content
    guarded = create_agent(_model(message), middleware=[ModelRefusalMiddleware()])
    with pytest.raises(ModelRefusalError) as caught:
        await guarded.ainvoke(inputs) if asynchronous else guarded.invoke(inputs)
    assert caught.value.message.content == message.content
    assert caught.value.message.response_metadata["stop_reason"] == "refusal"
    assert caught.value.is_retryable is False


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "message",
    [
        AIMessage("done"),
        AIMessage(""),
        AIMessage("incomplete", response_metadata={"status": "incomplete"}),
        AIMessage("I cannot help with that question."),
    ],
)
async def test_non_refusals_pass_through(message: AIMessage, *, asynchronous: bool) -> None:
    agent = create_agent(_model(message), middleware=[ModelRefusalMiddleware()])
    inputs: InputAgentState = {"messages": [HumanMessage("request")]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert result["messages"][-1].content == message.content


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fallback_can_handle_refusal(*, asynchronous: bool) -> None:
    primary = _model(AIMessage(content_blocks=[{"type": "refusal", "refusal": "blocked"}]))
    fallback = _model(AIMessage("fallback answer"))
    agent = create_agent(
        primary,
        middleware=[ModelFallbackMiddleware(fallback), ModelRefusalMiddleware()],
    )
    inputs: InputAgentState = {"messages": [HumanMessage("request")]}
    result = await agent.ainvoke(inputs) if asynchronous else agent.invoke(inputs)
    assert result["messages"][-1].content == "fallback answer"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_surfaces_refusal(*, asynchronous: bool) -> None:
    agent = create_agent(
        RefusalStreamingModel(
            messages=iter([AIMessage("blocked", additional_kwargs={"refusal": "blocked"})])
        ),
        middleware=[ModelRefusalMiddleware()],
    )
    inputs: InputAgentState = {"messages": [HumanMessage("request")]}
    with pytest.raises(ModelRefusalError):
        await _consume_stream(agent, inputs, asynchronous=asynchronous)


async def _consume_stream(agent: Any, inputs: Any, *, asynchronous: bool) -> None:
    if asynchronous:
        async for _ in agent.astream(inputs, stream_mode="messages"):
            pass
    else:
        await asyncio.to_thread(lambda: list(agent.stream(inputs, stream_mode="messages")))
