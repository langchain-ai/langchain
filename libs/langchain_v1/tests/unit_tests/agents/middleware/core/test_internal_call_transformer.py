"""Tests for `InternalCallTransformer` filtering middleware-internal model calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.stream.transformers import MessagesTransformer

from langchain.agents._internal_call_transformer import (
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse


class _InternalCallMiddleware(AgentMiddleware):
    """Calls a side model tagged as internal before letting the main turn run."""

    def __init__(self, internal_model: GenericFakeChatModel) -> None:
        super().__init__()
        self.internal_model = internal_model

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        self.internal_model.invoke(
            [HumanMessage("internal check")],
            config={"metadata": internal_call_metadata()},
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        await self.internal_model.ainvoke(
            [HumanMessage("internal check")],
            config={"metadata": internal_call_metadata()},
        )
        return await handler(request)


def test_internal_call_transformer_registered_before_messages_transformer() -> None:
    """`InternalCallTransformer` is registered unconditionally, ahead of built-ins."""
    agent = create_agent(model=FakeToolCallingModel(), tools=[])

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
    transformers = run._mux._transformers

    internal_idx = next(
        i for i, t in enumerate(transformers) if isinstance(t, InternalCallTransformer)
    )
    messages_idx = next(i for i, t in enumerate(transformers) if isinstance(t, MessagesTransformer))
    assert internal_idx < messages_idx, (
        "InternalCallTransformer must be registered before MessagesTransformer "
        "so tagged events never reach the messages projection"
    )

    # Drain to close cleanly.
    list(run.tool_calls)


def test_internal_model_calls_excluded_from_messages_projection_sync() -> None:
    """Sync `stream_events` should also exclude internal middleware calls."""
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    internal_model = GenericFakeChatModel(messages=iter([AIMessage(content="internal check")]))
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalCallMiddleware(internal_model)]
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    streams = list(run.messages)
    assert len(streams) == 1

    # Drain the surviving stream so the run closes cleanly.
    list(streams[0].text)


async def test_internal_model_calls_excluded_from_messages_projection() -> None:
    """`run.messages` should only surface the main turn, not internal middleware calls."""
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    internal_model = GenericFakeChatModel(messages=iter([AIMessage(content="internal check")]))
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalCallMiddleware(internal_model)]
    )

    run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

    streams = [msg async for msg in run.messages]
    assert len(streams) == 1

    # Drain the surviving stream so the run closes cleanly.
    async for _ in streams[0].text:
        pass


async def test_internal_model_calls_excluded_from_raw_event_log() -> None:
    """Internal calls must not leak into the raw protocol event stream either."""
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    internal_model = GenericFakeChatModel(messages=iter([AIMessage(content="internal check")]))
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalCallMiddleware(internal_model)]
    )

    run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

    message_start_count = 0
    async for event in run:
        if event.get("method") != "messages":
            continue
        data = event["params"].get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            continue
        payload = data[0]
        if isinstance(payload, dict) and payload.get("event") == "message-start":
            message_start_count += 1

    assert message_start_count == 1
