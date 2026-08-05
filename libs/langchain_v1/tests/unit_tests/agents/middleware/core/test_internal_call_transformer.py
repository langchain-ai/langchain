"""Tests for `InternalCallTransformer` filtering middleware-internal model calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.stream.transformers import MessagesTransformer

from langchain.agents import _internal_call_transformer
from langchain.agents._internal_call_transformer import (
    INTERNAL_CALL_METADATA_KEY,
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.factory import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse


class _InternalCallMiddleware(AgentMiddleware):
    """Calls a side model tagged as internal before letting the main turn run."""

    transformers = (InternalCallTransformer,)

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


class _InternalWholeMessageMiddleware(AgentMiddleware):
    """Calls a non-streaming internal model that surfaces via `on_chain_end`.

    A model with no `_stream`/`_astream` override reports its output as a
    whole `AIMessage` `messages`-mode event rather than streamed protocol
    events — the shape `MessagesTransformer` falls back to on Python 3.10,
    or for any model/provider that doesn't stream.
    """

    transformers = (InternalCallTransformer,)

    def __init__(self, internal_model: FakeToolCallingModel) -> None:
        super().__init__()
        self.internal_model = internal_model

    def before_model(self, state: Any, runtime: Any) -> None:  # noqa: ARG002
        self.internal_model.invoke(
            [HumanMessage("internal check")],
            config={"metadata": internal_call_metadata()},
        )

    async def abefore_model(self, state: Any, runtime: Any) -> None:  # noqa: ARG002
        await self.internal_model.ainvoke(
            [HumanMessage("internal check")],
            config={"metadata": internal_call_metadata()},
        )


def test_internal_call_transformer_not_registered_without_offending_middleware() -> None:
    """A plain agent with no internal-call middleware doesn't pay for the filter."""
    agent = create_agent(model=FakeToolCallingModel(), tools=[])

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
    transformers = run._mux._transformers

    assert not any(isinstance(t, InternalCallTransformer) for t in transformers)

    # Drain to close cleanly.
    list(run.tool_calls)


def test_internal_call_transformer_registered_before_messages_transformer() -> None:
    """Middleware declaring `transformers` registers it, ahead of built-ins."""
    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[
            _InternalCallMiddleware(
                GenericFakeChatModel(messages=iter([AIMessage(content="internal")]))
            )
        ],
    )

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


def test_internal_call_transformer_deduped_across_middleware() -> None:
    """Combining two internal-call middleware doesn't double-register the transformer."""
    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[
            _InternalCallMiddleware(
                GenericFakeChatModel(messages=iter([AIMessage(content="internal")]))
            ),
            _InternalWholeMessageMiddleware(FakeToolCallingModel(index=1000)),
        ],
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
    transformers = run._mux._transformers

    assert sum(isinstance(t, InternalCallTransformer) for t in transformers) == 1

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
    """Internal calls must not leak into the raw protocol event stream either.

    `stream_events(version="v3")` may deliver a chat model's output either as
    streamed `content-block-delta` protocol events or (e.g. when the runtime
    doesn't propagate streaming context, as on Python 3.10) as a single
    whole-`AIMessage` event, so this collects text from both shapes rather
    than assuming one.
    """
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    internal_model = GenericFakeChatModel(messages=iter([AIMessage(content="internal check")]))
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalCallMiddleware(internal_model)]
    )

    run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

    seen_text: list[str] = []
    async for event in run:
        if event.get("method") != "messages":
            continue
        data = event["params"].get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            continue
        payload = data[0]
        if isinstance(payload, dict):
            delta = payload.get("delta") or {}
            if text := delta.get("text"):
                seen_text.append(text)
        elif isinstance(payload, BaseMessage):
            seen_text.append(payload.text)

    combined = "".join(seen_text)
    assert "internal check" not in combined
    assert "final answer" in combined


async def test_internal_whole_message_excluded_from_messages_projection() -> None:
    """A non-streaming internal call's whole-`AIMessage` event is filtered too."""
    main_model = FakeToolCallingModel()
    # Offset so the two fake models don't both mint id="0" (FakeToolCallingModel
    # ids messages from a per-instance counter starting at 0), which would make
    # `MessagesTransformer`'s own id-based dedupe hide the main turn regardless
    # of this transformer.
    internal_model = FakeToolCallingModel(index=1000)
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalWholeMessageMiddleware(internal_model)]
    )

    run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

    texts: list[str] = []
    async for msg in run.messages:
        text = ""
        async for chunk in msg.text:
            text += chunk
        texts.append(text)

    assert texts == ["hi"]


async def test_internal_whole_message_excluded_from_raw_event_log() -> None:
    """A non-streaming internal call's whole-`AIMessage` event never hits the raw log."""
    main_model = FakeToolCallingModel()
    internal_model = FakeToolCallingModel(index=1000)
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalWholeMessageMiddleware(internal_model)]
    )

    run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

    seen_messages: list[BaseMessage] = []
    async for event in run:
        if event.get("method") != "messages":
            continue
        data = event["params"].get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            continue
        payload = data[0]
        if isinstance(payload, BaseMessage):
            seen_messages.append(payload)

    assert len(seen_messages) == 1
    assert seen_messages[0].text == "hi"


async def test_caller_supplied_metadata_cannot_forge_internal_marker() -> None:
    """A caller can't hide the agent's real answer by guessing the marker's value.

    `config["metadata"]` on the top-level `invoke`/`stream_events` call flows
    into every model call made during the run, including the main agent
    turn's own — so if the marker were a predictable value (e.g. `True`), an
    attacker-controlled caller (or an API layer forwarding user-supplied
    metadata) could set `lc_internal_call` themselves and make the agent's
    real answer disappear from `run.messages` and the raw event log.
    """
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    summarizer_model = GenericFakeChatModel(messages=iter([AIMessage(content="summary")]))
    middleware = SummarizationMiddleware(model=summarizer_model, trigger=("messages", 1))
    agent = create_agent(model=main_model, tools=[], middleware=[middleware])

    forged_metadata = {INTERNAL_CALL_METADATA_KEY: True}
    assert forged_metadata != internal_call_metadata(), (
        "the real marker must not be a guessable value like `True`"
    )

    run = await agent.astream_events(
        {"messages": [HumanMessage("hi"), HumanMessage("there"), HumanMessage("again")]},
        config={"metadata": forged_metadata},
        version="v3",
    )

    texts: list[str] = []
    async for msg in run.messages:
        text = ""
        async for chunk in msg.text:
            text += chunk
        texts.append(text)

    assert texts == ["final answer"]


def test_internal_call_token_is_not_a_predictable_value() -> None:
    """The marker's value must not be something a caller could plausibly guess."""
    token = _internal_call_transformer._INTERNAL_CALL_TOKEN
    assert token not in (True, False, None, "", "true", "internal", INTERNAL_CALL_METADATA_KEY)
    assert len(str(token)) >= 16
