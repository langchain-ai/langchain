"""Tests for `InternalCallTransformer` filtering middleware-internal model calls."""

# `run.tool_calls`/`run.subagents` are stream projections registered dynamically by
# `create_agent` (via langgraph-prebuilt's `ToolCallTransformer` and langchain's
# `SubagentTransformer`), not declared on langgraph's typed `GraphRunStream`
# (langgraph#8389). The `# type: ignore[attr-defined]` below self-remove once
# langgraph adds a `__getattr__` fallback (strict mode's `warn_unused_ignores`).

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.stream.transformers import MessagesTransformer

from langchain.agents.factory import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import (
    internal_call_transformer as internal_call_transformer_module,
)
from langchain.agents.middleware.internal_call_transformer import (
    INTERNAL_CALL_METADATA_KEY,
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware.summarization import SummarizationMiddleware
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
    list(run.tool_calls)  # type: ignore[attr-defined]


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
    list(run.tool_calls)  # type: ignore[attr-defined]


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
    list(run.tool_calls)  # type: ignore[attr-defined]


def test_internal_call_transformer_deduped_alongside_builtins() -> None:
    """Dedup covers the whole transformer list, not just middleware-supplied ones."""
    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[
            _InternalCallMiddleware(
                GenericFakeChatModel(messages=iter([AIMessage(content="internal")]))
            )
        ],
        # Re-supplying a built-in explicitly should still collapse to one instance.
        transformers=[InternalCallTransformer],
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
    transformers = run._mux._transformers

    assert sum(isinstance(t, InternalCallTransformer) for t in transformers) == 1

    # Drain to close cleanly.
    list(run.tool_calls)  # type: ignore[attr-defined]


def test_internal_call_transformer_dedup_accepts_unhashable_factories() -> None:
    """De-dup must not require transformer factories to be hashable."""

    class _UnhashableFactory:
        """A callable factory with hashing disabled, like an `eq`-only dataclass."""

        __hash__ = None  # type: ignore[assignment]

        def __call__(self, scope: tuple[str, ...]) -> InternalCallTransformer:
            return InternalCallTransformer(scope)

    class _UnhashableTransformerMiddleware(AgentMiddleware):
        transformers = (_UnhashableFactory(),)

    # Must not raise `TypeError: unhashable type`.
    agent = create_agent(
        model=FakeToolCallingModel(), tools=[], middleware=[_UnhashableTransformerMiddleware()]
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
    list(run.tool_calls)  # type: ignore[attr-defined]


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

    Mutating an event to influence `MessagesTransformer` (there's no metadata
    hook it consults, so mutation is the only lever) would misrepresent the
    call to any consumer that also sees the raw log — a real AI response
    reported as `role: "tool"`, for instance. So tagged calls are dropped
    from the raw log entirely rather than published in a mutated form.
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

    payloads: list[Any] = []
    async for event in run:
        if event.get("method") != "messages":
            continue
        data = event["params"].get("data")
        if isinstance(data, tuple) and len(data) == 2:
            payloads.append(data[0])

    # Only the main turn's message made it to the raw log.
    assert len(payloads) == 1
    assert isinstance(payloads[0], BaseMessage)
    assert payloads[0].text == "hi"


async def test_caller_supplied_metadata_cannot_forge_internal_marker() -> None:
    """A caller can't hide the agent's real answer by guessing the marker's value.

    `config["metadata"]` on the top-level `invoke`/`stream_events` call flows
    into every model call made during the run, including the main agent
    turn's own — so if the marker were a predictable value (e.g. `True`), an
    attacker-controlled caller (or an API layer forwarding user-supplied
    metadata) could set `lc_internal_call` themselves and make the agent's
    real answer disappear from `run.messages`.
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
    token = internal_call_transformer_module._INTERNAL_CALL_TOKEN
    assert token not in (True, False, None, "", "true", "internal", INTERNAL_CALL_METADATA_KEY)
    assert len(str(token)) >= 16


def test_process_ignores_events_outside_its_own_scope() -> None:
    """A root-scoped transformer must not touch a nested subgraph's events.

    `MessagesTransformer` itself ignores events whose namespace doesn't
    match its own scope, and a correctly-scoped `InternalCallTransformer`
    instance (if the offending middleware runs inside that subgraph too)
    would already handle them — so mutating or dropping them here would be
    both unnecessary and would corrupt a namespace this instance doesn't own.
    """
    transformer = InternalCallTransformer(())
    payload = {"event": "message-start", "role": "ai", "id": "msg-1"}
    metadata = {**internal_call_metadata(), "run_id": "run-1"}
    event: Any = {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": ["some-subgraph"],
            "timestamp": 0,
            "data": (payload, metadata),
        },
    }

    assert transformer.process(event) is True
    assert event["params"]["data"] == (payload, metadata)
    assert payload["role"] == "ai"
