"""Tests for `InternalCallTransformer` filtering middleware-internal model calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.stream.transformers import MessagesTransformer

from langchain.agents.factory import create_agent
from langchain.agents.middleware import (
    INTERNAL_CALL_METADATA_KEY,
    AgentMiddleware,
    InternalCallTransformer,
    internal_call_metadata,
)
from langchain.agents.middleware import (
    internal_call_transformer as internal_call_transformer_module,
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
    list(run.tool_calls)


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


def test_internal_model_calls_preserved_in_raw_event_log() -> None:
    """Internal calls stay visible on the raw event log for audit purposes.

    Only `run.messages` should exclude internal middleware calls — the raw
    `stream_events` iterator (and any tracing/observability consumer reading
    it) should still see everything, including the internal call's tokens.

    Sync `stream_events`, not `astream_events`: on Python 3.10, an internal
    call made from `wrap_model_call` doesn't reach the `messages` stream mode
    at all under the async path (streaming context isn't propagated the same
    way it is on 3.11+), which would make this assertion meaningless there
    regardless of what this transformer does.
    """
    main_model = GenericFakeChatModel(messages=iter([AIMessage(content="final answer")]))
    internal_model = GenericFakeChatModel(messages=iter([AIMessage(content="internal check")]))
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalCallMiddleware(internal_model)]
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    seen_text: list[str] = []
    for event in run:
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
    assert "internal check" in combined
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


def test_internal_whole_message_redacted_but_present_in_raw_event_log() -> None:
    """A non-streaming internal call's whole-`AIMessage` event stays on the raw log.

    Its content is cleared (there's no way to keep `MessagesTransformer` from
    routing the very same object into `run.messages` while leaving the
    original text visible elsewhere), but the event itself — an audit trace
    that a call happened — is not dropped.

    Sync `stream_events`, not `astream_events`: see
    `test_internal_model_calls_preserved_in_raw_event_log` for why the async
    path can't reliably reach the `messages` stream mode on Python 3.10.
    """
    main_model = FakeToolCallingModel()
    internal_model = FakeToolCallingModel(index=1000)
    agent = create_agent(
        model=main_model, tools=[], middleware=[_InternalWholeMessageMiddleware(internal_model)]
    )

    run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

    payloads: list[Any] = []
    for event in run:
        if event.get("method") != "messages":
            continue
        data = event["params"].get("data")
        if isinstance(data, tuple) and len(data) == 2:
            payloads.append(data[0])

    # Both the main turn's message and the internal call's (redacted) message
    # are still on the raw log — neither was silently dropped.
    assert len(payloads) == 2
    assert any(isinstance(p, BaseMessage) and p.text == "hi" for p in payloads)
    assert any(p is None for p in payloads)


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
