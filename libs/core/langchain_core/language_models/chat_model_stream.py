"""Per-message streaming objects for content-block protocol events.

`ChatModelStream` is the synchronous variant returned by
`BaseChatModel.stream_v2()`.  `AsyncChatModelStream` is the
asynchronous variant returned by `BaseChatModel.astream_v2()`.

Both expose typed projection properties (`.text`, `.reasoning`,
`.tool_calls`, `.usage`, `.output`) that accumulate protocol
events as they arrive.  Projections can be iterated for deltas or
drained for the final accumulated value.

Raw protocol events are also available via direct iteration on the
stream object (replay-buffer semantics — multiple independent
consumers supported).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from langchain_protocol.protocol import (
    ContentBlockDeltaData,
    ContentBlockFinishData,
    InvalidToolCallBlock,
    MessageFinishData,
    MessageMetadata,
    MessageStartData,
    ReasoningBlock,
    ServerToolCallBlock,
    ServerToolCallChunkBlock,
    TextBlock,
    ToolCallBlock,
    ToolCallChunkBlock,
    UsageInfo,
)

from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from langchain_protocol.protocol import FinalizedContentBlock, MessagesData


# ---------------------------------------------------------------------------
# Tool-call chunk helpers (shared by tool_call_chunk and server_tool_call_chunk)
# ---------------------------------------------------------------------------


def _merge_chunk_into_store(
    store: dict[int, dict[str, Any]],
    idx: int,
    block: dict[str, Any],
) -> None:
    """Merge a tool-call-chunk delta: sticky id/name, concat args."""
    existing = store.get(idx, {})
    if block.get("id") and "id" not in existing:
        existing["id"] = block["id"]
    if block.get("name") and "name" not in existing:
        existing["name"] = block["name"]
    existing["args"] = existing.get("args", "") + (block.get("args") or "")
    store[idx] = existing


def _sweep_chunk_store(
    store: dict[int, dict[str, Any]],
    *,
    finalized_type: str,
    finalized_blocks: dict[int, FinalizedContentBlock],
    tool_calls_acc: list[ToolCallBlock] | None,
    invalid_acc: list[InvalidToolCallBlock],
) -> None:
    """Parse each unswept chunk's `args`; record as `finalized_type` or invalid.

    `tool_calls_acc` is only populated when `finalized_type == "tool_call"`
    (server-side calls don't surface through `.tool_calls`).
    """
    for idx in sorted(store):
        chunk = store[idx]
        raw_args = chunk.get("args", "{}")
        try:
            parsed = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            invalid: InvalidToolCallBlock = {
                "type": "invalid_tool_call",
                "args": raw_args or "",
                "error": "Failed to parse tool call arguments as JSON",
            }
            if chunk.get("id"):
                invalid["id"] = chunk["id"]
            if chunk.get("name"):
                invalid["name"] = chunk["name"]
            invalid_acc.append(invalid)
            finalized_blocks[idx] = invalid
            continue
        final_block = cast(
            "FinalizedContentBlock",
            {
                "type": finalized_type,
                "id": chunk.get("id", ""),
                "name": chunk.get("name", ""),
                "args": parsed,
            },
        )
        if tool_calls_acc is not None and finalized_type == "tool_call":
            tool_calls_acc.append(cast("ToolCallBlock", final_block))
        finalized_blocks[idx] = final_block
    store.clear()


# ---------------------------------------------------------------------------
# Projection base — shared producer API
# ---------------------------------------------------------------------------


class _ProjectionBase:
    """Shared state and producer API for sync and async projections.

    The ``push`` / ``complete`` / ``fail`` methods are the producer-side
    API — called by the stream as events arrive. Subclasses add the
    consumer protocol (sync iteration or async iteration + await).

    ``done`` and ``error`` are safe read-only views of the terminal state
    for iterators and other siblings that need to observe lifecycle
    without reaching into the underlying fields.
    """

    __slots__ = ("_deltas", "_done", "_error", "_final_set", "_final_value")

    def __init__(self) -> None:
        """Initialize empty projection state."""
        self._deltas: list[Any] = []
        self._final_value: Any = None
        self._final_set: bool = False
        self._done: bool = False
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        """Whether the projection has finished (successfully or via error)."""
        return self._done

    @property
    def error(self) -> BaseException | None:
        """The terminal error, if any."""
        return self._error

    def push(self, delta: Any) -> None:
        """Append a delta value. Producer-side API."""
        self._deltas.append(delta)

    def complete(self, final_value: Any) -> None:
        """Set the final accumulated value and mark as done. Producer-side API."""
        self._final_value = final_value
        self._final_set = True
        self._done = True

    def fail(self, error: BaseException) -> None:
        """Mark as errored. Producer-side API."""
        self._error = error
        self._done = True


# ---------------------------------------------------------------------------
# Sync projections
# ---------------------------------------------------------------------------


class SyncProjection(_ProjectionBase):
    """Sync iterable of deltas with pull-based backpressure.

    Follows the same ``_request_more`` convention as langgraph's
    ``EventLog``: when the cursor catches up to the buffer and the
    projection is not done, it calls ``_request_more()`` to pull more
    events from the producer.

    Each call to ``__iter__`` creates a new cursor at position 0.
    Multiple iterators replay all deltas from the start.
    """

    __slots__ = ("_request_more",)

    def __init__(self) -> None:
        """Initialize with no pull callback."""
        super().__init__()
        self._request_more: Callable[[], bool] | None = None

    def set_request_more(self, cb: Callable[[], bool] | None) -> None:
        """Install the pull callback the iterator uses to drain the source."""
        self._request_more = cb

    def __iter__(self) -> Iterator[Any]:
        """Yield deltas, pulling via ``_request_more`` when caught up."""
        cursor = 0
        while True:
            if cursor < len(self._deltas):
                yield self._deltas[cursor]
                cursor += 1
            elif self._error is not None:
                raise self._error
            elif self._done:
                return
            elif self._request_more is not None:
                while cursor >= len(self._deltas) and not self._done:
                    if not self._request_more():
                        break
                if cursor >= len(self._deltas):
                    if self._error is not None:
                        raise self._error
                    return
            else:
                return

    def get(self) -> Any:
        """Drain via `_request_more` and return the final value."""
        if not self._done and self._request_more is not None:
            while not self._done:
                if not self._request_more():
                    break
        if self._error is not None:
            raise self._error
        return self._final_value


class SyncTextProjection(SyncProjection):
    """String-specialized sync projection.

    Adds `__str__`, `__bool__`, `__repr__` for ergonomic use with
    `.text` and `.reasoning` projections.
    """

    __slots__ = ()

    def __str__(self) -> str:
        """Drain and return the full accumulated string."""
        val = self.get()
        return val if val is not None else ""

    def __bool__(self) -> bool:
        """Return whether any deltas have been pushed."""
        return len(self._deltas) > 0

    def __repr__(self) -> str:
        """Return repr of the accumulated text so far."""
        if self._final_set:
            return repr(self._final_value)
        return repr("".join(self._deltas))


# ---------------------------------------------------------------------------
# Async projection
# ---------------------------------------------------------------------------


class AsyncProjection(_ProjectionBase):
    """Async iterable of deltas that is also awaitable for the final value.

    Uses an `asyncio.Event` to notify consumers of state changes. Each
    waiter — the awaitable (`__await__`) and each async iterator cursor
    — shares the event and re-checks its own condition on wake. The event
    is cleared before a waiter awaits, so stale "something happened"
    signals don't cause spin loops.

    This is single-loop only — producers and consumers must share an
    event loop. If cross-thread wake is ever required, revert to a
    list-of-futures pattern with `call_soon_threadsafe`.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Initialize with an un-set event."""
        super().__init__()
        self._event = asyncio.Event()

    def push(self, delta: Any) -> None:
        """Append a delta and notify waiters."""
        super().push(delta)
        self._event.set()

    def complete(self, final_value: Any) -> None:
        """Set the final value, mark done, and notify waiters."""
        super().complete(final_value)
        self._event.set()

    def fail(self, error: BaseException) -> None:
        """Mark errored and notify waiters."""
        super().fail(error)
        self._event.set()

    # -- Async iterable (yields deltas) ------------------------------------

    def __aiter__(self) -> _AsyncProjectionIterator:
        """Return an async iterator over deltas."""
        return _AsyncProjectionIterator(self)

    # -- Awaitable (returns final value) -----------------------------------

    def __await__(self) -> Generator[Any, None, Any]:
        """Await the final accumulated value."""
        return self._await_impl().__await__()

    async def _await_impl(self) -> Any:
        """Wait until the final value is set and return it."""
        while not self._final_set:
            if self._error is not None:
                raise self._error
            self._event.clear()
            await self._event.wait()
        if self._error is not None:
            raise self._error
        return self._final_value


class _AsyncProjectionIterator:
    """Async iterator over an :class:`AsyncProjection`'s deltas."""

    __slots__ = ("_offset", "_proj")

    def __init__(self, proj: AsyncProjection) -> None:
        """Initialize cursor at position 0."""
        self._proj = proj
        self._offset = 0

    def __aiter__(self) -> _AsyncProjectionIterator:
        """Return self for the async iteration protocol."""
        return self

    async def __anext__(self) -> Any:
        """Return the next delta, awaiting if necessary."""
        while True:
            # Direct access to the projection's internal list/event is
            # intentional — the iterator is the projection's sidekick and
            # depends on reading the shared buffer by cursor.
            if self._offset < len(self._proj._deltas):  # noqa: SLF001
                item = self._proj._deltas[self._offset]  # noqa: SLF001
                self._offset += 1
                return item
            if self._proj.error is not None:
                raise self._proj.error
            if self._proj.done:
                raise StopAsyncIteration
            self._proj._event.clear()  # noqa: SLF001
            await self._proj._event.wait()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Sync stream
# ---------------------------------------------------------------------------


class ChatModelStream:
    """Synchronous per-message streaming object for a single LLM response.

    Returned by `BaseChatModel.stream_v2()`.  Content-block protocol
    events are fed into this object and accumulated into typed projections.

    Projections (always return the same cached object):

    - `.text` — iterable of `str` deltas; `str()` for full text
    - `.reasoning` — same as `.text` for reasoning content
    - `.tool_calls` — iterable of `ToolCallChunkBlock` deltas;
      `.get()` returns `list[ToolCallBlock]`
    - `.usage` — blocking property, returns `UsageInfo | None`
    - `.output` — blocking property, returns assembled `AIMessage`

    Raw event iteration::

        for event in stream:
            print(event)  # MessagesData dicts
    """

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        self._namespace = namespace or []
        self._node = node
        self._message_id = message_id

        # Accumulated state
        self._text_acc: str = ""
        self._reasoning_acc: str = ""
        self._tool_call_chunks: dict[int, dict[str, Any]] = {}
        self._tool_calls_acc: list[ToolCallBlock] = []
        self._invalid_tool_calls_acc: list[InvalidToolCallBlock] = []
        self._server_tool_call_chunks: dict[int, dict[str, Any]] = {}
        # Ordered snapshot of every finalized block, keyed by event index.
        # Single source of truth for .output.content. Typed accumulators
        # (text/reasoning/tool_calls/invalid_tool_calls) continue to serve
        # the public projections.
        self._blocks: dict[int, FinalizedContentBlock] = {}
        self._usage_value: UsageInfo | None = None
        self._finish_reason: str | None = None
        self._start_metadata: MessageMetadata | None = None
        self._finish_metadata: dict[str, Any] | None = None
        self._done: bool = False
        self._error: BaseException | None = None
        self._output_message: AIMessage | None = None

        # Raw event replay buffer
        self._events: list[MessagesData] = []

        # Projections — created eagerly
        self._text_proj = SyncTextProjection()
        self._reasoning_proj = SyncTextProjection()
        self._tool_calls_proj = SyncProjection()

        # Pull callback (set by bind_pump or set_request_more)
        self._request_more: Callable[[], bool] | None = None

    # -- Pump/pull wiring --------------------------------------------------

    def bind_pump(self, pump_one: Callable[[], bool]) -> None:
        """Bind a pump for standalone streaming.

        Delegates to :meth:`set_request_more`.  Used by
        `BaseChatModel.stream_v2()`.
        """
        self.set_request_more(pump_one)

    def set_request_more(self, cb: Callable[[], bool]) -> None:
        """Set the pull callback on this stream and all its projections.

        Used by langgraph's `GraphRunStream._wire_request_more` to
        connect the shared graph pump.
        """
        self._request_more = cb
        self._text_proj.set_request_more(cb)
        self._reasoning_proj.set_request_more(cb)
        self._tool_calls_proj.set_request_more(cb)

    # -- Public projections ------------------------------------------------

    @property
    def text(self) -> SyncTextProjection:
        """Text content — iterable of `str` deltas, `str()` for full."""
        return self._text_proj

    @property
    def reasoning(self) -> SyncTextProjection:
        """Reasoning content — same interface as :attr:`text`."""
        return self._reasoning_proj

    @property
    def tool_calls(self) -> SyncProjection:
        """Tool calls — iterable of `ToolCallChunkBlock` deltas.

        `.get()` returns finalized `list[ToolCallBlock]`.
        """
        return self._tool_calls_proj

    @property
    def usage(self) -> UsageInfo | None:
        """Usage info — blocks until the stream finishes."""
        self._drain()
        if self._error is not None:
            raise self._error
        return self._usage_value

    @property
    def output(self) -> AIMessage:
        """Assembled `AIMessage` — blocks until the stream finishes."""
        self._drain()
        if self._error is not None:
            raise self._error
        if self._output_message is None:
            msg = "Stream finished without producing a message"
            raise RuntimeError(msg)
        return self._output_message

    @property
    def namespace(self) -> list[str]:
        """Graph namespace path for this message."""
        return self._namespace

    @property
    def node(self) -> str | None:
        """Graph node that produced this message."""
        return self._node

    @property
    def message_id(self) -> str | None:
        """Stable message identifier."""
        return self._message_id

    @property
    def done(self) -> bool:
        """Whether the stream has finished."""
        return self._done

    @property
    def output_message(self) -> AIMessage | None:
        """The assembled message if the stream has finished, else `None`.

        Unlike :attr:`output`, this never blocks or pumps and never raises.
        Intended for the stream driver (`stream_v2` / `astream_v2`) to
        check whether the stream produced a message before firing
        `on_llm_end` callbacks.
        """
        return self._output_message

    # -- Raw event iteration (replay buffer) -------------------------------

    def __iter__(self) -> Iterator[MessagesData]:
        """Iterate raw protocol events with replay-buffer semantics."""
        cursor = 0
        while True:
            if cursor < len(self._events):
                yield self._events[cursor]
                cursor += 1
            elif self._error is not None:
                raise self._error
            elif self._done:
                return
            elif self._request_more is not None:
                while cursor >= len(self._events) and not self._done:
                    if not self._request_more():
                        break
                if cursor >= len(self._events):
                    if self._error is not None:
                        raise self._error
                    return
            else:
                return

    # -- Event ingestion (public) ------------------------------------------

    def dispatch(self, event: MessagesData) -> None:
        """Route a protocol event to the appropriate internal handler.

        Public entry point for feeding events into the stream. Called by
        the stream driver (`stream_v2` / `astream_v2`'s pump) and by
        any observer or test that needs to inject protocol events.
        """
        self._record_event(event)
        event_type = event.get("event")
        if event_type == "message-start":
            self._push_message_start(cast("MessageStartData", event))
        elif event_type == "content-block-delta":
            self._push_content_block_delta(cast("ContentBlockDeltaData", event))
        elif event_type == "content-block-finish":
            self._push_content_block_finish(cast("ContentBlockFinishData", event))
        elif event_type == "message-finish":
            self._finish(cast("MessageFinishData", event))
        elif event_type == "error":
            self.fail(RuntimeError(event.get("message", "Unknown error")))
        # content-block-start is informational — no accumulation needed

    # -- Internal helpers --------------------------------------------------

    def _drain(self) -> None:
        """Pull all remaining events until done."""
        if self._done:
            return
        if self._request_more is not None:
            while not self._done:
                if not self._request_more():
                    break
        # If the source exhausted without a message-finish event
        # (e.g., empty response), finalize with what we have.
        if not self._done:
            self._finish(MessageFinishData(event="message-finish", reason="stop"))

    # -- Internal push API (called by dispatch) ----------------------------

    def _record_event(self, event: MessagesData) -> None:
        """Append a raw event to the replay buffer."""
        self._events.append(event)

    def _push_message_start(self, data: MessageStartData) -> None:
        """Process a ``message-start`` event."""
        self._start_metadata = data.get("metadata")

    def _push_content_block_delta(self, data: ContentBlockDeltaData) -> None:
        """Process a ``content-block-delta`` event."""
        block = data.get("content_block")
        if block is None:
            return
        btype = block.get("type", "")

        if btype == "text":
            text_block = cast("TextBlock", block)
            delta_text = text_block.get("text", "")
            if delta_text:
                self._text_acc += delta_text
                self._text_proj.push(delta_text)
        elif btype == "reasoning":
            reasoning_block = cast("ReasoningBlock", block)
            delta_r = reasoning_block.get("reasoning", "")
            if delta_r:
                self._reasoning_acc += delta_r
                self._reasoning_proj.push(delta_r)
        elif btype == "tool_call_chunk":
            tcc = cast("ToolCallChunkBlock", block)
            # The protocol puts the block index on the event
            # (``ContentBlockDeltaData``), not inside ``content_block``.
            # Fall back to ``content_block.index`` for providers that echo
            # it there.
            idx = data.get("index")
            if idx is None:
                idx = tcc.get("index", len(self._tool_call_chunks))
            _merge_chunk_into_store(self._tool_call_chunks, idx, dict(tcc))
            chunk_block = ToolCallChunkBlock(type="tool_call_chunk")
            if tcc.get("id"):
                chunk_block["id"] = tcc["id"]
            if tcc.get("name"):
                chunk_block["name"] = tcc["name"]
            if "args" in tcc:
                chunk_block["args"] = tcc["args"]
            if "index" in tcc:
                chunk_block["index"] = tcc["index"]
            self._tool_calls_proj.push(chunk_block)
        elif btype == "server_tool_call_chunk":
            stcc = cast("ServerToolCallChunkBlock", block)
            idx = data.get("index")
            if idx is None:
                idx = len(self._server_tool_call_chunks)
            _merge_chunk_into_store(
                self._server_tool_call_chunks, idx, dict(stcc),
            )

    def _push_content_block_finish(self, data: ContentBlockFinishData) -> None:
        """Process a `content-block-finish` event."""
        block = data.get("content_block")
        if block is None:
            return
        btype = block.get("type", "")
        idx = data.get("index")
        finalized: FinalizedContentBlock | None = None

        if btype == "text":
            text_block = cast("TextBlock", block)
            full_text = text_block.get("text", "")
            if full_text and full_text != self._text_acc:
                self._text_acc = full_text
            finalized = cast(
                "FinalizedContentBlock",
                {"type": "text", "text": self._text_acc},
            )
        elif btype == "reasoning":
            reasoning_block = cast("ReasoningBlock", block)
            full_r = reasoning_block.get("reasoning", "")
            if full_r and full_r != self._reasoning_acc:
                self._reasoning_acc = full_r
            finalized = cast(
                "FinalizedContentBlock",
                {"type": "reasoning", "reasoning": self._reasoning_acc},
            )
        elif btype == "tool_call":
            tcb = cast("ToolCallBlock", block)
            tc = ToolCallBlock(
                type="tool_call",
                id=tcb.get("id", ""),
                name=tcb.get("name", ""),
                args=tcb.get("args", {}),
            )
            self._tool_calls_acc.append(tc)
            if idx is not None and idx in self._tool_call_chunks:
                del self._tool_call_chunks[idx]
            finalized = tc
        elif btype == "invalid_tool_call":
            itc = cast("InvalidToolCallBlock", block)
            self._invalid_tool_calls_acc.append(itc)
            # Critical: drop the stale chunk so _finish's sweep doesn't revive
            # it as an empty-args ToolCallBlock.
            if idx is not None and idx in self._tool_call_chunks:
                del self._tool_call_chunks[idx]
            if idx is not None and idx in self._server_tool_call_chunks:
                del self._server_tool_call_chunks[idx]
            finalized = itc
        elif btype in (
            "server_tool_call",
            "server_tool_result",
            "image",
            "audio",
            "video",
            "file",
            "non_standard",
        ):
            if btype == "server_tool_call" and idx is not None:
                self._server_tool_call_chunks.pop(idx, None)
            finalized = block

        if finalized is not None and idx is not None:
            self._blocks[idx] = finalized

    def _finish(self, data: MessageFinishData) -> None:
        """Process a `message-finish` event."""
        self._done = True
        self._usage_value = data.get("usage")
        self._finish_reason = data.get("reason")
        self._finish_metadata = data.get("metadata")

        # Finalize any unswept chunks — both client- and server-side.
        _sweep_chunk_store(
            self._tool_call_chunks,
            finalized_type="tool_call",
            finalized_blocks=self._blocks,
            tool_calls_acc=self._tool_calls_acc,
            invalid_acc=self._invalid_tool_calls_acc,
        )
        _sweep_chunk_store(
            self._server_tool_call_chunks,
            finalized_type="server_tool_call",
            finalized_blocks=self._blocks,
            tool_calls_acc=None,
            invalid_acc=self._invalid_tool_calls_acc,
        )

        self._text_proj.complete(self._text_acc)
        self._reasoning_proj.complete(self._reasoning_acc)
        self._tool_calls_proj.complete(self._tool_calls_acc)
        self._output_message = self._assemble_message()

    def fail(self, error: BaseException) -> None:
        """Mark the stream as errored and propagate to all projections.

        Public API — called by the stream driver (`stream_v2` /
        `astream_v2`) when the underlying producer raises, by
        :meth:`dispatch` when an `error` protocol event arrives, and by
        cancellation paths.
        """
        self._done = True
        self._error = error
        self._text_proj.fail(error)
        self._reasoning_proj.fail(error)
        self._tool_calls_proj.fail(error)

    def _assemble_message(self) -> AIMessage:
        """Build an `AIMessage` from accumulated state.

        Content is built from `self._blocks`, an index-ordered snapshot of
        finalized protocol blocks. The bare-string fast path is used when
        the message has exactly one `text` block (the common chat case);
        otherwise content is a list of protocol-shape block dicts.
        """
        content: Any
        if not self._blocks:
            content = self._text_acc
        else:
            ordered_blocks = [self._blocks[idx] for idx in sorted(self._blocks)]
            if (
                len(ordered_blocks) == 1
                and ordered_blocks[0].get("type") == "text"
            ):
                content = cast("TextBlock", ordered_blocks[0]).get("text", "")
            else:
                content = [dict(b) for b in ordered_blocks]

        response_metadata: dict[str, Any] = {}
        if self._finish_reason:
            response_metadata["finish_reason"] = self._finish_reason
        if self._start_metadata:
            if "provider" in self._start_metadata:
                response_metadata["model_provider"] = self._start_metadata["provider"]
            if "model" in self._start_metadata:
                response_metadata["model_name"] = self._start_metadata["model"]
        if self._finish_metadata:
            response_metadata.update(self._finish_metadata)

        tool_calls = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "type": "tool_call",
            }
            for tc in self._tool_calls_acc
        ]

        invalid_tool_calls = [
            {
                "type": "invalid_tool_call",
                "id": itc.get("id") or None,
                "name": itc.get("name") or None,
                "args": itc.get("args") or None,
                "error": itc.get("error"),
            }
            for itc in self._invalid_tool_calls_acc
        ]

        return AIMessage(
            content=content,
            id=self._message_id,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            usage_metadata=self._usage_value,
            response_metadata=response_metadata,
        )


# ---------------------------------------------------------------------------
# Async stream
# ---------------------------------------------------------------------------


class AsyncChatModelStream(ChatModelStream):
    """Asynchronous per-message streaming object for a single LLM response.

    Returned by `BaseChatModel.astream_v2()`.  Content-block events
    are fed into this object by a background producer task.

    Projections:

    - `.text` — async iterable of text deltas; awaitable for full text
    - `.reasoning` — async iterable of reasoning deltas; awaitable
    - `.tool_calls` — async iterable of `ToolCallChunkBlock` deltas;
      awaitable for `list[ToolCallBlock]`
    - `.usage` — awaitable for `UsageInfo`
    - `.output` — awaitable for assembled `AIMessage`

    The stream itself is awaitable (`msg = await stream`) and
    async-iterable (`async for event in stream`).
    """

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
        self._text_proj = AsyncProjection()  # type: ignore[assignment]
        self._reasoning_proj = AsyncProjection()  # type: ignore[assignment]
        self._tool_calls_proj = AsyncProjection()  # type: ignore[assignment]
        self._usage_proj = AsyncProjection()
        self._output_proj = AsyncProjection()
        self._events_proj = AsyncProjection()
        self._producer_task: asyncio.Task[None] | None = None

    # -- Public projections (override sync properties) ---------------------

    @property
    def text(self) -> AsyncProjection:  # type: ignore[override]
        """Text content — async iterable of deltas, awaitable for full."""
        return self._text_proj  # type: ignore[return-value]

    @property
    def reasoning(self) -> AsyncProjection:  # type: ignore[override]
        """Reasoning content — same interface as :attr:`text`."""
        return self._reasoning_proj  # type: ignore[return-value]

    @property
    def tool_calls(self) -> AsyncProjection:  # type: ignore[override]
        """Tool calls — async iterable, awaitable for finalized list."""
        return self._tool_calls_proj  # type: ignore[return-value]

    @property
    def usage(self) -> AsyncProjection:  # type: ignore[override]
        """Usage info — awaitable for `UsageInfo`."""
        return self._usage_proj

    @property
    def output(self) -> AsyncProjection:  # type: ignore[override]
        """Assembled `AIMessage` — awaitable."""
        return self._output_proj

    def __await__(self) -> Generator[Any, None, AIMessage]:
        """Await the assembled `AIMessage` and full producer lifecycle.

        The producer task is awaited after the output projection resolves so
        that post-stream work (notably `on_llm_end` callbacks) has run by
        the time the caller's `await` returns.
        """
        return self._await_full().__await__()

    async def _await_full(self) -> AIMessage:
        message: AIMessage = await self._output_proj
        if self._producer_task is not None:
            await self._producer_task
        return message

    def __aiter__(self) -> _AsyncProjectionIterator:
        """Iterate raw protocol events asynchronously."""
        return _AsyncProjectionIterator(self._events_proj)

    # -- Internal API (extend base to drive async projections) -------------

    def _record_event(self, event: MessagesData) -> None:
        """Record event and push to async event replay projection."""
        super()._record_event(event)
        self._events_proj.push(event)

    def _finish(self, data: MessageFinishData) -> None:
        """Finish base projections and async-only projections."""
        super()._finish(data)
        self._usage_proj.complete(self._usage_value)
        self._output_proj.complete(self._output_message)
        self._events_proj.complete(self._events)

    def fail(self, error: BaseException) -> None:
        """Fail base projections and async-only projections."""
        super().fail(error)
        self._usage_proj.fail(error)
        self._output_proj.fail(error)
        self._events_proj.fail(error)


# ---------------------------------------------------------------------------
# Legacy dispatch helper (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def dispatch_event(
    event: MessagesData,
    stream: ChatModelStream,
) -> None:
    """Route a protocol event to the stream's :meth:`dispatch` method.

    .. deprecated::
        Prefer `stream.dispatch(event)` directly. Kept for callers that
        already import this helper.
    """
    stream.dispatch(event)


__all__ = [
    "AsyncChatModelStream",
    "AsyncProjection",
    "ChatModelStream",
    "SyncProjection",
    "SyncTextProjection",
    "dispatch_event",
]
