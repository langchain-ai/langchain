"""Per-message streaming objects for content-block protocol events.

`ChatModelStream` is the synchronous variant returned by
`BaseChatModel.stream_events(version="v3")`.  `AsyncChatModelStream` is the
asynchronous variant returned by `BaseChatModel.astream_events(version="v3")`.

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
import contextlib
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models._compat_bridge import finalize_tool_call_chunk
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping

    from langchain_protocol.protocol import (
        ContentBlockDeltaData,
        ContentBlockFinishData,
        FinalizedContentBlock,
        InvalidToolCall,
        MessageFinishData,
        MessageMetadata,
        MessagesData,
        MessageStartData,
        ReasoningContentBlock,
        ServerToolCallChunk,
        TextContentBlock,
        ToolCall,
        ToolCallChunk,
        UsageInfo,
    )
    from typing_extensions import Self


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


def _merge_block_delta_into_store(
    store: dict[int, dict[str, Any]],
    idx: int,
    fields: dict[str, Any],
) -> None:
    """Shallow-merge a block-delta snapshot into an indexed chunk store."""
    existing = store.get(idx, {})
    for key, value in fields.items():
        if value is not None:
            existing[key] = value
    store[idx] = existing


def _event_content_block(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return start/finish content, tolerating the pre-delta field name."""
    block = data.get("content") or data.get("content_block")
    return block if isinstance(block, dict) else None


def _legacy_block_to_delta(block: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the old content-block delta shape to an explicit delta."""
    btype = block.get("type")
    if btype == "text":
        return {"type": "text-delta", "text": block.get("text", "")}
    if btype == "reasoning":
        return {
            "type": "reasoning-delta",
            "reasoning": block.get("reasoning", ""),
        }
    if "data" in block:
        delta = {"type": "data-delta", "data": block.get("data", "")}
        if block.get("encoding") == "base64":
            delta["encoding"] = "base64"
        return delta
    return {"type": "legacy-block-delta", "fields": block}


def _event_delta(data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return an explicit delta, converting legacy content-block deltas."""
    delta = data.get("delta")
    if isinstance(delta, dict):
        return delta
    block = data.get("content_block")
    if isinstance(block, dict):
        return _legacy_block_to_delta(block)
    return None


def _sweep_chunk_store(
    store: dict[int, dict[str, Any]],
    *,
    finalized_type: str,
    finalized_blocks: dict[int, FinalizedContentBlock],
    tool_calls_acc: list[ToolCall] | None,
    invalid_acc: list[InvalidToolCall],
) -> None:
    """Parse each unswept chunk's `args`; record as `finalized_type` or invalid.

    `tool_calls_acc` is only populated when `finalized_type == "tool_call"`
    (server-side calls don't surface through `.tool_calls`).

    Deliberately does not backfill `index` onto finalized tool-call blocks:
    matches v1 (`AIMessage.init_tool_calls` drops `index` when substituting
    `tool_call_chunk` → `tool_call`) and prevents `merge_lists` from
    re-merging further chunks into an already-parsed args dict.
    """
    for idx in sorted(store):
        chunk = store[idx]
        # Carry over any non-finalize-rewritten fields the chunk collected
        # (e.g., `extras`). `_merge_chunk_into_store` only populates
        # `id` / `name` / `args`, so this is empty in practice today;
        # future provider-specific fields would flow through here.
        extras = {
            k: v
            for k, v in chunk.items()
            if k not in ("type", "id", "name", "args") and v is not None
        }
        final_block = finalize_tool_call_chunk(
            raw_args=chunk.get("args"),
            id_=chunk.get("id"),
            name=chunk.get("name"),
            extras=extras,
            finalized_type=finalized_type,
        )
        if final_block["type"] == "invalid_tool_call":
            invalid_acc.append(final_block)
        elif tool_calls_acc is not None and finalized_type == "tool_call":
            tool_calls_acc.append(cast("ToolCall", final_block))
        finalized_blocks[idx] = final_block
    store.clear()


# ---------------------------------------------------------------------------
# Projection base — shared producer API
# ---------------------------------------------------------------------------


class _ProjectionBase:
    """Shared state and producer API for sync and async projections.

    The `push` / `complete` / `fail` methods are the producer-side
    API — called by the stream as events arrive. Subclasses add the
    consumer protocol (sync iteration or async iteration + await).

    `done` and `error` are safe read-only views of the terminal state
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

    Follows the same `_request_more` convention as langgraph's
    `EventLog`: when the cursor catches up to the buffer and the
    projection is not done, it calls `_request_more()` to pull more
    events from the producer.

    Each call to `__iter__` creates a new cursor at position 0.
    Multiple iterators replay all deltas from the start.
    """

    __slots__ = ("_ensure_started", "_request_more")

    def __init__(self) -> None:
        """Initialize with no pull callback."""
        super().__init__()
        self._ensure_started: Callable[[], None] | None = None
        self._request_more: Callable[[], bool] | None = None

    def set_start(self, cb: Callable[[], None] | None) -> None:
        """Install a lazy-start callback invoked on first consumption."""
        self._ensure_started = cb

    def set_request_more(self, cb: Callable[[], bool] | None) -> None:
        """Install the pull callback the iterator uses to drain the source."""
        self._request_more = cb

    def __iter__(self) -> Iterator[Any]:
        """Yield deltas, pulling via `_request_more` when caught up."""
        if self._ensure_started is not None:
            self._ensure_started()
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
        if self._ensure_started is not None:
            self._ensure_started()
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

    __slots__ = ("_arequest_more", "_ensure_started", "_event")

    def __init__(self) -> None:
        """Initialize with an un-set event and no pump callback."""
        super().__init__()
        self._event = asyncio.Event()
        self._arequest_more: Callable[[], Awaitable[bool]] | None = None
        self._ensure_started: Callable[[], Awaitable[None]] | None = None

    def set_start(self, cb: Callable[[], Awaitable[None]] | None) -> None:
        """Install a lazy-start callback invoked on first consumption."""
        self._ensure_started = cb

    def set_arequest_more(self, cb: Callable[[], Awaitable[bool]] | None) -> None:
        """Wire the async pull callback iterators use to drive the source.

        Mirrors `SyncProjection.set_request_more`. Under caller-driven
        streaming, consumers call this callback when their buffer is
        empty so that the owning graph advances one step.

        Args:
            cb: Async no-arg callable returning `True` when a new event
                was produced, `False` when the source is exhausted. Pass
                `None` to unwire.
        """
        self._arequest_more = cb

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
        """Wait until the final value is set and return it.

        When a caller-driven pump is wired via `set_arequest_more`, drive
        it instead of blocking on `self._event`; otherwise fall back to
        the event (used by tests that dispatch manually).
        """
        if self._ensure_started is not None:
            await self._ensure_started()
        while not self._final_set:
            if self._error is not None:
                raise self._error
            if self._arequest_more is not None:
                if not await self._arequest_more() and not self._final_set:
                    # Pump exhausted without completing this projection —
                    # nothing more will arrive. Return current state and
                    # let callers observe the missing final via the
                    # returned None / unset error.
                    break
            else:
                self._event.clear()
                await self._event.wait()
        if self._error is not None:
            raise self._error
        return self._final_value


class _AsyncProjectionIterator:
    """Async iterator over an `AsyncProjection`'s deltas."""

    __slots__ = ("_offset", "_proj")

    def __init__(self, proj: AsyncProjection) -> None:
        """Initialize cursor at position 0."""
        self._proj = proj
        self._offset = 0

    def __aiter__(self) -> _AsyncProjectionIterator:
        """Return self for the async iteration protocol."""
        return self

    async def __anext__(self) -> Any:
        """Return the next delta, awaiting if necessary.

        When the projection has an `_arequest_more` pump wired, drain it
        in an inner loop (mirrors `SyncProjection.__iter__`) until this
        cursor advances or the pump reports exhaustion. Without a pump,
        fall back to waiting on the shared event.
        """
        proj = self._proj
        if proj._ensure_started is not None:  # noqa: SLF001
            await proj._ensure_started()  # noqa: SLF001
        while True:
            # Direct access to the projection's internal list/event is
            # intentional — the iterator is the projection's sidekick and
            # depends on reading the shared buffer by cursor.
            if self._offset < len(proj._deltas):  # noqa: SLF001
                item = proj._deltas[self._offset]  # noqa: SLF001
                self._offset += 1
                return item
            if proj.error is not None:
                raise proj.error
            if proj.done:
                raise StopAsyncIteration
            if proj._arequest_more is not None:  # noqa: SLF001
                # Caller-driven: drive the producer. Pump may land new
                # deltas for a sibling projection — loop until our cursor
                # advances, the projection terminates, or the pump is
                # exhausted.
                while (
                    self._offset >= len(proj._deltas)  # noqa: SLF001
                    and not proj.done
                ):
                    if not await proj._arequest_more():  # noqa: SLF001
                        break
                if (
                    self._offset >= len(proj._deltas)  # noqa: SLF001
                    and not proj.done
                ):
                    if proj.error is not None:
                        raise proj.error
                    raise StopAsyncIteration
            else:
                proj._event.clear()  # noqa: SLF001
                await proj._event.wait()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Sync stream
# ---------------------------------------------------------------------------


class _ChatModelStreamBase:
    """Shared state and event dispatch for chat-model streams.

    Holds accumulated protocol state (text, reasoning, tool calls,
    usage, metadata) and the event-dispatch machinery that drives the
    typed projections. `ChatModelStream` (sync) and
    `AsyncChatModelStream` (async) inherit from this base and add the
    projection types and consumer APIs for their flavor.
    """

    # Projection instances — concrete subclasses create them as sync or
    # async variants in their own __init__ after calling super().
    _text_proj: _ProjectionBase
    _reasoning_proj: _ProjectionBase
    _tool_calls_proj: _ProjectionBase

    def __init__(
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
        # Per-block text / reasoning storage keyed by wire index. Used to
        # populate the finalized block payload without cross-contaminating
        # other blocks of the same type in the same message. Without
        # per-block storage the message-wide accumulator would bleed
        # earlier block text into later finalized blocks.
        self._text_per_block: dict[int, str] = {}
        self._reasoning_per_block: dict[int, str] = {}
        self._tool_call_chunks: dict[int, dict[str, Any]] = {}
        self._tool_calls_acc: list[ToolCall] = []
        self._invalid_tool_calls_acc: list[InvalidToolCall] = []
        self._server_tool_call_chunks: dict[int, dict[str, Any]] = {}
        # Ordered snapshot of every finalized block, keyed by event index.
        # Single source of truth for .output.content. Typed accumulators
        # (text/reasoning/tool_calls/invalid_tool_calls) continue to serve
        # the public projections.
        self._blocks: dict[int, FinalizedContentBlock] = {}
        self._usage_value: UsageInfo | None = None
        self._start_metadata: MessageMetadata | None = None
        self._finish_metadata: dict[str, Any] | None = None
        self._done: bool = False
        self._error: BaseException | None = None
        self._output_message: AIMessage | None = None

        # Raw event replay buffer
        self._events: list[MessagesData] = []

    # -- Common properties ------------------------------------------------

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

    def set_message_id(self, message_id: str) -> None:
        """Assign the stable message identifier once the run starts.

        Called by the stream driver (`stream_events(version="v3")` /
        `astream_events(version="v3")`) after `on_chat_model_start` produces a run
        id. Not intended for end-user code.
        """
        self._message_id = message_id

    @property
    def done(self) -> bool:
        """Whether the stream has finished."""
        return self._done

    @property
    def has_events(self) -> bool:
        """Whether any protocol events have been recorded."""
        return bool(self._events)

    @property
    def output_message(self) -> AIMessage | None:
        """The assembled message if the stream has finished, else `None`.

        Unlike `ChatModelStream.output` (which blocks until the stream
        finishes), this never pumps, blocks, or raises. Intended for the
        stream driver (`stream_events(version="v3")` and its async
        equivalent) to check whether the stream produced a message before
        firing `on_llm_end` callbacks.
        """
        return self._output_message

    # -- Event ingestion (public) ------------------------------------------

    def dispatch(self, event: Mapping[str, Any]) -> None:
        """Route a protocol event to the appropriate internal handler.

        Public entry point for feeding events into the stream. Called by
        the stream driver (the `stream_events(version="v3")` pump and its
        async equivalent) and by any observer or test that needs to
        inject protocol events.
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

    # -- Internal push API (called by dispatch) ----------------------------

    def _record_event(self, event: Mapping[str, Any]) -> None:
        """Append a raw event to the replay buffer."""
        self._events.append(cast("MessagesData", event))

    def _push_message_start(self, data: MessageStartData) -> None:
        """Process a `message-start` event."""
        self._start_metadata = data.get("metadata")
        message_id = data.get("id")
        if message_id:
            self._message_id = message_id

    def _push_content_block_delta(self, data: ContentBlockDeltaData) -> None:
        """Process a `content-block-delta` event."""
        delta = _event_delta(data)
        if delta is None:
            return
        event_idx = data.get("index")
        dtype = delta.get("type", "")

        if dtype == "text-delta":
            delta_text = delta.get("text", "")
            if delta_text:
                self._text_acc += delta_text
                if event_idx is not None:
                    self._text_per_block[event_idx] = (
                        self._text_per_block.get(event_idx, "") + delta_text
                    )
                self._text_proj.push(delta_text)
        elif dtype == "reasoning-delta":
            delta_r = delta.get("reasoning", "")
            if delta_r:
                self._reasoning_acc += delta_r
                if event_idx is not None:
                    self._reasoning_per_block[event_idx] = (
                        self._reasoning_per_block.get(event_idx, "") + delta_r
                    )
                self._reasoning_proj.push(delta_r)
        elif dtype == "block-delta":
            fields = delta.get("fields")
            if not isinstance(fields, dict):
                return
            btype = fields.get("type", "")
            if btype == "tool_call_chunk":
                tcc = cast("ToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = tcc.get("index", len(self._tool_call_chunks))
                _merge_block_delta_into_store(self._tool_call_chunks, idx, dict(tcc))
                chunk_block: ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tcc.get("id"),
                    "name": tcc.get("name"),
                    "args": tcc.get("args"),
                }
                if "index" in tcc:
                    chunk_block["index"] = tcc["index"]
                self._tool_calls_proj.push(chunk_block)
            elif btype == "server_tool_call_chunk":
                stcc = cast("ServerToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = len(self._server_tool_call_chunks)
                _merge_block_delta_into_store(
                    self._server_tool_call_chunks,
                    idx,
                    dict(stcc),
                )
        elif dtype == "legacy-block-delta":
            fields = delta.get("fields")
            if not isinstance(fields, dict):
                return
            btype = fields.get("type", "")
            if btype == "tool_call_chunk":
                tcc = cast("ToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = tcc.get("index", len(self._tool_call_chunks))
                _merge_chunk_into_store(self._tool_call_chunks, idx, dict(tcc))
                legacy_chunk_block: ToolCallChunk = {
                    "type": "tool_call_chunk",
                    "id": tcc.get("id"),
                    "name": tcc.get("name"),
                    "args": tcc.get("args"),
                }
                if "index" in tcc:
                    legacy_chunk_block["index"] = tcc["index"]
                self._tool_calls_proj.push(legacy_chunk_block)
            elif btype == "server_tool_call_chunk":
                stcc = cast("ServerToolCallChunk", fields)
                idx = data.get("index")
                if idx is None:
                    idx = len(self._server_tool_call_chunks)
                _merge_chunk_into_store(
                    self._server_tool_call_chunks,
                    idx,
                    dict(stcc),
                )
        elif dtype == "data-delta":
            # Binary/modal payload deltas are reflected in the final
            # content-block finish event; there is no dedicated projection.
            return
        else:
            # Transitional legacy path for old `content_block` deltas that
            # should not be reachable after `_event_delta` conversion, kept
            # here for custom in-tree test fixtures or third-party emitters.
            block = data.get("content_block")
            if not isinstance(block, dict):
                return
            btype = block.get("type", "")
            if btype != "tool_call_chunk":
                return
            tcc = cast("ToolCallChunk", block)
            idx = data.get("index")
            if idx is None:
                idx = tcc.get("index", len(self._tool_call_chunks))
            _merge_chunk_into_store(self._tool_call_chunks, idx, dict(tcc))
            fallback_chunk_block: ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tcc.get("id"),
                "name": tcc.get("name"),
                "args": tcc.get("args"),
            }
            if "index" in tcc:
                fallback_chunk_block["index"] = tcc["index"]
            self._tool_calls_proj.push(fallback_chunk_block)

    def _resolve_block_text(self, idx: int | None, full_text: str) -> str:
        """Return authoritative text for a single text block at `idx`.

        Prefers per-block delta accumulation; reconciles with the finish
        event's `full_text` when the provider emits authoritative text
        that differs from what the deltas built up.

        Does not mutate `self._text_acc` (the delta-sum accumulator) —
        the message-wide projection value is derived from per-block
        storage at `_finish` time, so reconciliation remains correct
        regardless of finish ordering across blocks.
        """
        if idx is None:
            # No wire index — legacy behavior: use the message-wide
            # accumulator. Preserved for pre-index semantics; not
            # exercised by the compat bridge or any in-tree provider.
            if full_text and full_text != self._text_acc:
                self._text_acc = full_text
            return self._text_acc
        existing = self._text_per_block.get(idx, "")
        if full_text and full_text != existing:
            if not existing:
                # No deltas arrived for this block — surface the full
                # text as a single delta so the stream projection
                # reflects it.
                self._text_acc += full_text
                self._text_proj.push(full_text)
            elif full_text.startswith(existing):
                # Authoritative text extends the partial deltas — emit
                # the tail so delta consumers see the completion.
                tail = full_text[len(existing) :]
                self._text_acc += tail
                self._text_proj.push(tail)
            # else: authoritative text replaces the partial deltas
            # entirely. No corrective delta is emitted (semantics
            # would be ambiguous mid-stream). `_text_acc` is not
            # spliced — the final value is computed from per-block
            # storage at `_finish`, so this remains correct even when
            # other blocks have added to `_text_acc` in between.
            self._text_per_block[idx] = full_text
        return self._text_per_block.get(idx, "")

    def _resolve_block_reasoning(self, idx: int | None, full_r: str) -> str:
        """Return authoritative reasoning text for a single block at `idx`.

        Mirrors `_resolve_block_text` for the reasoning projection.
        """
        if idx is None:
            if full_r and full_r != self._reasoning_acc:
                self._reasoning_acc = full_r
            return self._reasoning_acc
        existing = self._reasoning_per_block.get(idx, "")
        if full_r and full_r != existing:
            if not existing:
                self._reasoning_acc += full_r
                self._reasoning_proj.push(full_r)
            elif full_r.startswith(existing):
                tail = full_r[len(existing) :]
                self._reasoning_acc += tail
                self._reasoning_proj.push(tail)
            self._reasoning_per_block[idx] = full_r
        return self._reasoning_per_block.get(idx, "")

    def _push_content_block_finish(self, data: ContentBlockFinishData) -> None:
        """Process a `content-block-finish` event."""
        block = _event_content_block(data)
        if block is None:
            return
        btype = block.get("type", "")
        idx = data.get("index")
        finalized: FinalizedContentBlock | None = None

        if btype == "text":
            text_block = cast("TextContentBlock", block)
            full_text = text_block.get("text", "")
            block_text = self._resolve_block_text(idx, full_text)
            finalized = cast(
                "FinalizedContentBlock",
                {
                    **text_block,
                    "type": "text",
                    "text": block_text,
                },
            )
        elif btype == "reasoning":
            reasoning_block = cast("ReasoningContentBlock", block)
            full_r = reasoning_block.get("reasoning", "")
            block_reasoning = self._resolve_block_reasoning(idx, full_r)
            # Keep provider-specific fields alongside the accumulated
            # reasoning text. Anthropic's `signature` arrives under
            # `extras` and is required on follow-up turns. Only overwrite
            # `reasoning` when we have accumulated content; OpenAI can
            # emit a reasoning block with no text deltas, and writing an
            # empty string there makes downstream serializers synthesize
            # an empty summary entry.
            finalized_dict: dict[str, Any] = {**reasoning_block, "type": "reasoning"}
            if block_reasoning:
                finalized_dict["reasoning"] = block_reasoning
            finalized = cast("FinalizedContentBlock", finalized_dict)
        elif btype == "tool_call":
            tcb = cast("ToolCall", block)
            # Preserve provider-specific fields (extras, etc.) on the
            # content block. `_assemble_message` separately projects the
            # minimal {id, name, args, type} shape onto
            # `AIMessage.tool_calls`. Strip `index` to match v1
            # (`AIMessage.init_tool_calls` rebuilds the block without
            # `index`); see `_finalize_block` in `_compat_bridge.py`.
            tc = cast(
                "ToolCall",
                {
                    **{k: v for k, v in tcb.items() if k != "index"},
                    "type": "tool_call",
                    "id": tcb.get("id", ""),
                    "name": tcb.get("name", ""),
                    "args": tcb.get("args", {}),
                },
            )
            self._tool_calls_acc.append(tc)
            if idx is not None and idx in self._tool_call_chunks:
                del self._tool_call_chunks[idx]
            finalized = tc
        elif btype == "invalid_tool_call":
            itc = cast("InvalidToolCall", block)
            # Strip `index` on the stored block to stay symmetric with
            # the `tool_call` path.
            itc = cast(
                "InvalidToolCall",
                {k: v for k, v in itc.items() if k != "index"},
            )
            self._invalid_tool_calls_acc.append(itc)
            # Critical: drop the stale chunk so _finish's sweep doesn't revive
            # it as an empty-args ToolCall.
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
            finalized = cast("FinalizedContentBlock", block)

        if finalized is not None and idx is not None:
            # Backfill the wire index onto the finalized block when the
            # source didn't supply one. `langchain_core.utils._merge`'s
            # block-merger (used by `AIMessageChunk.__add__` /
            # `add_ai_message_chunks`) keys on `block["index"]` to group
            # deltas into the same output block — without it, a v2-
            # assembled `AIMessage` that later re-enters the chunk
            # aggregation path won't merge cleanly. Client-side
            # `tool_call` / `invalid_tool_call` blocks are excluded: v1
            # finalization drops `index` on them so further deltas
            # cannot clobber already-parsed args, and v2 mirrors that.
            if btype not in ("tool_call", "invalid_tool_call"):
                finalized.setdefault("index", idx)
            self._blocks[idx] = finalized

    def _finish(self, data: MessageFinishData) -> None:
        """Process a `message-finish` event."""
        self._done = True
        self._usage_value = data.get("usage")
        self._finish_metadata = cast("dict[str, Any] | None", data.get("metadata"))

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

        # Prefer the per-block sum when any indexed text / reasoning
        # arrived — it stays correct regardless of finish ordering and
        # of whether finish events carried authoritative text that
        # differed from the deltas. Fall back to the delta-sum
        # accumulator only for the legacy no-index path.
        if self._text_per_block:
            text_final = "".join(
                self._text_per_block[i] for i in sorted(self._text_per_block)
            )
        else:
            text_final = self._text_acc
        if self._reasoning_per_block:
            reasoning_final = "".join(
                self._reasoning_per_block[i] for i in sorted(self._reasoning_per_block)
            )
        else:
            reasoning_final = self._reasoning_acc

        self._text_proj.complete(text_final)
        self._reasoning_proj.complete(reasoning_final)
        self._tool_calls_proj.complete(self._tool_calls_acc)
        self._output_message = self._assemble_message()

    def fail(self, error: BaseException) -> None:
        """Mark the stream as errored and propagate to all projections.

        Public API — called by the stream driver (`stream_events(version="v3")` /
        `astream_events(version="v3")`) when the underlying producer raises, by
        `dispatch` when an `error` protocol event arrives, and by
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
            # No protocol blocks ever arrived. Fall back to the accumulated
            # text (possibly empty) as bare-string content.
            content = self._text_acc
        else:
            # `ChatModelStream` is the v1 content-block surface: content
            # is always a list of protocol blocks when any block arrived.
            # Do not collapse a single text block down to a bare string —
            # that would drop block-level fields (`id`, `index`,
            # annotations, extras) that downstream serializers need to
            # round-trip the message on a follow-up turn.
            ordered_blocks = [self._blocks[idx] for idx in sorted(self._blocks)]
            content = [dict(b) for b in ordered_blocks]

        response_metadata: dict[str, Any] = {}
        if self._start_metadata:
            if "provider" in self._start_metadata:
                response_metadata["model_provider"] = self._start_metadata["provider"]
            if "model" in self._start_metadata:
                response_metadata["model_name"] = self._start_metadata["model"]
        if self._finish_metadata:
            response_metadata.update(self._finish_metadata)
        # Pin `output_version` last: `stream_events(version="v3")` always
        # assembles content as v1 protocol blocks, regardless of the
        # provider's configured output format.
        # A provider-supplied `output_version` in finish metadata (e.g.
        # `"responses/v1"` from `ChatOpenAI(use_responses_api=True, ...)`) would
        # otherwise cause `AIMessage.content_blocks` to re-run the wrong
        # translator on already-v1 content.
        response_metadata["output_version"] = "v1"

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
# Sync stream
# ---------------------------------------------------------------------------


class ChatModelStream(_ChatModelStreamBase):
    """Synchronous per-message streaming object for a single LLM response.

    Returned by `BaseChatModel.stream_events(version="v3")`.  Content-block protocol
    events are fed into this object and accumulated into typed projections.

    Projections (always return the same cached object):

    - `.text` — iterable of `str` deltas; `str()` for full text
    - `.reasoning` — same as `.text` for reasoning content
    - `.tool_calls` — iterable of `ToolCallChunk` deltas;
      `.get()` returns `list[ToolCall]`
    - `.output` — blocking property, returns assembled `AIMessage`

    Usage info is available on `.output.usage_metadata` once the stream
    has finished.

    !!! note "Output shape is always v1 content blocks"

        `.output.content` is always a list of v1 protocol blocks
        (text, reasoning, tool_call, image, …), regardless of the
        underlying model's `output_version` setting. That attribute
        only controls the legacy `stream()` / `astream()` / `invoke()`
        paths; `ChatModelStream` is built on the content-block
        protocol and emits v1 shapes by construction.

    Raw event iteration::

        for event in stream:
            print(event)  # MessagesData dicts
    """

    _text_proj: SyncTextProjection
    _reasoning_proj: SyncTextProjection
    _tool_calls_proj: SyncProjection

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
        # Projections — created eagerly
        self._text_proj = SyncTextProjection()
        self._reasoning_proj = SyncTextProjection()
        self._tool_calls_proj = SyncProjection()
        # Pull callback (set by bind_pump or set_request_more)
        self._ensure_started: Callable[[], None] | None = None
        self._request_more: Callable[[], bool] | None = None

    # -- Pump/pull wiring --------------------------------------------------

    def bind_pump(self, pump_one: Callable[[], bool]) -> None:
        """Bind a pump for standalone streaming.

        Delegates to `set_request_more`.  Used by
        `BaseChatModel.stream_events(version="v3")`.
        """
        self.set_request_more(pump_one)

    def set_start(self, cb: Callable[[], None] | None) -> None:
        """Install a lazy-start callback on this stream and its projections."""
        self._ensure_started = cb
        self._text_proj.set_start(cb)
        self._reasoning_proj.set_start(cb)
        self._tool_calls_proj.set_start(cb)

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
        """Tool calls — iterable of `ToolCallChunk` deltas.

        `.get()` returns finalized `list[ToolCall]`.
        """
        return self._tool_calls_proj

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

    # -- Raw event iteration (replay buffer) -------------------------------

    def __iter__(self) -> Iterator[MessagesData]:
        """Iterate raw protocol events with replay-buffer semantics."""
        if self._ensure_started is not None:
            self._ensure_started()
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

    # -- Internal helpers --------------------------------------------------

    def _drain(self) -> None:
        """Pull all remaining events until done."""
        if self._done:
            return
        if self._ensure_started is not None:
            self._ensure_started()
        if self._request_more is not None:
            while not self._done:
                if not self._request_more():
                    break


# ---------------------------------------------------------------------------
# Async stream
# ---------------------------------------------------------------------------


class AsyncChatModelStream(_ChatModelStreamBase):
    """Asynchronous per-message streaming object for a single LLM response.

    Returned by `BaseChatModel.astream_events(version="v3")`.  Content-block events
    are fed into this object by a background producer task.

    Projections:

    - `.text` — async iterable of text deltas; awaitable for full text
    - `.reasoning` — async iterable of reasoning deltas; awaitable
    - `.tool_calls` — async iterable of `ToolCallChunk` deltas;
      awaitable for `list[ToolCall]`
    - `.output` — awaitable for assembled `AIMessage`

    Usage info is available on `.output.usage_metadata` once the stream
    has finished.

    !!! note "Output shape is always v1 content blocks"

        The assembled message's content is always a list of v1
        protocol blocks, regardless of the model's `output_version`
        setting — see `ChatModelStream` for the full rationale.

    The stream itself is awaitable (`msg = await stream`) and
    async-iterable (`async for event in stream`).
    """

    _text_proj: AsyncProjection
    _reasoning_proj: AsyncProjection
    _tool_calls_proj: AsyncProjection

    def __init__(  # noqa: D107
        self,
        *,
        namespace: list[str] | None = None,
        node: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(namespace=namespace, node=node, message_id=message_id)
        self._text_proj = AsyncProjection()
        self._reasoning_proj = AsyncProjection()
        self._tool_calls_proj = AsyncProjection()
        self._output_proj = AsyncProjection()
        self._events_proj = AsyncProjection()
        self._ensure_started: Callable[[], Awaitable[None]] | None = None
        self._producer_task: asyncio.Task[None] | None = None
        # Teardown callback invoked by `aclose()` only when the producer
        # task was cancelled before its body ran (so the normal
        # `_produce` CancelledError handler — which fires
        # `on_llm_error` — never executed). Set by `astream_events(version="v3")`.
        self._on_aclose_fail: Callable[[BaseException], Awaitable[None]] | None = None

    # -- Pump/pull wiring (async) ------------------------------------------

    def set_arequest_more(self, cb: Callable[[], Awaitable[bool]] | None) -> None:
        """Fan the async pump callback out to every projection.

        Used by langgraph's `AsyncGraphRunStream._wire_arequest_more` so
        cursors on `stream.text`, `stream.reasoning`, etc. can drive the
        shared graph pump when their buffer is empty.

        Args:
            cb: Async no-arg callable returning `True` when a new event
                was produced, `False` when the source is exhausted. Pass
                `None` to unwire.
        """
        for proj in (
            self._text_proj,
            self._reasoning_proj,
            self._tool_calls_proj,
            self._output_proj,
            self._events_proj,
        ):
            proj.set_arequest_more(cb)

    def set_start(self, cb: Callable[[], Awaitable[None]] | None) -> None:
        """Install a lazy-start callback on this stream and its projections."""
        self._ensure_started = cb
        for proj in (
            self._text_proj,
            self._reasoning_proj,
            self._tool_calls_proj,
            self._output_proj,
            self._events_proj,
        ):
            proj.set_start(cb)

    # -- Public projections ------------------------------------------------

    @property
    def text(self) -> AsyncProjection:
        """Text content — async iterable of deltas, awaitable for full."""
        return self._text_proj

    @property
    def reasoning(self) -> AsyncProjection:
        """Reasoning content — same interface as :attr:`text`."""
        return self._reasoning_proj

    @property
    def tool_calls(self) -> AsyncProjection:
        """Tool calls — async iterable, awaitable for finalized list."""
        return self._tool_calls_proj

    @property
    def output(self) -> AsyncProjection:
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
        if self._ensure_started is not None:
            await self._ensure_started()
        message: AIMessage = await self._output_proj
        if self._producer_task is not None:
            await self._producer_task
        return message

    def __aiter__(self) -> _AsyncProjectionIterator:
        """Iterate raw protocol events asynchronously."""
        return _AsyncProjectionIterator(self._events_proj)

    # -- Cleanup -----------------------------------------------------------

    async def aclose(self) -> None:
        """Cancel the background producer task and release resources.

        If a consumer cancels mid-stream or decides to stop iterating
        early, the producer task keeps pumping the provider HTTP call to
        completion because `asyncio.Task` has no implicit link to its
        awaiter. Call this method to cancel the producer explicitly; the
        stream transitions to an errored state with `CancelledError`.

        If the stream has already produced a message successfully (for
        example, after `await stream.output`), the producer may still be
        running post-stream work such as `on_llm_end` callbacks. In that
        case `aclose()` awaits the task rather than cancelling it —
        turning a successful run into a cancelled one would drop the
        end callback and corrupt tracing.

        Idempotent: safe to call multiple times, including after the
        stream has finished normally. Also invoked by the async context
        manager protocol on `__aexit__`.
        """
        if self._ensure_started is not None and self._producer_task is None:
            await self._ensure_started()

        task = self._producer_task
        if task is None:
            return
        if task.done() and self._done:
            return

        we_cancelled = not (self._output_message is not None and self._error is None)
        if we_cancelled and not task.done():
            task.cancel()

        # Wait for the task via a linked `Future`, not by awaiting the
        # task directly. Awaiting the task would raise `CancelledError`
        # in two indistinguishable cases: (1) the task we just cancelled
        # completed, (2) our caller cancelled us. `asyncio.Task.cancelling()`
        # disambiguates on 3.11+ but doesn't exist on 3.10.
        #
        # The `done_future` resolves with `None` whenever the task
        # finishes (any reason). It is not a `Task` itself, so its
        # `await` only raises when our caller is cancelled — giving us
        # a portable, unambiguous signal to propagate.
        if not task.done():
            loop = asyncio.get_running_loop()
            done_future: asyncio.Future[None] = loop.create_future()

            def _link(_: asyncio.Task[None]) -> None:
                if not done_future.done():
                    done_future.set_result(None)

            task.add_done_callback(_link)
            try:
                await done_future
            finally:
                task.remove_done_callback(_link)

        # If the task was cancelled before `_produce` ran (e.g.
        # `astream_events(version="v3")` immediately followed by `aclose()`), the stream
        # never reached `_produce`'s CancelledError handler — its
        # projections are still pending and no end-of-lifecycle callback
        # has fired. Resolve both here so callers of `await stream.output`
        # don't hang and tracing sees a matching end event.
        if we_cancelled and not self._done:
            cancel_exc = asyncio.CancelledError()
            self.fail(cancel_exc)
            teardown = self._on_aclose_fail
            if teardown is not None:
                with contextlib.suppress(Exception):
                    await teardown(cancel_exc)

    async def __aenter__(self) -> Self:
        """Enter the async context — returns self."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Exit the async context — cancels the producer via `aclose()`."""
        del exc_type, exc, tb
        await self.aclose()

    # -- Internal API (extend base to drive async projections) -------------

    def _record_event(self, event: Mapping[str, Any]) -> None:
        """Record event and push to async event replay projection."""
        super()._record_event(event)
        self._events_proj.push(cast("MessagesData", event))

    def _finish(self, data: MessageFinishData) -> None:
        """Finish base projections and async-only projections."""
        super()._finish(data)
        self._output_proj.complete(self._output_message)
        self._events_proj.complete(self._events)

    def fail(self, error: BaseException) -> None:
        """Fail base projections and async-only projections."""
        super().fail(error)
        self._output_proj.fail(error)
        self._events_proj.fail(error)


__all__ = [
    "AsyncChatModelStream",
    "AsyncProjection",
    "ChatModelStream",
    "SyncProjection",
    "SyncTextProjection",
]
