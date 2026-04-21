"""Compat bridge: convert `AIMessageChunk` streams to protocol events.

The bridge trusts :meth:`AIMessageChunk.content_blocks` as the single
protocol view of any chunk.  That property runs the three-tier lookup
(`output_version == "v1"` short-circuit, registered translator, or
best-effort parsing) and returns a `list[ContentBlock]` for every
well-formed message — whether the provider is a registered partner, an
unregistered community model, or not tagged at all.

Per-chunk `content_blocks` output is a **delta slice**, not accumulated
state: providers in this ecosystem emit SSE-style chunks that each carry
their own increment.  The bridge therefore forwards each slice straight
through as a `content-block-delta` event, and accumulates per-index
state only so the final `content-block-finish` event can report a
finalized block (e.g. `tool_call_chunk` args parsed to a dict).

Lifecycle::

    message-start
      -> content-block-start   (first time each index is observed)
      -> content-block-delta*  (per chunk, carrying the slice)
      -> content-block-finish  (finalized block)
    -> message-finish

Public API:

- :func:`chunks_to_events` / :func:`achunks_to_events` — for live streams
  where chunks arrive over time.
- :func:`message_to_events` / :func:`amessage_to_events` — for replaying a
  finalized :class:`AIMessage` (cache hit, checkpoint restore, graph-node
  return value) as a synthetic event lifecycle.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from langchain_protocol.protocol import (
    ContentBlock,
    ContentBlockDeltaData,
    ContentBlockFinishData,
    ContentBlockStartData,
    FinalizedContentBlock,
    FinishReason,
    InvalidToolCallBlock,
    MessageFinishData,
    MessageMetadata,
    MessagesData,
    MessageStartData,
    ReasoningBlock,
    ServerToolCallBlock,
    ServerToolCallChunkBlock,
    TextBlock,
    ToolCallBlock,
    ToolCallChunkBlock,
    UsageInfo,
)

from langchain_core.messages import AIMessageChunk, BaseMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.outputs import ChatGenerationChunk


CompatBlock = dict[str, Any]
"""Internal working type for a content block.

The bridge works with plain dicts internally because two separate but
structurally similar `ContentBlock` Unions exist — one in
:mod:`langchain_core.messages.content` (returned by
`msg.content_blocks`), one in :mod:`langchain_protocol.protocol` (the
wire/event shape).  They are not mypy-compatible despite being
near-isomorphic.  Passing through `dict[str, Any]` launders between
them.  See :func:`_to_protocol_block` for the single seam where the
laundering cast lives.
"""


# ---------------------------------------------------------------------------
# Type laundering between core and protocol `ContentBlock` unions
# ---------------------------------------------------------------------------


def _to_protocol_block(block: CompatBlock) -> ContentBlock:
    """Narrow an internal working dict to a protocol `ContentBlock`.

    Single seam between the two `ContentBlock` type systems:
    :mod:`langchain_core.messages.content` (what `msg.content_blocks`
    returns) and :mod:`langchain_protocol.protocol` (what event payloads
    require).  The two Unions overlap structurally but are nominally
    distinct to mypy, so we launder through `dict[str, Any]`.  When the
    Unions are unified, this helper and its finalized counterpart can be
    deleted.
    """
    return cast("ContentBlock", block)


def _to_finalized_block(block: CompatBlock) -> FinalizedContentBlock:
    """Counterpart of :func:`_to_protocol_block` for finalized blocks."""
    return cast("FinalizedContentBlock", block)


# ---------------------------------------------------------------------------
# Block iteration
# ---------------------------------------------------------------------------


def _iter_protocol_blocks(msg: BaseMessage) -> list[tuple[Any, CompatBlock]]:
    """Read per-chunk protocol blocks from `msg.content_blocks`.

    Returns `(key, block)` pairs.  The key is the block's stable identifier
    across the stream: the block's `index` field when present (can be an
    int or a string — some providers use string identifiers like
    `"lc_rs_305f30"`), or the positional index within the message as a
    fallback.  Callers are responsible for allocating wire-level `uint`
    indices; this helper only surfaces the source-side identity.

    For finalized :class:`AIMessage`, also surfaces `invalid_tool_calls`
    — which `AIMessage.content_blocks` currently omits from its return
    value even though they are a defined protocol block type.
    """
    try:
        raw = msg.content_blocks
    except Exception:
        return []

    result: list[tuple[Any, CompatBlock]] = []
    for i, block in enumerate(raw):
        if not isinstance(block, dict):
            continue
        key = block.get("index", i)
        result.append((key, dict(block)))

    if not isinstance(msg, AIMessageChunk):
        # Finalized AIMessage: pull invalid_tool_calls from the dedicated
        # field — AIMessage.content_blocks does not currently include them.
        for itc in getattr(msg, "invalid_tool_calls", None) or []:
            itc_block: CompatBlock = {"type": "invalid_tool_call"}
            for key_name in ("id", "name", "args", "error"):
                if itc.get(key_name) is not None:
                    itc_block[key_name] = itc[key_name]
            result.append((len(result), itc_block))

    return result


# ---------------------------------------------------------------------------
# Per-block helpers
# ---------------------------------------------------------------------------


# Fields that can carry large payloads (inline base64 media, parsed args,
# arbitrary dicts).  Stripped from `content-block-start` for self-contained
# block types so the payload rides on `content-block-finish` alone instead
# of being serialized twice on the wire.
_HEAVY_FIELDS = frozenset({"args", "data", "output", "transcript", "value"})


def _start_skeleton(block: CompatBlock) -> ContentBlock:
    """Empty-content placeholder for the `content-block-start` event.

    Deltaable block types (text, reasoning, the `_chunk` tool variants)
    get an empty payload so the lifecycle's "start" signal is distinct
    from the first incremental delta.  Self-contained types (image,
    audio, video, file, non_standard, finalized tool calls) drop their
    heavy payload fields; those are carried by `content-block-finish`.
    Correlation fields (id, name, toolCallId) and small metadata
    (mime_type, url, status, …) are preserved on the start event.
    """
    btype = block.get("type", "text")
    if btype == "text":
        return TextBlock(type="text", text="")
    if btype == "reasoning":
        return ReasoningBlock(type="reasoning", reasoning="")
    if btype == "tool_call_chunk":
        skel = ToolCallChunkBlock(type="tool_call_chunk", args="")
        if block.get("id") is not None:
            skel["id"] = block["id"]
        if block.get("name") is not None:
            skel["name"] = block["name"]
        return skel
    if btype == "server_tool_call_chunk":
        s_skel = ServerToolCallChunkBlock(
            type="server_tool_call_chunk",
            args="",
        )
        if block.get("id") is not None:
            s_skel["id"] = block["id"]
        if block.get("name") is not None:
            s_skel["name"] = block["name"]
        return s_skel

    stripped: CompatBlock = {k: v for k, v in block.items() if k not in _HEAVY_FIELDS}
    # Restore required-but-heavy fields with minimal placeholders so the
    # start event still validates against the CDDL shape of the block type.
    if btype in ("tool_call", "server_tool_call"):
        stripped["args"] = {}
    elif btype == "server_tool_call_result":
        stripped["output"] = None
    elif btype == "non_standard":
        stripped["value"] = {}
    return _to_protocol_block(stripped)


def _should_emit_delta(block: CompatBlock) -> bool:
    """Whether a per-chunk block carries content worth a delta event.

    Deltaable types emit only when they have fresh content.  Self-contained
    / already-finalized types skip the delta entirely — the `finish`
    event carries them.
    """
    btype = block.get("type")
    if btype == "text":
        return bool(block.get("text"))
    if btype == "reasoning":
        return bool(block.get("reasoning"))
    if btype in ("tool_call_chunk", "server_tool_call_chunk"):
        return bool(
            block.get("args") or block.get("id") or block.get("name"),
        )
    return False


def _accumulate(state: CompatBlock | None, delta: CompatBlock) -> CompatBlock:
    """Merge a per-chunk delta slice into accumulated per-index state.

    Used only for the finalization pass — live delta events are emitted
    directly from the per-chunk block, without round-tripping through
    accumulated state.
    """
    if state is None:
        return dict(delta)
    btype = state.get("type")
    dtype = delta.get("type")
    if btype == "text" and dtype == "text":
        state["text"] = state.get("text", "") + delta.get("text", "")
    elif btype == "reasoning" and dtype == "reasoning":
        state["reasoning"] = state.get("reasoning", "") + delta.get("reasoning", "")
    elif btype in ("tool_call_chunk", "server_tool_call_chunk") and dtype == btype:
        state["args"] = state.get("args", "") + (delta.get("args") or "")
        if delta.get("id") is not None:
            state["id"] = delta["id"]
        if delta.get("name") is not None:
            state["name"] = delta["name"]
    else:
        # Self-contained or already-finalized types: replace wholesale.
        state.clear()
        state.update(delta)
    return state


def _finalize_block(block: CompatBlock) -> FinalizedContentBlock:
    """Promote chunk variants to their finalized form.

    `tool_call_chunk` becomes `tool_call` — or `invalid_tool_call`
    if the accumulated `args` don't parse as JSON.
    `server_tool_call_chunk` becomes `server_tool_call` under the same
    rule.  Everything else passes through: text/reasoning blocks carry
    their accumulated snapshot, and self-contained types are already in
    their terminal shape.
    """
    btype = block.get("type")
    if btype in ("tool_call_chunk", "server_tool_call_chunk"):
        raw = block.get("args") or "{}"
        try:
            parsed = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            invalid = InvalidToolCallBlock(
                type="invalid_tool_call",
                args=raw,
                error="Failed to parse tool call arguments as JSON",
            )
            if block.get("id") is not None:
                invalid["id"] = block["id"]
            if block.get("name") is not None:
                invalid["name"] = block["name"]
            return invalid
        if btype == "tool_call_chunk":
            return ToolCallBlock(
                type="tool_call",
                id=block.get("id", ""),
                name=block.get("name", ""),
                args=parsed,
            )
        return ServerToolCallBlock(
            type="server_tool_call",
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=parsed,
        )
    return _to_finalized_block(block)


# ---------------------------------------------------------------------------
# Metadata, usage, finish-reason
# ---------------------------------------------------------------------------


def _extract_start_metadata(response_metadata: dict[str, Any]) -> MessageMetadata:
    """Pull provider/model hints for the `message-start` event."""
    metadata: MessageMetadata = {}
    if "model_provider" in response_metadata:
        metadata["provider"] = response_metadata["model_provider"]
    if "model_name" in response_metadata:
        metadata["model"] = response_metadata["model_name"]
    return metadata


def _normalize_finish_reason(value: Any) -> FinishReason:
    """Map provider-specific stop reasons to protocol finish reasons."""
    if value == "length":
        return "length"
    if value == "content_filter":
        return "content_filter"
    if value in ("tool_use", "tool_calls"):
        return "tool_use"
    return "stop"


def _accumulate_usage(
    current: dict[str, Any] | None, delta: Any
) -> dict[str, Any] | None:
    """Sum usage counts and merge detail dicts across chunks."""
    if not isinstance(delta, dict):
        return current
    if current is None:
        return dict(delta)
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
        if key in delta:
            current[key] = current.get(key, 0) + delta[key]
    for detail_key in ("input_token_details", "output_token_details"):
        if detail_key in delta and isinstance(delta[detail_key], dict):
            if detail_key not in current:
                current[detail_key] = {}
            current[detail_key].update(delta[detail_key])
    return current


def _to_protocol_usage(usage: dict[str, Any] | None) -> UsageInfo | None:
    """Convert accumulated usage to the protocol's `UsageInfo` shape."""
    if usage is None:
        return None
    result: UsageInfo = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
        if key in usage:
            result[key] = usage[key]
    return result or None


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def _build_message_start(
    msg: BaseMessage,
    message_id: str | None,
) -> MessageStartData:
    start_data = MessageStartData(event="message-start", role="ai")
    resolved_id = message_id if message_id is not None else getattr(msg, "id", None)
    if resolved_id:
        start_data["message_id"] = resolved_id
    start_metadata = _extract_start_metadata(msg.response_metadata or {})
    if start_metadata:
        start_data["metadata"] = start_metadata
    return start_data


def _build_message_finish(
    *,
    finish_reason: FinishReason,
    has_valid_tool_call: bool,
    usage: dict[str, Any] | None,
    response_metadata: dict[str, Any] | None,
) -> MessageFinishData:
    # Infer tool_use only from finalized (parsed) tool_calls.  An
    # invalid_tool_call means parsing failed — the model didn't
    # successfully request a tool, so leave finish_reason alone.
    if finish_reason == "stop" and has_valid_tool_call:
        finish_reason = "tool_use"
    finish_data = MessageFinishData(event="message-finish", reason=finish_reason)
    usage_info = _to_protocol_usage(usage)
    if usage_info is not None:
        finish_data["usage"] = usage_info
    if response_metadata:
        metadata = {
            k: v
            for k, v in response_metadata.items()
            if k not in ("finish_reason", "stop_reason")
        }
        if metadata:
            finish_data["metadata"] = metadata
    return finish_data


def _finalize_and_build_finish(
    wire_idx: int,
    block: CompatBlock,
) -> tuple[MessagesData, bool]:
    """Finalize a block and wrap it in a `content-block-finish` event.

    Returns the event plus a flag indicating whether the finalized block
    was a valid `tool_call` (used for finish-reason inference).
    """
    finalized = _finalize_block(block)
    has_valid_tool_call = finalized.get("type") == "tool_call"
    event = ContentBlockFinishData(
        event="content-block-finish",
        index=wire_idx,
        content_block=finalized,
    )
    return event, has_valid_tool_call


# ---------------------------------------------------------------------------
# Main generators
# ---------------------------------------------------------------------------


def chunks_to_events(
    chunks: Iterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Convert a stream of `ChatGenerationChunk` to protocol events.

    Blocks stream one at a time: when a chunk carries a different block
    identifier than the currently-open one, the open block is finished
    before the new block starts, matching the protocol's no-interleave
    rule.  Source-side identifiers (from the block's `index` field, which
    may be int or string) are translated to sequential `uint` wire
    indices.

    Args:
        chunks: Iterator of `ChatGenerationChunk` from `_stream()`.
        message_id: Optional stable message ID.

    Yields:
        `MessagesData` lifecycle events.
    """
    started = False
    open_key: Any = None
    open_block: CompatBlock | None = None
    open_wire_idx: int = 0
    next_wire_idx = 0
    has_valid_tool_call = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {}
    finish_reason: FinishReason = "stop"

    for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

        if msg.response_metadata:
            response_metadata.update(msg.response_metadata)

        if not started:
            started = True
            yield _build_message_start(msg, message_id)

        for key, block in _iter_protocol_blocks(msg):
            if key != open_key:
                if open_block is not None:
                    event, tc = _finalize_and_build_finish(open_wire_idx, open_block)
                    has_valid_tool_call = has_valid_tool_call or tc
                    yield event
                open_key = key
                open_wire_idx = next_wire_idx
                next_wire_idx += 1
                open_block = dict(block)
                yield ContentBlockStartData(
                    event="content-block-start",
                    index=open_wire_idx,
                    content_block=_start_skeleton(block),
                )
            else:
                open_block = _accumulate(open_block, block)
            if _should_emit_delta(block):
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=open_wire_idx,
                    content_block=_to_protocol_block(block),
                )

        if msg.usage_metadata:
            usage = _accumulate_usage(usage, msg.usage_metadata)

        rm = msg.response_metadata or {}
        raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
        if raw_reason:
            finish_reason = _normalize_finish_reason(raw_reason)

    if not started:
        return

    if open_block is not None:
        event, tc = _finalize_and_build_finish(open_wire_idx, open_block)
        has_valid_tool_call = has_valid_tool_call or tc
        yield event

    yield _build_message_finish(
        finish_reason=finish_reason,
        has_valid_tool_call=has_valid_tool_call,
        usage=usage,
        response_metadata=response_metadata,
    )


async def achunks_to_events(
    chunks: AsyncIterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async variant of :func:`chunks_to_events`."""
    started = False
    open_key: Any = None
    open_block: CompatBlock | None = None
    open_wire_idx: int = 0
    next_wire_idx = 0
    has_valid_tool_call = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {}
    finish_reason: FinishReason = "stop"

    async for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

        if msg.response_metadata:
            response_metadata.update(msg.response_metadata)

        if not started:
            started = True
            yield _build_message_start(msg, message_id)

        for key, block in _iter_protocol_blocks(msg):
            if key != open_key:
                if open_block is not None:
                    event, tc = _finalize_and_build_finish(open_wire_idx, open_block)
                    has_valid_tool_call = has_valid_tool_call or tc
                    yield event
                open_key = key
                open_wire_idx = next_wire_idx
                next_wire_idx += 1
                open_block = dict(block)
                yield ContentBlockStartData(
                    event="content-block-start",
                    index=open_wire_idx,
                    content_block=_start_skeleton(block),
                )
            else:
                open_block = _accumulate(open_block, block)
            if _should_emit_delta(block):
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=open_wire_idx,
                    content_block=_to_protocol_block(block),
                )

        if msg.usage_metadata:
            usage = _accumulate_usage(usage, msg.usage_metadata)

        rm = msg.response_metadata or {}
        raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
        if raw_reason:
            finish_reason = _normalize_finish_reason(raw_reason)

    if not started:
        return

    if open_block is not None:
        event, tc = _finalize_and_build_finish(open_wire_idx, open_block)
        has_valid_tool_call = has_valid_tool_call or tc
        yield event

    yield _build_message_finish(
        finish_reason=finish_reason,
        has_valid_tool_call=has_valid_tool_call,
        usage=usage,
        response_metadata=response_metadata,
    )


def message_to_events(
    msg: BaseMessage,
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Replay a finalized message as a synthetic event lifecycle.

    For a message returned whole (from a graph node, checkpoint, or
    cache), produce the same `message-start` / per-block /
    `message-finish` event stream a live call would produce.  Consumers
    downstream see a uniform event shape regardless of source.

    Text and reasoning blocks emit a single `content-block-delta` with
    the full accumulated content.  Already-finalized blocks (tool_call,
    server_tool_call, image, etc.) skip the delta and rely on the
    `content-block-finish` event alone.

    Args:
        msg: The finalized message — typically an `AIMessage`.
        message_id: Optional stable message ID; falls back to `msg.id`.

    Yields:
        `MessagesData` lifecycle events.
    """
    response_metadata = msg.response_metadata or {}
    yield _build_message_start(msg, message_id)

    has_valid_tool_call = False
    for wire_idx, (_key, block) in enumerate(_iter_protocol_blocks(msg)):
        yield ContentBlockStartData(
            event="content-block-start",
            index=wire_idx,
            content_block=_start_skeleton(block),
        )
        if _should_emit_delta(block):
            yield ContentBlockDeltaData(
                event="content-block-delta",
                index=wire_idx,
                content_block=_to_protocol_block(block),
            )
        finalized = _finalize_block(block)
        if finalized.get("type") == "tool_call":
            has_valid_tool_call = True
        yield ContentBlockFinishData(
            event="content-block-finish",
            index=wire_idx,
            content_block=finalized,
        )

    raw_reason = response_metadata.get("finish_reason") or response_metadata.get(
        "stop_reason"
    )
    finish_reason: FinishReason = (
        _normalize_finish_reason(raw_reason) if raw_reason else "stop"
    )
    yield _build_message_finish(
        finish_reason=finish_reason,
        has_valid_tool_call=has_valid_tool_call,
        usage=getattr(msg, "usage_metadata", None),
        response_metadata=response_metadata,
    )


async def amessage_to_events(
    msg: BaseMessage,
    *,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async variant of :func:`message_to_events`."""
    for event in message_to_events(msg, message_id=message_id):
        yield event


__all__ = [
    "CompatBlock",
    "achunks_to_events",
    "amessage_to_events",
    "chunks_to_events",
    "message_to_events",
]
