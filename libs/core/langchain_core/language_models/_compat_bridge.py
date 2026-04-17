"""Compat bridge: convert ``AIMessageChunk`` streams to protocol events.

This module extracts the chunk-to-event conversion logic so it can be used
both by ``BaseChatModel.stream_v2()`` (as the default compat bridge for
providers that only implement ``_stream()``) and by LangGraph's
``StreamProtocolMessagesHandler``.

The higher-level generators :func:`chunks_to_events` and
:func:`achunks_to_events` manage the full message lifecycle::

    message-start
      -> content-block-start  (per block)
      -> content-block-delta* (per block, per chunk)
      -> content-block-finish (per block)
    -> message-finish
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
    TextBlock,
    ToolCallBlock,
    ToolCallChunkBlock,
    UsageInfo,
)

from langchain_core.messages import AIMessageChunk, BaseMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.outputs import ChatGenerationChunk

# A "compatible content block" is a plain dict matching one of the protocol
# block TypedDicts (TextBlock, ReasoningBlock, ToolCallChunkBlock, etc.).
CompatBlock = dict[str, Any]

# Protocol block types that arrive as a single fully-populated block rather
# than through incremental deltas. Accumulation replaces wholesale; a single
# delta event carries the full block.
_SELF_CONTAINED_BLOCK_TYPES = frozenset(
    {
        "server_tool_call",
        "server_tool_call_result",
        "invalid_tool_call",
        "image",
        "audio",
        "video",
        "file",
        "non_standard",
    }
)

# Protocol block types the extractor passes through verbatim when encountered
# in ``msg.content`` as a protocol-shape dict. Text, reasoning and tool-call
# fields are handled by dedicated extraction paths above; everything else
# flows through unchanged so the stream can accumulate and finalize it.
_PROTOCOL_PASS_THROUGH_TYPES = frozenset(
    {
        "tool_call",
        "tool_call_chunk",
        "invalid_tool_call",
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_call_result",
        "image",
        "audio",
        "video",
        "file",
        "non_standard",
    }
)


# ---------------------------------------------------------------------------
# Content-block accumulation helpers
# ---------------------------------------------------------------------------


def _accumulate_block(accumulated: CompatBlock, delta: CompatBlock) -> CompatBlock:
    """Merge *delta* into *accumulated*, returning the updated block.

    Deltaable types (text / reasoning / ``_chunk`` variants) concatenate
    their streamed field; non-deltaable types (media / server-tool results /
    non_standard) pass through as the latest value since a single emission
    carries the full block.
    """
    btype = accumulated.get("type", "text")
    if btype == "text" and delta.get("type", "text") == "text":
        accumulated["text"] = accumulated.get("text", "") + delta.get("text", "")
    elif btype == "reasoning" and delta.get("type") == "reasoning":
        accumulated["reasoning"] = accumulated.get("reasoning", "") + delta.get(
            "reasoning", ""
        )
    elif btype in ("tool_call_chunk", "server_tool_call_chunk") and delta.get(
        "type"
    ) == btype:
        accumulated["args"] = accumulated.get("args", "") + delta.get("args", "")
        if delta.get("id") is not None:
            accumulated["id"] = delta["id"]
        if delta.get("name") is not None:
            accumulated["name"] = delta["name"]
    elif btype in _SELF_CONTAINED_BLOCK_TYPES:
        # Self-contained block types — replace wholesale rather than merge.
        accumulated.clear()
        accumulated.update(delta)
    return accumulated


def _delta_block(previous: CompatBlock, current: CompatBlock) -> ContentBlock | None:
    """Compute the delta between *previous* and *current*.

    Returns ``None`` if there is nothing new to emit.
    """
    btype = current.get("type", "text")
    if btype == "text":
        prev_text = previous.get("text", "")
        cur_text = current.get("text", "")
        delta_text = cur_text[len(prev_text) :]
        if not delta_text:
            return None
        return TextBlock(type="text", text=delta_text)
    if btype == "reasoning":
        prev_r = previous.get("reasoning", "")
        cur_r = current.get("reasoning", "")
        delta_r = cur_r[len(prev_r) :]
        if not delta_r:
            return None
        return ReasoningBlock(type="reasoning", reasoning=delta_r)
    if btype == "tool_call_chunk":
        prev_args = previous.get("args", "")
        cur_args = current.get("args", "")
        delta_args = cur_args[len(prev_args) :]
        has_meta = current.get("id") is not None or current.get("name") is not None
        if not delta_args and not has_meta:
            return None
        chunk = ToolCallChunkBlock(type="tool_call_chunk", args=delta_args)
        if current.get("id") is not None and previous.get("id") is None:
            chunk["id"] = current["id"]
        if current.get("name") is not None and previous.get("name") is None:
            chunk["name"] = current["name"]
        return chunk
    if btype == "server_tool_call_chunk":
        prev_args = previous.get("args", "")
        cur_args = current.get("args", "")
        delta_args = cur_args[len(prev_args) :]
        has_meta = current.get("id") is not None or current.get("name") is not None
        if not delta_args and not has_meta:
            return None
        s_chunk: CompatBlock = {
            "type": "server_tool_call_chunk",
            "args": delta_args,
        }
        if current.get("id") is not None and previous.get("id") is None:
            s_chunk["id"] = current["id"]
        if current.get("name") is not None and previous.get("name") is None:
            s_chunk["name"] = current["name"]
        return cast("ContentBlock", s_chunk)
    if btype in _SELF_CONTAINED_BLOCK_TYPES:
        # Self-contained blocks: emit once on transition from skeleton
        # (type-only) start to populated form; subsequent identical
        # accumulations suppress.
        if len(previous) <= 1:
            return cast("ContentBlock", dict(current))
        return None
    # Unrecognized block type — pass through unchanged.  Caller is
    # responsible for ensuring the dict matches a valid ``ContentBlock``.
    return cast("ContentBlock", current)


def _finalize_block(block: CompatBlock) -> FinalizedContentBlock:
    """Promote chunk variants to their finalized form.

    ``tool_call_chunk`` becomes ``tool_call`` (or ``invalid_tool_call`` if
    JSON parsing fails), and ``server_tool_call_chunk`` becomes
    ``server_tool_call`` under the same rule. Non-chunk blocks are already
    finalized and pass through unchanged.
    """
    btype = block.get("type")
    if btype == "tool_call_chunk":
        raw_args = block.get("args", "{}")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            invalid = InvalidToolCallBlock(
                type="invalid_tool_call",
                args=raw_args,
                error="Failed to parse tool call arguments as JSON",
            )
            if block.get("id") is not None:
                invalid["id"] = block["id"]
            if block.get("name") is not None:
                invalid["name"] = block["name"]
            return invalid
        return ToolCallBlock(
            type="tool_call",
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=parsed_args,
        )
    if btype == "server_tool_call_chunk":
        raw_args = block.get("args", "{}")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            s_invalid = InvalidToolCallBlock(
                type="invalid_tool_call",
                args=raw_args,
                error="Failed to parse tool call arguments as JSON",
            )
            if block.get("id") is not None:
                s_invalid["id"] = block["id"]
            if block.get("name") is not None:
                s_invalid["name"] = block["name"]
            return s_invalid
        return cast(
            "FinalizedContentBlock",
            {
                "type": "server_tool_call",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "args": parsed_args,
            },
        )
    # Already a finalized block variant — pass through.
    return cast("FinalizedContentBlock", block)


def _make_start_block(block: CompatBlock) -> ContentBlock:
    """Create an empty start placeholder for a content block."""
    btype = block.get("type", "text")
    if btype == "text":
        return TextBlock(type="text", text="")
    if btype == "reasoning":
        return ReasoningBlock(type="reasoning", reasoning="")
    if btype == "tool_call_chunk":
        chunk = ToolCallChunkBlock(type="tool_call_chunk", args="")
        if "id" in block:
            chunk["id"] = block["id"]
        if "name" in block:
            chunk["name"] = block["name"]
        return chunk
    if btype == "server_tool_call_chunk":
        s_chunk: CompatBlock = {"type": "server_tool_call_chunk", "args": ""}
        if "id" in block:
            s_chunk["id"] = block["id"]
        if "name" in block:
            s_chunk["name"] = block["name"]
        return cast("ContentBlock", s_chunk)
    if btype == "tool_call":
        # Already finalized — return as-is for start event
        return ToolCallBlock(
            type="tool_call",
            id=block.get("id", ""),
            name=block.get("name", ""),
            args=block.get("args", {}),
        )
    if btype in _SELF_CONTAINED_BLOCK_TYPES:
        # Emit a type-only skeleton so the first delta can carry the block's
        # populated content without looking like a no-op.
        return cast("ContentBlock", {"type": btype})
    # Any other recognized ContentBlock variant — pass through unchanged.
    return cast("ContentBlock", block)


def _extract_start_metadata(response_metadata: dict[str, Any]) -> MessageMetadata:
    """Extract provider/model metadata for the ``message-start`` event."""
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
    # "end_turn", "stop", None, and anything else -> "stop"
    return "stop"


def _accumulate_usage(
    current: dict[str, Any] | None, delta: Any
) -> dict[str, Any] | None:
    """Accumulate usage metadata from streamed chunks."""
    if not isinstance(delta, dict):
        return current
    if current is None:
        return dict(delta)
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
        if key in delta:
            current[key] = current.get(key, 0) + delta[key]
    # Merge detail dicts
    for detail_key in ("input_token_details", "output_token_details"):
        if detail_key in delta and isinstance(delta[detail_key], dict):
            if detail_key not in current:
                current[detail_key] = {}
            current[detail_key].update(delta[detail_key])
    return current


def _to_protocol_usage(usage: dict[str, Any] | None) -> UsageInfo | None:
    """Convert LangChain usage metadata to protocol ``UsageInfo``."""
    if usage is None:
        return None
    result: UsageInfo = {}
    if "input_tokens" in usage:
        result["input_tokens"] = usage["input_tokens"]
    if "output_tokens" in usage:
        result["output_tokens"] = usage["output_tokens"]
    if "total_tokens" in usage:
        result["total_tokens"] = usage["total_tokens"]
    if "cached_tokens" in usage:
        result["cached_tokens"] = usage["cached_tokens"]
    return result or None


# ---------------------------------------------------------------------------
# Extracting content blocks from LangChain messages
# ---------------------------------------------------------------------------


def _extract_blocks_from_chunk(msg: AIMessageChunk) -> list[tuple[int, CompatBlock]]:
    """Extract ``(index, block)`` pairs from an ``AIMessageChunk``.

    LangChain stores content in several places:
    - ``content: str`` — a single text block at index 0
    - ``content: list[dict]`` — explicit content blocks with their own types
    - ``tool_call_chunks`` — separate list for streamed tool call deltas
    """
    blocks: list[tuple[int, CompatBlock]] = []
    content = msg.content
    if isinstance(content, str) and content:
        blocks.append((0, dict(TextBlock(type="text", text=content))))
    elif isinstance(content, list):
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "")
            if ctype == "text" and item.get("text"):
                blocks.append(
                    (
                        item.get("index", i),
                        dict(TextBlock(type="text", text=item["text"])),
                    )
                )
            elif ctype in ("reasoning_content", "reasoning", "thinking"):
                reasoning_text = (
                    item.get("reasoning_content")
                    or item.get("reasoning")
                    or item.get("thinking", "")
                )
                if reasoning_text:
                    blocks.append(
                        (
                            item.get("index", i),
                            dict(
                                ReasoningBlock(
                                    type="reasoning", reasoning=reasoning_text
                                )
                            ),
                        )
                    )
            elif ctype in _PROTOCOL_PASS_THROUGH_TYPES:
                blocks.append((item.get("index", i), dict(item)))

    # Tool call chunks live in a separate field
    for tc in msg.tool_call_chunks or []:
        idx = tc.get("index")
        if idx is None:
            # Assign indices after text content blocks
            idx = len(blocks)
        block: CompatBlock = {"type": "tool_call_chunk", "args": tc.get("args", "")}
        if tc.get("id") is not None:
            block["id"] = tc["id"]
        if tc.get("name") is not None:
            block["name"] = tc["name"]
        blocks.append((idx, block))

    return blocks


def _extract_final_blocks(msg: BaseMessage) -> list[tuple[int, CompatBlock]]:
    """Extract ``(index, block)`` pairs from a finalized ``AIMessage``."""
    blocks: list[tuple[int, CompatBlock]] = []
    content = msg.content

    if isinstance(content, str) and content:
        blocks.append((0, dict(TextBlock(type="text", text=content))))
    elif isinstance(content, list):
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            ctype = item.get("type", "")
            if ctype == "text" and item.get("text"):
                blocks.append((i, dict(TextBlock(type="text", text=item["text"]))))
            elif ctype in ("reasoning_content", "reasoning", "thinking"):
                reasoning_text = (
                    item.get("reasoning_content")
                    or item.get("reasoning")
                    or item.get("thinking", "")
                )
                if reasoning_text:
                    blocks.append(
                        (
                            i,
                            dict(
                                ReasoningBlock(
                                    type="reasoning", reasoning=reasoning_text
                                )
                            ),
                        )
                    )
            elif ctype in _PROTOCOL_PASS_THROUGH_TYPES:
                blocks.append((i, dict(item)))

    # Finalized tool calls (already parsed, not chunks)
    for tc in getattr(msg, "tool_calls", None) or []:
        idx = len(blocks)
        blocks.append(
            (
                idx,
                dict(
                    ToolCallBlock(
                        type="tool_call",
                        id=tc.get("id", ""),
                        name=tc.get("name", ""),
                        args=tc.get("args", {}),
                    )
                ),
            )
        )

    # Standard langchain invalid_tool_calls field
    for itc in getattr(msg, "invalid_tool_calls", None) or []:
        idx = len(blocks)
        block: CompatBlock = {"type": "invalid_tool_call"}
        for k in ("id", "name", "args", "error"):
            if itc.get(k) is not None:
                block[k] = itc[k]
        blocks.append((idx, block))

    return blocks


# ---------------------------------------------------------------------------
# High-level generators: full message lifecycle
# ---------------------------------------------------------------------------


def _process_chunk(
    msg: AIMessageChunk,
    blocks: dict[int, CompatBlock],
    usage: dict[str, Any] | None,
) -> tuple[list[MessagesData], dict[str, Any] | None]:
    """Process a single chunk, updating block state and returning events."""
    events: list[MessagesData] = []

    extracted = _extract_blocks_from_chunk(msg)
    for idx, delta_block in extracted:
        if idx not in blocks:
            # New block — emit content-block-start
            blocks[idx] = dict(delta_block)
            start_block = _make_start_block(delta_block)
            events.append(
                ContentBlockStartData(
                    event="content-block-start",
                    index=idx,
                    content_block=start_block,
                )
            )
            # Then emit the first delta
            first_delta = _delta_block(
                dict(_make_start_block(delta_block)), blocks[idx]
            )
            if first_delta is not None:
                events.append(
                    ContentBlockDeltaData(
                        event="content-block-delta",
                        index=idx,
                        content_block=first_delta,
                    )
                )
        else:
            # Existing block — compute delta, accumulate, emit
            previous = dict(blocks[idx])
            blocks[idx] = _accumulate_block(blocks[idx], delta_block)
            delta = _delta_block(previous, blocks[idx])
            if delta is not None:
                events.append(
                    ContentBlockDeltaData(
                        event="content-block-delta",
                        index=idx,
                        content_block=delta,
                    )
                )

    # Accumulate usage from chunk
    if msg.usage_metadata:
        usage = _accumulate_usage(usage, msg.usage_metadata)

    return events, usage


def _finish_events(
    blocks: dict[int, CompatBlock],
    usage: dict[str, Any] | None,
    finish_reason: FinishReason,
    response_metadata: dict[str, Any] | None = None,
) -> list[MessagesData]:
    """Generate content-block-finish and message-finish events."""
    events: list[MessagesData] = []

    has_valid_tool_call = False
    for idx in sorted(blocks):
        finalized = _finalize_block(blocks[idx])
        if finalized.get("type") == "tool_call":
            has_valid_tool_call = True
        events.append(
            ContentBlockFinishData(
                event="content-block-finish",
                index=idx,
                content_block=finalized,
            )
        )

    # Infer tool_use only from finalized (successfully parsed) tool calls.
    # An invalid_tool_call means JSON parsing failed — the model didn't
    # successfully request a tool, so keep finish_reason as-is.
    if finish_reason == "stop" and has_valid_tool_call:
        finish_reason = "tool_use"

    finish_data = MessageFinishData(event="message-finish", reason=finish_reason)
    usage_info = _to_protocol_usage(usage)
    if usage_info is not None:
        finish_data["usage"] = usage_info
    # Preserve response_metadata for v1 parity
    if response_metadata:
        # Exclude keys already represented in the protocol event
        metadata = {
            k: v
            for k, v in response_metadata.items()
            if k not in ("finish_reason", "stop_reason")
        }
        if metadata:
            finish_data["metadata"] = metadata
    events.append(finish_data)

    return events


def chunks_to_events(
    chunks: Iterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Convert a stream of ``ChatGenerationChunk`` to protocol events.

    Manages the full message lifecycle::

        message-start
          -> content-block-start  (per block)
          -> content-block-delta* (per block, per chunk)
          -> content-block-finish (per block)
        -> message-finish

    Args:
        chunks: Iterator of ``ChatGenerationChunk`` from ``_stream()``.
        message_id: Optional stable message ID.

    Yields:
        ``MessagesData`` events.
    """
    blocks: dict[int, CompatBlock] = {}
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {}
    started = False
    finish_reason: FinishReason = "stop"

    for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

        # Accumulate response_metadata across chunks
        if msg.response_metadata:
            response_metadata.update(msg.response_metadata)

        # Emit message-start on first token
        if not started:
            started = True
            start_data = MessageStartData(event="message-start", role="ai")
            if message_id:
                start_data["message_id"] = message_id
            elif msg.id:
                start_data["message_id"] = msg.id
            # Populate metadata from response_metadata for v1 parity
            start_metadata = _extract_start_metadata(msg.response_metadata or {})
            if start_metadata:
                start_data["metadata"] = start_metadata
            yield start_data

        # Process chunk content
        events, usage = _process_chunk(msg, blocks, usage)
        yield from events

        # Extract finish reason from response_metadata if present
        rm = msg.response_metadata or {}
        raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
        if raw_reason:
            finish_reason = _normalize_finish_reason(raw_reason)

    if not started:
        # No chunks at all — nothing to emit
        return

    # tool_use inference happens inside _finish_events based on the
    # *finalized* block types — so invalid_tool_call doesn't count.
    yield from _finish_events(blocks, usage, finish_reason, response_metadata)


async def achunks_to_events(
    chunks: AsyncIterator[ChatGenerationChunk],
    *,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async variant of :func:`chunks_to_events`."""
    blocks: dict[int, CompatBlock] = {}
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {}
    started = False
    finish_reason: FinishReason = "stop"

    async for chunk in chunks:
        msg = chunk.message
        if not isinstance(msg, AIMessageChunk):
            continue

        # Accumulate response_metadata across chunks
        if msg.response_metadata:
            response_metadata.update(msg.response_metadata)

        # Emit message-start on first token
        if not started:
            started = True
            start_data = MessageStartData(event="message-start", role="ai")
            if message_id:
                start_data["message_id"] = message_id
            elif msg.id:
                start_data["message_id"] = msg.id
            start_metadata = _extract_start_metadata(msg.response_metadata or {})
            if start_metadata:
                start_data["metadata"] = start_metadata
            yield start_data

        # Process chunk content
        events, usage = _process_chunk(msg, blocks, usage)
        for event in events:
            yield event

        # Extract finish reason from response_metadata if present
        rm = msg.response_metadata or {}
        raw_reason = rm.get("finish_reason") or rm.get("stop_reason")
        if raw_reason:
            finish_reason = _normalize_finish_reason(raw_reason)

    if not started:
        return

    # tool_use inference happens inside _finish_events based on the
    # *finalized* block types — so invalid_tool_call doesn't count.
    for event in _finish_events(blocks, usage, finish_reason, response_metadata):
        yield event


def message_to_events(
    msg: BaseMessage,
    *,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Replay a finalized message as a synthetic event lifecycle.

    Converts an already-complete message (e.g. one returned whole from a
    graph node, loaded from checkpoint state, or fetched from cache) into
    the same ``message-start`` / per-block / ``message-finish`` event
    stream a live call would produce. Consumers downstream see a uniform
    event shape regardless of source.

    Text and reasoning blocks emit a single ``content-block-delta`` with
    the full content so projections like ``.text`` pick up the value.
    Tool-call blocks skip the delta (finalized ``args`` are a dict, not a
    chunk string) and rely on the finish event alone.

    Args:
        msg: The finalized message — typically an ``AIMessage``.
        message_id: Optional stable message ID; falls back to ``msg.id``.

    Yields:
        ``MessagesData`` lifecycle events.
    """
    blocks = _extract_final_blocks(msg)

    start_data = MessageStartData(event="message-start", role="ai")
    resolved_id = message_id if message_id is not None else msg.id
    if resolved_id:
        start_data["message_id"] = resolved_id
    response_metadata = msg.response_metadata or {}
    start_metadata = _extract_start_metadata(response_metadata)
    if start_metadata:
        start_data["metadata"] = start_metadata
    yield start_data

    has_valid_tool_call = False
    for idx, block in blocks:
        yield ContentBlockStartData(
            event="content-block-start",
            index=idx,
            content_block=_make_start_block(block),
        )
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=idx,
                    content_block=TextBlock(type="text", text=text),
                )
        elif btype == "reasoning":
            reasoning = block.get("reasoning", "")
            if reasoning:
                yield ContentBlockDeltaData(
                    event="content-block-delta",
                    index=idx,
                    content_block=ReasoningBlock(type="reasoning", reasoning=reasoning),
                )
        finalized = _finalize_block(block)
        if finalized.get("type") == "tool_call":
            has_valid_tool_call = True
        yield ContentBlockFinishData(
            event="content-block-finish",
            index=idx,
            content_block=finalized,
        )

    raw_reason = response_metadata.get("finish_reason") or response_metadata.get(
        "stop_reason"
    )
    finish_reason: FinishReason = (
        _normalize_finish_reason(raw_reason) if raw_reason else "stop"
    )
    if finish_reason == "stop" and has_valid_tool_call:
        finish_reason = "tool_use"

    finish_data = MessageFinishData(event="message-finish", reason=finish_reason)
    usage_info = _to_protocol_usage(getattr(msg, "usage_metadata", None))
    if usage_info is not None:
        finish_data["usage"] = usage_info
    metadata = {
        k: v
        for k, v in response_metadata.items()
        if k not in ("finish_reason", "stop_reason")
    }
    if metadata:
        finish_data["metadata"] = metadata
    yield finish_data


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
