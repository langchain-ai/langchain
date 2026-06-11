"""Native content-block streaming-event converter for Perplexity.

Builds text and tool-call blocks directly from Perplexity's OpenAI-shaped
delta, feeding the shared `BlockStreamTracker`. Unlike the compat bridge (which
buries tool_calls in `additional_kwargs` and loses citations on the v3 path),
this surfaces tool calls as proper blocks and puts search extras (citations,
search_results, images, etc.) in `message-finish` `response_metadata`.

Usage is **cumulative**: each chunk's `usage` field is a running total, so the
message total is the value from the *last* chunk that carries usage — not an
accumulation across chunks.

Note: Perplexity reasoning is inline `<think>…</think>` text inside `content`;
there is no separate `reasoning_content` field, so it surfaces as plain text.

Citations and other search extras are emitted by the legacy `_stream` path via
`additional_kwargs`. On the v3 path `additional_kwargs` is dropped by the compat
bridge, so this converter puts them in `response_metadata` instead —
native v3 output is a strict superset of what the bridge provides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.stream_events import (
    BlockStreamTracker,
    build_message_finish,
)

from langchain_perplexity.chat_models import _create_usage_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_protocol.protocol import (
        MessageMetadata,
        MessagesData,
        MessageStartData,
    )


def _message_start(message_id: str | None, model: str | None) -> MessageStartData:
    # Do not use a provider completion id here: on the v3 path core seeds the
    # stream with the LangChain run id, and an empty id lets that stand
    # (matching the compat bridge). Only an explicit `message_id` wins.
    metadata: MessageMetadata = {"provider": "perplexity"}
    if model:
        metadata["model"] = model
    return {
        "event": "message-start",
        "role": "ai",
        "id": message_id or "",
        "metadata": metadata,
    }


def _feed_delta(
    tracker: BlockStreamTracker, delta: dict[str, Any]
) -> Iterator[MessagesData]:
    """Yield block events for one OpenAI-shaped Perplexity delta."""
    if content := delta.get("content"):
        yield from tracker.feed("text", {"type": "text", "text": content})
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        yield from tracker.feed(
            f"tool:{idx}",
            {
                "type": "tool_call_chunk",
                "id": tc.get("id"),
                "name": fn.get("name"),
                "args": args or "",
                "index": idx,
            },
        )


def _collect_extras(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build response_metadata extras from the first chunk.

    Mirrors the first-chunk block in `_stream`: always includes `citations`
    (default `[]`), the present-keyed `images`/`related_questions`/
    `search_results`, and the truthy-keyed `videos`/`reasoning_steps`.
    """
    extras: dict[str, Any] = {"citations": chunk.get("citations", [])}
    for key in ("images", "related_questions", "search_results"):
        if key in chunk:
            extras[key] = chunk[key]
    for key in ("videos", "reasoning_steps"):
        if chunk.get(key):
            extras[key] = chunk[key]
    return extras


def convert_perplexity_stream(
    raw: Iterator[Any], *, message_id: str | None = None
) -> Iterator[MessagesData]:
    """Convert a raw Perplexity chat stream to protocol events.

    Args:
        raw: Raw Perplexity chat-completion stream chunks (dicts or SDK objects).
        message_id: Overrides the provider message id on `message-start`.

    Yields:
        Protocol `MessagesData` lifecycle events.
    """
    tracker = BlockStreamTracker()
    started = False
    latest_usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "perplexity"}
    model: str | None = None
    first_chunk = True

    for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None:
            model = chunk.get("model")
        if usage := chunk.get("usage"):
            # Usage is cumulative; track the latest total — do NOT accumulate.
            latest_usage = dict(_create_usage_metadata(usage))
            if "num_search_queries" not in response_metadata:
                if num_sq := usage.get("num_search_queries"):
                    response_metadata["num_search_queries"] = num_sq
            if "search_context_size" not in response_metadata:
                if scs := usage.get("search_context_size"):
                    response_metadata["search_context_size"] = scs
        choices = chunk.get("choices") or []
        if len(choices) == 0:
            continue
        if first_chunk:
            response_metadata.update(_collect_extras(chunk))
            first_chunk = False
        if not started:
            started = True
            yield _message_start(message_id, model)
        choice = choices[0]
        yield from _feed_delta(tracker, choice.get("delta") or {})
        if finish_reason := choice.get("finish_reason"):
            response_metadata["finish_reason"] = finish_reason

    if not started:
        return
    yield from tracker.finish_all()
    yield build_message_finish(usage=latest_usage, response_metadata=response_metadata)


async def aconvert_perplexity_stream(
    raw: AsyncIterator[Any], *, message_id: str | None = None
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_perplexity_stream`."""
    tracker = BlockStreamTracker()
    started = False
    latest_usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "perplexity"}
    model: str | None = None
    first_chunk = True

    async for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None:
            model = chunk.get("model")
        if usage := chunk.get("usage"):
            # Usage is cumulative; track the latest total — do NOT accumulate.
            latest_usage = dict(_create_usage_metadata(usage))
            if "num_search_queries" not in response_metadata:
                if num_sq := usage.get("num_search_queries"):
                    response_metadata["num_search_queries"] = num_sq
            if "search_context_size" not in response_metadata:
                if scs := usage.get("search_context_size"):
                    response_metadata["search_context_size"] = scs
        choices = chunk.get("choices") or []
        if len(choices) == 0:
            continue
        if first_chunk:
            response_metadata.update(_collect_extras(chunk))
            first_chunk = False
        if not started:
            started = True
            yield _message_start(message_id, model)
        choice = choices[0]
        for ev in _feed_delta(tracker, choice.get("delta") or {}):
            yield ev
        if finish_reason := choice.get("finish_reason"):
            response_metadata["finish_reason"] = finish_reason

    if not started:
        return
    for ev in tracker.finish_all():
        yield ev
    yield build_message_finish(usage=latest_usage, response_metadata=response_metadata)
