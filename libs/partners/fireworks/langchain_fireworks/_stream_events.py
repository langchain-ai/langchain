"""Native content-block streaming-event converter for Fireworks.

Builds text, reasoning, and tool-call blocks directly from Fireworks' raw
OpenAI-shaped delta (Fireworks streams tool-call args incrementally), feeding
the shared `BlockStreamTracker`. Unlike the compat bridge (which leaves
Fireworks' `tool_calls`/`reasoning_content` in `additional_kwargs`), this
surfaces them as first-class blocks.

Reasoning delta key is ``reasoning_content`` (not ``reasoning``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.stream_events import (
    BlockStreamTracker,
    accumulate_usage,
    build_message_finish,
)

from langchain_fireworks.chat_models import _usage_to_metadata

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
    metadata: MessageMetadata = {"provider": "fireworks"}
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
    """Yield block events for one OpenAI-shaped Fireworks delta."""
    if reasoning := delta.get("reasoning_content"):
        yield from tracker.feed(
            "reasoning", {"type": "reasoning", "reasoning": reasoning}
        )
    if content := delta.get("content"):
        yield from tracker.feed("text", {"type": "text", "text": content})
    # Deferred-as-scope-cut: function_call (legacy) and logprobs are not
    # surfaced as blocks; the bridge already drops them.
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args == "null":  # normalize JSON null for no-arg tools
            args = "{}"
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


def convert_fireworks_stream(
    raw: Iterator[Any], *, message_id: str | None = None
) -> Iterator[MessagesData]:
    """Convert a raw Fireworks chat stream to protocol events.

    Args:
        raw: Raw Fireworks chat-completion stream chunks (dicts or SDK objects).
        message_id: Overrides the provider message id on `message-start`.

    Yields:
        Protocol `MessagesData` lifecycle events.
    """
    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "fireworks"}
    model: str | None = None

    for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None:
            model = chunk.get("model")
        if service_tier := chunk.get("service_tier"):
            response_metadata["service_tier"] = service_tier
        # Usage arrives on the final no-choices chunk when
        # `stream_options.include_usage=True`; also tolerate it on other chunks.
        if fw_usage := chunk.get("usage"):
            usage = accumulate_usage(usage, dict(_usage_to_metadata(fw_usage)))
        choices = chunk.get("choices") or []
        if len(choices) == 0:
            continue
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
    yield build_message_finish(usage=usage, response_metadata=response_metadata)


async def aconvert_fireworks_stream(
    raw: AsyncIterator[Any], *, message_id: str | None = None
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_fireworks_stream`."""
    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "fireworks"}
    model: str | None = None

    async for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None:
            model = chunk.get("model")
        if service_tier := chunk.get("service_tier"):
            response_metadata["service_tier"] = service_tier
        if fw_usage := chunk.get("usage"):
            usage = accumulate_usage(usage, dict(_usage_to_metadata(fw_usage)))
        choices = chunk.get("choices") or []
        if len(choices) == 0:
            continue
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
    yield build_message_finish(usage=usage, response_metadata=response_metadata)
