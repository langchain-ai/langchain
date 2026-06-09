"""Native content-block streaming-event converter for Anthropic.

Maps the raw Anthropic SDK stream (``message_start`` /
``content_block_start`` / ``content_block_delta`` / ``content_block_stop`` /
``message_delta`` / ``message_stop``) to the protocol ``MessagesData``
event lifecycle, reusing the shared ``BlockStreamTracker`` for block
mechanics and the existing per-event chunk builder for content extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.stream_events import (
    BlockStreamTracker,
    accumulate_usage,
    build_message_finish,
    iter_protocol_blocks,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.messages import AIMessageChunk
    from langchain_protocol.protocol import (
        MessageMetadata,
        MessagesData,
        MessageStartData,
    )

# The bound method ``ChatAnthropic._make_message_chunk_from_anthropic_event``.
MakeChunk = Callable[..., "tuple[AIMessageChunk | None, Any]"]


def _message_start(event: Any, message_id: str | None) -> MessageStartData:
    msg_obj = getattr(event, "message", None)
    # Do not use the provider message id (`msg_...`) here: on the v3 path core
    # seeds the stream with the LangChain run id, and an empty id lets that
    # stand (matching the compat bridge). Only an explicit `message_id` wins.
    resolved_id = message_id or ""
    metadata: MessageMetadata = {"provider": "anthropic"}
    model = getattr(msg_obj, "model", None)
    if model:
        metadata["model"] = model
    return {
        "event": "message-start",
        "role": "ai",
        "id": resolved_id,
        "metadata": metadata,
    }


def convert_anthropic_stream(
    raw: Iterator[Any],
    make_chunk: MakeChunk,
    *,
    stream_usage: bool = True,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Convert a raw Anthropic event stream to protocol events.

    Args:
        raw: Raw Anthropic SDK stream events.
        make_chunk: ``ChatAnthropic._make_message_chunk_from_anthropic_event``,
            injected so the converter stays pure and unit-testable.
        stream_usage: Forwarded to ``make_chunk``.
        message_id: Overrides the provider message id on ``message-start``.

    Yields:
        Protocol ``MessagesData`` lifecycle events.
    """
    tracker = BlockStreamTracker()
    started = False
    block_start_event: Any = None
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "anthropic"}

    for event in raw:
        etype = getattr(event, "type", None)
        chunk_msg, block_start_event = make_chunk(
            event,
            stream_usage=stream_usage,
            coerce_content_to_string=False,
            block_start_event=block_start_event,
        )
        if not started:
            started = True
            yield _message_start(event, message_id)
        if chunk_msg is not None:
            for key, block in iter_protocol_blocks(chunk_msg):
                yield from tracker.feed(key, block)
            if chunk_msg.usage_metadata:
                usage = accumulate_usage(usage, chunk_msg.usage_metadata)
            if chunk_msg.response_metadata:
                response_metadata.update(chunk_msg.response_metadata)
        if etype == "content_block_stop":
            yield from tracker.finish_block(getattr(event, "index", None))

    if not started:
        return
    yield from tracker.finish_all()
    yield build_message_finish(usage=usage, response_metadata=response_metadata)


async def aconvert_anthropic_stream(
    raw: AsyncIterator[Any],
    make_chunk: MakeChunk,
    *,
    stream_usage: bool = True,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_anthropic_stream`. `make_chunk` is sync."""
    tracker = BlockStreamTracker()
    started = False
    block_start_event: Any = None
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "anthropic"}

    async for event in raw:
        etype = getattr(event, "type", None)
        chunk_msg, block_start_event = make_chunk(
            event,
            stream_usage=stream_usage,
            coerce_content_to_string=False,
            block_start_event=block_start_event,
        )
        if not started:
            started = True
            yield _message_start(event, message_id)
        if chunk_msg is not None:
            for key, block in iter_protocol_blocks(chunk_msg):
                for ev in tracker.feed(key, block):
                    yield ev
            if chunk_msg.usage_metadata:
                usage = accumulate_usage(usage, chunk_msg.usage_metadata)
            if chunk_msg.response_metadata:
                response_metadata.update(chunk_msg.response_metadata)
        if etype == "content_block_stop":
            for ev in tracker.finish_block(getattr(event, "index", None)):
                yield ev

    if not started:
        return
    for ev in tracker.finish_all():
        yield ev
    yield build_message_finish(usage=usage, response_metadata=response_metadata)
