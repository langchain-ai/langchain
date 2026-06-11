"""Native content-block streaming-event converter for MistralAI.

Mirrors `ChatMistralAI._stream`: threads `(index, index_type, default_class)`
through `_convert_chunk_to_message_chunk` (injected to avoid a circular import)
and feeds each resulting `AIMessageChunk`'s content blocks into the shared
`BlockStreamTracker`.
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
from langchain_core.messages import AIMessageChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.messages import BaseMessageChunk
    from langchain_protocol.protocol import (
        MessageMetadata,
        MessagesData,
        MessageStartData,
    )

# The module-level `_convert_chunk_to_message_chunk`, injected so the converter
# stays pure and avoids a circular import. It takes a raw chunk plus the running
# `default_class`, `index`, and `index_type`, returning the built chunk and the
# updated index/index_type.
ConvertChunk = Callable[..., "tuple[BaseMessageChunk, int, str]"]


def _message_start(message_id: str | None, model: str | None) -> MessageStartData:
    # Do not use the provider chunk id here: on the v3 path core seeds the
    # stream with the LangChain run id, and an empty id lets that stand
    # (matching the compat bridge). Only an explicit `message_id` wins.
    metadata: MessageMetadata = {"provider": "mistralai"}
    if model:
        metadata["model"] = model
    return {
        "event": "message-start",
        "role": "ai",
        "id": message_id or "",
        "metadata": metadata,
    }


def convert_mistral_stream(
    raw: Iterator[Any],
    convert_chunk: ConvertChunk,
    *,
    output_version: str | None = None,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Convert a raw Mistral chat stream to protocol events.

    Args:
        raw: Raw Mistral chat-completion chunks (OpenAI-shaped dicts).
        convert_chunk: `_convert_chunk_to_message_chunk`, injected so the
            converter stays pure and avoids a circular import.
        output_version: Forwarded to `convert_chunk`; reasoning blocks only
            surface under `"v1"`.
        message_id: Overrides the id on `message-start`.

    Yields:
        Protocol `MessagesData` lifecycle events.
    """
    tracker = BlockStreamTracker()
    started = False
    index = -1
    index_type = ""
    default_class: type[BaseMessageChunk] = AIMessageChunk
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "mistralai"}
    model: str | None = None

    for chunk in raw:
        if len(chunk.get("choices", [])) == 0:
            continue
        if model is None:
            model = chunk.get("model")
        new_chunk, index, index_type = convert_chunk(
            chunk, default_class, index, index_type, output_version
        )
        # Make future chunks the same type as the first chunk.
        default_class = new_chunk.__class__
        if not started:
            started = True
            yield _message_start(message_id, model)
        if isinstance(new_chunk, AIMessageChunk):
            for key, block in iter_protocol_blocks(new_chunk):
                yield from tracker.feed(key, block)
            if new_chunk.usage_metadata:
                usage = accumulate_usage(usage, new_chunk.usage_metadata)
            if new_chunk.response_metadata:
                response_metadata.update(new_chunk.response_metadata)

    if not started:
        return
    yield from tracker.finish_all()
    yield build_message_finish(usage=usage, response_metadata=response_metadata)


async def aconvert_mistral_stream(
    raw: AsyncIterator[Any],
    convert_chunk: ConvertChunk,
    *,
    output_version: str | None = None,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_mistral_stream`. `convert_chunk` is sync."""
    tracker = BlockStreamTracker()
    started = False
    index = -1
    index_type = ""
    default_class: type[BaseMessageChunk] = AIMessageChunk
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "mistralai"}
    model: str | None = None

    async for chunk in raw:
        if len(chunk.get("choices", [])) == 0:
            continue
        if model is None:
            model = chunk.get("model")
        new_chunk, index, index_type = convert_chunk(
            chunk, default_class, index, index_type, output_version
        )
        # Make future chunks the same type as the first chunk.
        default_class = new_chunk.__class__
        if not started:
            started = True
            yield _message_start(message_id, model)
        if isinstance(new_chunk, AIMessageChunk):
            for key, block in iter_protocol_blocks(new_chunk):
                for ev in tracker.feed(key, block):
                    yield ev
            if new_chunk.usage_metadata:
                usage = accumulate_usage(usage, new_chunk.usage_metadata)
            if new_chunk.response_metadata:
                response_metadata.update(new_chunk.response_metadata)

    if not started:
        return
    for ev in tracker.finish_all():
        yield ev
    yield build_message_finish(usage=usage, response_metadata=response_metadata)
