"""Native content-block streaming-event converter for OpenAI Chat Completions.

Drives raw OpenAI Chat Completions chunks into the shared `BlockStreamTracker`,
reusing `BaseChatOpenAI._convert_chunk_to_generation_chunk` for content
extraction (it already yields indexed content blocks + tool-call chunks). This
converter is the reuse seam for OpenAI-compatible providers (deepseek, groq,
fireworks, xai, openrouter), which adapt their chunk shape to OpenAI's and call
it with a different `provider`.
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

    from langchain_core.outputs import ChatGenerationChunk
    from langchain_protocol.protocol import (
        MessageMetadata,
        MessagesData,
        MessageStartData,
    )

# Bound `BaseChatOpenAI._convert_chunk_to_generation_chunk`.
MakeChunk = Callable[..., "ChatGenerationChunk | None"]


def _message_start(
    message_id: str | None, model: str | None, provider: str
) -> MessageStartData:
    metadata: MessageMetadata = {"provider": provider}
    if model:
        metadata["model"] = model
    return {
        "event": "message-start",
        "role": "ai",
        "id": message_id or "",
        "metadata": metadata,
    }


def convert_openai_completions_stream(
    raw: Iterator[Any],
    make_chunk: MakeChunk,
    *,
    base_generation_info: dict[str, Any] | None = None,
    message_id: str | None = None,
    provider: str = "openai",
) -> Iterator[MessagesData]:
    """Convert a raw OpenAI Chat Completions chunk stream to protocol events.

    Args:
        raw: Raw OpenAI chunks (dicts or SDK objects with `model_dump`).
        make_chunk: `BaseChatOpenAI._convert_chunk_to_generation_chunk`, injected
            so the converter stays pure and unit-testable.
        base_generation_info: Passed to `make_chunk` for the first chunk only
            (mirrors `_stream`), `{}` thereafter.
        message_id: Message id for `message-start`. Left empty by default so
            the v3 stream's seeded LangChain run id stands (matching the compat
            bridge); the provider completion id is deliberately not used here.
        provider: `model_provider` id for downstream reuse (groq, deepseek, ...).

    Yields:
        Protocol `MessagesData` lifecycle events.
    """
    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": provider}
    model: str | None = None
    first = True

    for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None and chunk.get("model"):
            model = chunk["model"]
        gen = make_chunk(chunk, AIMessageChunk, base_generation_info if first else {})
        first = False
        if gen is None:
            continue
        msg = gen.message
        if not started:
            started = True
            yield _message_start(message_id, model, provider)
        for key, block in iter_protocol_blocks(msg):
            yield from tracker.feed(key, block)
        usage_metadata = getattr(msg, "usage_metadata", None)
        if usage_metadata:
            usage = accumulate_usage(usage, usage_metadata)
        merged = {**(gen.generation_info or {}), **(msg.response_metadata or {})}
        if merged:
            response_metadata.update(merged)
            # `_convert_chunk_to_generation_chunk` hardcodes
            # `model_provider="openai"`; re-apply the caller's `provider` so
            # OpenAI-compatible reuse (groq, deepseek, ...) isn't mislabeled.
            response_metadata["model_provider"] = provider

    if not started:
        return
    yield from tracker.finish_all()
    yield build_message_finish(usage=usage, response_metadata=response_metadata)


async def aconvert_openai_completions_stream(
    raw: AsyncIterator[Any],
    make_chunk: MakeChunk,
    *,
    base_generation_info: dict[str, Any] | None = None,
    message_id: str | None = None,
    provider: str = "openai",
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_openai_completions_stream`. `make_chunk` is sync."""
    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": provider}
    model: str | None = None
    first = True

    async for chunk in raw:
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump()
        if model is None and chunk.get("model"):
            model = chunk["model"]
        gen = make_chunk(chunk, AIMessageChunk, base_generation_info if first else {})
        first = False
        if gen is None:
            continue
        msg = gen.message
        if not started:
            started = True
            yield _message_start(message_id, model, provider)
        for key, block in iter_protocol_blocks(msg):
            for ev in tracker.feed(key, block):
                yield ev
        usage_metadata = getattr(msg, "usage_metadata", None)
        if usage_metadata:
            usage = accumulate_usage(usage, usage_metadata)
        merged = {**(gen.generation_info or {}), **(msg.response_metadata or {})}
        if merged:
            response_metadata.update(merged)
            # `_convert_chunk_to_generation_chunk` hardcodes
            # `model_provider="openai"`; re-apply the caller's `provider` so
            # OpenAI-compatible reuse (groq, deepseek, ...) isn't mislabeled.
            response_metadata["model_provider"] = provider

    if not started:
        return
    for ev in tracker.finish_all():
        yield ev
    yield build_message_finish(usage=usage, response_metadata=response_metadata)
