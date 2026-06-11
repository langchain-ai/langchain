"""Native content-block streaming-event converter for Ollama.

Maps the raw Ollama chat stream (dicts with `message.content`,
`message.thinking`, `message.tool_calls`, and final `done`/usage fields) to the
protocol `MessagesData` lifecycle, feeding the shared `BlockStreamTracker`.
Unlike the compat bridge (which buries reasoning in `additional_kwargs`), this
surfaces thinking as real `reasoning` content blocks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models.stream_events import (
    BlockStreamTracker,
    accumulate_usage,
    build_message_finish,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.messages import ToolCall
    from langchain_protocol.protocol import (
        MessageMetadata,
        MessagesData,
        MessageStartData,
    )

# Bound `_get_tool_calls_from_response`.
GetToolCalls = Callable[[Any], "list[ToolCall]"]

# Stable per-block keys for the tracker (Ollama gives no indices).
_TEXT_KEY = "text"
_REASONING_KEY = "reasoning"


def _message_start(message_id: str | None, model: str | None) -> MessageStartData:
    metadata: MessageMetadata = {"provider": "ollama"}
    if model:
        metadata["model"] = model
    # `cast` rather than a bare TypedDict literal: the strict `ty` checker
    # rejects the literal against the external `MessageStartData` TypedDict.
    return cast(
        "MessageStartData",
        {
            "event": "message-start",
            "role": "ai",
            "id": message_id or "",
            "metadata": metadata,
        },
    )


def convert_ollama_stream(
    raw: Iterator[Any],
    get_tool_calls: GetToolCalls,
    *,
    reasoning: bool | None = None,
    message_id: str | None = None,
) -> Iterator[MessagesData]:
    """Convert a raw Ollama chat stream to protocol events.

    Args:
        raw: Raw Ollama stream items (dicts; non-dicts skipped).
        get_tool_calls: `ChatOllama`'s `_get_tool_calls_from_response`, injected
            so the converter stays pure.
        reasoning: When truthy, surface `message.thinking` as reasoning blocks.
        message_id: Left empty by default so the v3 stream's seeded run id stands.

    Yields:
        Protocol `MessagesData` lifecycle events.
    """
    # Local import to avoid a circular import: `chat_models` imports this module.
    from langchain_ollama.chat_models import (  # noqa: PLC0415
        _get_usage_metadata_from_generation_info,
    )

    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "ollama"}
    tool_idx = 0

    for resp in raw:
        if not isinstance(resp, dict):
            continue

        message = resp.get("message") or {}

        # Skip "load" responses with empty content, matching the compat bridge
        # (`_iterate_over_stream`): the model was loaded but generated nothing,
        # so emitting an empty message-start/finish would diverge from the bridge.
        content = message.get("content") or ""
        if (
            resp.get("done") is True
            and resp.get("done_reason") == "load"
            and not content.strip()
        ):
            continue

        if not started:
            started = True
            yield _message_start(message_id, resp.get("model"))

        thinking = message.get("thinking")
        if reasoning and thinking:
            yield from tracker.feed(
                _REASONING_KEY, {"type": "reasoning", "reasoning": thinking}
            )

        if content:
            yield from tracker.feed(_TEXT_KEY, {"type": "text", "text": content})

        if message.get("tool_calls"):
            for tc in get_tool_calls(resp):
                yield from tracker.feed(
                    f"tool:{tool_idx}",
                    {
                        "type": "tool_call_chunk",
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": json.dumps(tc.get("args") or {}),
                    },
                )
                tool_idx += 1

        if resp.get("done") is True:
            usage = accumulate_usage(
                usage, _get_usage_metadata_from_generation_info(resp)
            )
            done_meta = {
                k: v for k, v in resp.items() if k != "message" and v is not None
            }
            if "model" in done_meta:
                done_meta["model_name"] = done_meta["model"]
            response_metadata.update(done_meta)

    if not started:
        return
    yield from tracker.finish_all()
    yield build_message_finish(usage=usage, response_metadata=response_metadata)


async def aconvert_ollama_stream(
    raw: AsyncIterator[Any],
    get_tool_calls: GetToolCalls,
    *,
    reasoning: bool | None = None,
    message_id: str | None = None,
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_ollama_stream`.

    `get_tool_calls` and the usage helper stay sync.
    """
    # Local import to avoid a circular import: `chat_models` imports this module.
    from langchain_ollama.chat_models import (  # noqa: PLC0415
        _get_usage_metadata_from_generation_info,
    )

    tracker = BlockStreamTracker()
    started = False
    usage: dict[str, Any] | None = None
    response_metadata: dict[str, Any] = {"model_provider": "ollama"}
    tool_idx = 0

    async for resp in raw:
        if not isinstance(resp, dict):
            continue

        message = resp.get("message") or {}

        # Skip "load" responses with empty content, matching the compat bridge
        # (`_aiterate_over_stream`): the model was loaded but generated nothing,
        # so emitting an empty message-start/finish would diverge from the bridge.
        content = message.get("content") or ""
        if (
            resp.get("done") is True
            and resp.get("done_reason") == "load"
            and not content.strip()
        ):
            continue

        if not started:
            started = True
            yield _message_start(message_id, resp.get("model"))

        thinking = message.get("thinking")
        if reasoning and thinking:
            for ev in tracker.feed(
                _REASONING_KEY, {"type": "reasoning", "reasoning": thinking}
            ):
                yield ev

        if content:
            for ev in tracker.feed(_TEXT_KEY, {"type": "text", "text": content}):
                yield ev

        if message.get("tool_calls"):
            for tc in get_tool_calls(resp):
                for ev in tracker.feed(
                    f"tool:{tool_idx}",
                    {
                        "type": "tool_call_chunk",
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": json.dumps(tc.get("args") or {}),
                    },
                ):
                    yield ev
                tool_idx += 1

        if resp.get("done") is True:
            usage = accumulate_usage(
                usage, _get_usage_metadata_from_generation_info(resp)
            )
            done_meta = {
                k: v for k, v in resp.items() if k != "message" and v is not None
            }
            if "model" in done_meta:
                done_meta["model_name"] = done_meta["model"]
            response_metadata.update(done_meta)

    if not started:
        return
    for ev in tracker.finish_all():
        yield ev
    yield build_message_finish(usage=usage, response_metadata=response_metadata)
