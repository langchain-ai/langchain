"""Native content-block streaming-event converter for openrouter.

Builds text, reasoning, and tool-call blocks directly from openrouter's raw
OpenAI-shaped delta (openrouter streams tool-call args incrementally), feeding
the shared `BlockStreamTracker`. Unlike the compat bridge (which leaves
openrouter's `tool_calls`/`reasoning` in `additional_kwargs`), this surfaces
them as blocks.

Differs from the groq converter in three ways: usage is a top-level
`chunk["usage"]` (carrying openrouter `cost`/`cost_details`), a no-choices
chunk may carry an `error` payload that must be propagated, and the reasoning
field is `delta.reasoning` (not `reasoning_content`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.stream_events import (
    BlockStreamTracker,
    accumulate_usage,
    build_message_finish,
)

from langchain_openrouter.chat_models import _create_usage_metadata

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
    metadata: MessageMetadata = {"provider": "openrouter"}
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
    """Yield block events for one OpenAI-shaped openrouter delta."""
    if reasoning := delta.get("reasoning"):
        yield from tracker.feed(
            "reasoning", {"type": "reasoning", "reasoning": reasoning}
        )
    if content := delta.get("content"):
        yield from tracker.feed("text", {"type": "text", "text": content})
    # openrouter structured reasoning fragments (delta["reasoning_details"]) are
    # not surfaced as blocks yet; tracked for a follow-up.
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        fn = tc.get("function") or {}
        yield from tracker.feed(
            f"tool:{idx}",
            {
                "type": "tool_call_chunk",
                "id": tc.get("id"),
                "name": fn.get("name"),
                "args": fn.get("arguments") or "",
                "index": idx,
            },
        )


def _error_message(error: dict[str, Any]) -> str:
    """Build the streaming-error message, matching `_stream`'s `ValueError`."""
    return (
        f"OpenRouter API returned an error during streaming: "
        f"{error.get('message', str(error))} "
        f"(code: {error.get('code', 'unknown')})"
    )


class _StreamState:
    """Accumulates per-stream state shared by the sync and async converters."""

    def __init__(self) -> None:
        self.tracker = BlockStreamTracker()
        self.started = False
        self.usage: dict[str, Any] | None = None
        self.response_metadata: dict[str, Any] = {"model_provider": "openrouter"}
        self.model: str | None = None

    def prepare(self, chunk: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Coerce a raw chunk, fold in usage/cost, and return it plus choices.

        Raises:
            ValueError: If a no-choices chunk carries an openrouter `error`.
        """
        if not isinstance(chunk, dict):
            chunk = chunk.model_dump(by_alias=True)
        if self.model is None:
            self.model = chunk.get("model")
        if usage_payload := chunk.get("usage"):
            self.usage = accumulate_usage(
                self.usage, dict(_create_usage_metadata(usage_payload))
            )
            # Surface OpenRouter cost data, mirroring
            # `_convert_chunk_to_message_chunk` so the v3 finish metadata keeps
            # parity with the compat bridge's `response_metadata`.
            if "cost" in usage_payload:
                self.response_metadata["cost"] = usage_payload["cost"]
            if "cost_details" in usage_payload:
                self.response_metadata["cost_details"] = usage_payload["cost_details"]
        choices = chunk.get("choices") or []
        if len(choices) == 0 and (error := chunk.get("error")):
            raise ValueError(_error_message(error))
        return chunk, choices

    def record_finish(self, chunk: dict[str, Any], choice: dict[str, Any]) -> None:
        """Capture finish-chunk response metadata, matching `_stream`.

        Mirrors the `generation_info` that `_stream`/`_astream` attach on the
        final chunk so the v3 `message-finish` metadata equals the bridge's.
        """
        self.response_metadata["finish_reason"] = choice["finish_reason"]
        self.response_metadata["model_name"] = chunk.get("model") or self.model
        if system_fingerprint := chunk.get("system_fingerprint"):
            self.response_metadata["system_fingerprint"] = system_fingerprint
        if native_finish_reason := choice.get("native_finish_reason"):
            self.response_metadata["native_finish_reason"] = native_finish_reason
        if response_id := chunk.get("id"):
            self.response_metadata["id"] = response_id
        if created := chunk.get("created"):
            self.response_metadata["created"] = int(created)
        if object_ := chunk.get("object"):
            self.response_metadata["object"] = object_


def convert_openrouter_stream(
    raw: Iterator[Any], *, message_id: str | None = None
) -> Iterator[MessagesData]:
    """Convert a raw openrouter chat stream to protocol events.

    Args:
        raw: Raw openrouter chat-completion stream chunks (dicts or SDK
            objects).
        message_id: Overrides the provider message id on `message-start`.

    Yields:
        Protocol `MessagesData` lifecycle events.

    Raises:
        ValueError: If a chunk carries an openrouter `error` payload.
    """
    state = _StreamState()
    for chunk in raw:
        chunk_dict, choices = state.prepare(chunk)
        if len(choices) == 0:
            continue
        if not state.started:
            state.started = True
            yield _message_start(message_id, state.model)
        choice = choices[0]
        yield from _feed_delta(state.tracker, choice.get("delta") or {})
        if choice.get("finish_reason"):
            state.record_finish(chunk_dict, choice)

    if not state.started:
        return
    yield from state.tracker.finish_all()
    yield build_message_finish(
        usage=state.usage, response_metadata=state.response_metadata
    )


async def aconvert_openrouter_stream(
    raw: AsyncIterator[Any], *, message_id: str | None = None
) -> AsyncIterator[MessagesData]:
    """Async twin of `convert_openrouter_stream`."""
    state = _StreamState()
    async for chunk in raw:
        chunk_dict, choices = state.prepare(chunk)
        if len(choices) == 0:
            continue
        if not state.started:
            state.started = True
            yield _message_start(message_id, state.model)
        choice = choices[0]
        for ev in _feed_delta(state.tracker, choice.get("delta") or {}):
            yield ev
        if choice.get("finish_reason"):
            state.record_finish(chunk_dict, choice)

    if not state.started:
        return
    for ev in state.tracker.finish_all():
        yield ev
    yield build_message_finish(
        usage=state.usage, response_metadata=state.response_metadata
    )
