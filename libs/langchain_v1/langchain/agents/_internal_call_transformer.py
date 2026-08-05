"""Tag and filter model calls middleware makes for its own bookkeeping.

Middleware occasionally calls a model outside the main agent turn — for
example `SummarizationMiddleware` condensing history, `LLMToolEmulator` faking
a tool result, or `LLMToolSelectionMiddleware` picking which tools to expose.
These calls run inside the same graph node/namespace as the main agent loop,
so `MessagesTransformer` (the built-in `messages` projection, which filters
only by checkpoint namespace) can't tell them apart from the model's real
turn — their tokens land in `run.messages` right alongside the actual answer.

Middleware that makes such a call should merge `internal_call_metadata()`
into the call's `config["metadata"]`. `InternalCallTransformer` then keeps the
resulting `messages` events out of `run.messages` and off the raw event log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream import StreamTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

INTERNAL_CALL_METADATA_KEY = "lc_internal_call"
"""`RunnableConfig` metadata key marking a model call as internal to middleware.

Kept separate from `lc_source` (used by `SummarizationMiddleware` to advertise
that a summarization call is in flight) so tagging a call for filtering here
never changes what other consumers observe via that key.
"""


def internal_call_metadata() -> dict[str, Any]:
    """Return metadata that marks a model call as internal to middleware.

    Returns:
        A mapping to merge into a model call's `config["metadata"]`.
    """
    return {INTERNAL_CALL_METADATA_KEY: True}


class InternalCallTransformer(StreamTransformer):
    """Keep `messages` events for internal model calls out of `run.messages`.

    Registered unconditionally on every compiled agent (alongside
    `ToolCallTransformer` / `SubagentTransformer`) rather than opted into by
    individual middleware, since any middleware can make an internal call.

    Runs before built-in transformers (`before_builtins = True`), which gives
    it two levers, both needed since a `process` return value alone only
    controls the raw event log — it doesn't stop sibling transformers like
    `MessagesTransformer` from independently building their own projection
    from the same event:

    - Returning `False` drops the event from the raw protocol log, so
      internal calls never leak there either.
    - Rewriting a marked `message-start` event's `role` to `"tool"` before
      `MessagesTransformer` sees it reuses its existing tool-result
      exclusion (`run.messages` is documented as "the chat-token
      projection"; tool-role runs are already skipped) so the internal call
      never becomes a `ChatModelStream` in `run.messages`.
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._internal_runs: set[str] = set()

    def init(self) -> dict[str, Any]:
        # No projection — this transformer only suppresses events.
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True

        payload, metadata = event["params"]["data"]
        if not (isinstance(payload, dict) and "event" in payload):
            # Only the streamed protocol-event shape is produced by internal
            # calls today; whole-`AIMessage` node output is left untouched.
            return True

        run_id = str(metadata.get("run_id", "")) if metadata else ""
        event_type = payload.get("event")

        if event_type == "message-start":
            if metadata and metadata.get(INTERNAL_CALL_METADATA_KEY):
                self._internal_runs.add(run_id)
                payload["role"] = "tool"
                return False
            return True

        if run_id in self._internal_runs:
            if event_type == "message-finish":
                self._internal_runs.discard(run_id)
            return False

        return True

    def finalize(self) -> None:
        self._internal_runs.clear()
