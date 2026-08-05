"""Tag and filter middleware-internal model calls.

Middleware may make bookkeeping model calls (e.g. summarization or tool
selection) in the same graph namespace as the main agent call, causing their
tokens to appear in `run.messages`.

Tag these calls with `internal_call_metadata()` so `InternalCallTransformer`
can filter them from `run.messages` and the raw event log.
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
    """Keep internal model calls out of `run.messages` and the raw event log.

    Registered on every compiled agent and runs before built-in transformers.
    For internal calls, it drops the raw event and marks `message-start` as a
    tool role so `MessagesTransformer` excludes it from `run.messages`.
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
