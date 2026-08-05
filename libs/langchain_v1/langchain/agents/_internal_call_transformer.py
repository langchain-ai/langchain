"""Tag and filter middleware-internal model calls.

Middleware may make bookkeeping model calls (e.g. summarization or tool
selection) in the same graph namespace as the main agent call, causing their
tokens to appear in `run.messages`.

Tag these calls with `internal_call_metadata()` and declare
`transformers = (InternalCallTransformer,)` on the middleware class so it's
only registered on agents that actually use it — see `AgentMiddleware.transformers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.messages import BaseMessage
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

    Declared on `transformers` by middleware that makes internal calls (e.g.
    `SummarizationMiddleware`), so it's only registered on agents using one of
    those, and runs before built-in transformers.

    Handles both streamed protocol events and whole-`AIMessage` events. For
    internal calls, it rewrites streamed `message-start` events to a `"tool"`
    role and clears whole-`AIMessage` payloads so `MessagesTransformer` ignores
    them, while dropping both from the raw event log.
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

        params = event["params"]
        payload, metadata = params["data"]
        is_internal = bool(metadata and metadata.get(INTERNAL_CALL_METADATA_KEY))

        if isinstance(payload, dict) and "event" in payload:
            return self._process_protocol_event(payload, metadata, is_internal=is_internal)

        if isinstance(payload, BaseMessage):
            if is_internal:
                # `MessagesTransformer`'s whole-message route doesn't consult
                # metadata, so clear the payload rather than rely on our
                # return value alone.
                params["data"] = (None, metadata)
                return False
            return True

        return True

    def _process_protocol_event(
        self, payload: dict[str, Any], metadata: dict[str, Any] | None, *, is_internal: bool
    ) -> bool:
        run_id = str(metadata.get("run_id", "")) if metadata else ""
        event_type = payload.get("event")

        if event_type == "message-start":
            if is_internal:
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
