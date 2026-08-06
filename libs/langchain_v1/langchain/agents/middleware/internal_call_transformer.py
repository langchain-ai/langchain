"""Tag and filter middleware-internal model calls.

Middleware may make bookkeeping model calls (e.g. summarization or tool
selection) in the same graph namespace as the main agent call, causing their
tokens to appear in `run.messages`.

Tag these calls with `internal_call_metadata()` and declare
`transformers = (InternalCallTransformer,)` on the middleware class so it's
only registered on agents that actually use it — see `AgentMiddleware.transformers`.
Both are public so third-party middleware can adopt the same pattern.
"""

from __future__ import annotations

import secrets
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

_INTERNAL_CALL_TOKEN = secrets.token_hex(16)
"""Unguessable marker value, regenerated on import.

`config["metadata"]` ultimately comes from a `RunnableConfig`, which callers
of `invoke`/`stream_events` can populate with arbitrary values — including
the main agent turn's own call, since it goes through the same ambient
config. If the marker were a fixed value like `True`, a caller who can
influence invocation metadata (e.g. an API layer that forwards user-supplied
metadata) could set `lc_internal_call` themselves and hide the agent's real
answer from `run.messages`. Comparing against this process-local secret
instead of truthiness means a caller can't forge it without already being
able to run code in this process.
"""


def internal_call_metadata() -> dict[str, Any]:
    """Return metadata that marks a model call as internal to middleware.

    Returns:
        A mapping to merge into a model call's `config["metadata"]`.
    """
    return {INTERNAL_CALL_METADATA_KEY: _INTERNAL_CALL_TOKEN}


class InternalCallTransformer(StreamTransformer):
    """Keep internal model calls out of `run.messages` and the raw event log.

    Declared on `transformers` by middleware that makes internal calls (e.g.
    `SummarizationMiddleware`), so it's only registered on agents using one of
    those, and runs before built-in transformers.

    `messages`-mode events come in two shapes, and influencing
    `MessagesTransformer`'s built-in exclusion rules for either one requires
    mutating the event in place (there's no metadata hook it consults). That
    mutation would misrepresent the call if it also reached raw event
    consumers — a real AI response reported as `role: "tool"`, or a payload
    replaced with `None`, neither a real messages-mode shape — so tagged
    events are dropped from the raw log entirely rather than published in a
    mutated form:

    - Streamed protocol events (`message-start` / `content-block-*` /
      `message-finish`): `message-start`'s `role` is rewritten to `"tool"`,
      reusing `MessagesTransformer`'s existing tool-result exclusion, then
      the event is dropped.
    - Whole-`AIMessage` events — the fallback `MessagesTransformer` uses when
      a chat model doesn't stream (notably, streaming context isn't
      propagated on Python 3.10) or when a node returns a finalized message
      as state: the payload is cleared so `MessagesTransformer` has nothing
      left to route, then the event is dropped.

    Only events in this transformer's own scope are touched — nested
    subgraphs get their own scoped instance (if the offending middleware runs
    there too), and `MessagesTransformer` itself ignores events outside its
    scope, so mutating them here would be both unnecessary and unsafe.
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        """Initialize the transformer, scoped to a single mux/namespace.

        Args:
            scope: The namespace tuple the owning mux is scoped to.
        """
        super().__init__(scope)
        self._scope_list: list[str] = list(scope)

    def init(self) -> dict[str, Any]:
        """Return an empty projection — this transformer only suppresses events.

        Returns:
            An empty mapping; no `run.<key>` projection is added.
        """
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        """Drop tagged `messages` events after nudging `MessagesTransformer` to ignore them.

        Args:
            event: The protocol event to observe, and possibly mutate.

        Returns:
            `False` to drop a tagged internal call from the raw event log and
            `run.messages`; `True` otherwise.
        """
        if event["method"] != "messages":
            return True

        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True

        payload, metadata = params["data"]
        is_internal = (
            bool(metadata) and metadata.get(INTERNAL_CALL_METADATA_KEY) == _INTERNAL_CALL_TOKEN
        )
        if not is_internal:
            return True

        # Only `message-start` needs mutating: once MessagesTransformer sees its
        # `role` spoofed as "tool", its own tool-result bookkeeping ignores every
        # later event for this run_id, so content-block-*/message-finish need no
        # action here, the whole run gets dropped below regardless.
        if isinstance(payload, dict) and payload.get("event") == "message-start":
            payload["role"] = "tool"
        elif isinstance(payload, BaseMessage):
            params["data"] = (None, metadata)

        return False
