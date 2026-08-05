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
    """Keep internal model calls out of `run.messages`.

    Declared on `transformers` by middleware that makes internal calls (e.g.
    `SummarizationMiddleware`), so it's only registered on agents using one of
    those, and runs before built-in transformers.

    Only `run.messages` is affected — internal calls stay visible on the raw
    event log (`stream_events`'s underlying event iterator) for audit and
    observability consumers; no new projection is added. `messages`-mode
    events come in two shapes:

    - Streamed protocol events (`message-start` / `content-block-*` /
      `message-finish`): for internal calls, `message-start`'s `role` is
      rewritten to `"tool"`, reusing `MessagesTransformer`'s existing
      tool-result exclusion so the run never becomes a `run.messages` entry.
    - Whole-`AIMessage` events — the fallback `MessagesTransformer` uses when
      a chat model doesn't stream (notably, streaming context isn't
      propagated on Python 3.10) or when a node returns a finalized message
      as state: for internal calls, the payload is cleared so
      `MessagesTransformer` has nothing left to route. Unlike the streamed
      shape, this does redact the event's content on the raw log too — there
      is no metadata-free way to keep the original message visible there
      while also keeping `MessagesTransformer` from routing the very same
      object into `run.messages`.
    """

    before_builtins: ClassVar[bool] = True
    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)

    def init(self) -> dict[str, Any]:
        """Return an empty projection — this transformer only mutates events in place.

        Returns:
            An empty mapping; no `run.<key>` projection is added.
        """
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        """Rewrite tagged `messages` events so `MessagesTransformer` skips them.

        Args:
            event: The protocol event to observe, and possibly mutate.

        Returns:
            `True` always — internal calls stay on the raw event log; only
            `run.messages` excludes them.
        """
        if event["method"] != "messages":
            return True

        params = event["params"]
        payload, metadata = params["data"]
        is_internal = (
            bool(metadata) and metadata.get(INTERNAL_CALL_METADATA_KEY) == _INTERNAL_CALL_TOKEN
        )
        if not is_internal:
            return True

        if isinstance(payload, dict) and payload.get("event") == "message-start":
            # `MessagesTransformer` already excludes tool-role `message-start`
            # runs (and every later event for that run) from `run.messages`;
            # piggyback on that instead of adding a new check upstream.
            payload["role"] = "tool"
        elif isinstance(payload, BaseMessage):
            params["data"] = (None, metadata)

        # Keep the (possibly mutated) event on the raw log/`run` iterator —
        # only `run.messages` should exclude internal calls.
        return True
