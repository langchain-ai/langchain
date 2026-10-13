"""Tag and filter middleware-internal model calls.

Tag internal calls with `internal_call_metadata()` and declare
`transformers = (InternalCallTransformer,)` on the middleware class to keep
them out of `run.messages`. Both APIs are public for third-party middleware.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.messages import BaseMessage
from langgraph.stream import StreamTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

INTERNAL_CALL_METADATA_KEY = "lc_internal_call"
"""Metadata key marking a model call as internal to middleware.

Kept separate from `lc_source` so filtering doesn't affect its existing
semantics.
"""

_INTERNAL_CALL_TOKEN = secrets.token_hex(16)
"""Process-local marker used to prevent callers from spoofing internal calls.

A random token prevents user-supplied metadata from hiding real model calls
from `run.messages`.
"""


def internal_call_metadata() -> dict[str, Any]:
    """Return metadata that marks a model call as internal to middleware.

    Returns:
        A mapping to merge into a model call's `config["metadata"]`.
    """
    return {INTERNAL_CALL_METADATA_KEY: _INTERNAL_CALL_TOKEN}


class InternalCallTransformer(StreamTransformer):
    """Keep internal model calls out of `run.messages` and the raw event log.

    Used by middleware that makes internal model calls and runs before built-in
    transformers.

    For tagged events, streamed `message-start` events are marked as tool-role and
    whole-`AIMessage` payloads are cleared so `MessagesTransformer` ignores them.
    The mutated events are then dropped from the raw log.

    Only events within this transformer's scope are modified.
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

        # Only `message-start` needs mutation: marking it as `"tool"` makes
        # `MessagesTransformer` ignore the rest of that run. All events are still
        # dropped from the raw log below.
        if isinstance(payload, dict) and payload.get("event") == "message-start":
            payload["role"] = "tool"
        elif isinstance(payload, BaseMessage):
            params["data"] = (None, metadata)

        return False
