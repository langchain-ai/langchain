"""Internal tracers used for `stream_log` and `astream` events implementations."""

import typing
from collections.abc import AsyncIterator, Iterator
from uuid import UUID

T = typing.TypeVar("T")


# THIS IS USED IN LANGGRAPH.
@typing.runtime_checkable
class _StreamingCallbackHandler(typing.Protocol[T]):
    """Types for streaming callback handlers.

    This is a common mixin that the callback handlers for both astream events and
    astream log inherit from.

    The `tap_output_aiter` method is invoked in some contexts to produce callbacks for
    intermediate results.
    """

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Used for internal astream_log and astream events implementations."""

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Used for internal astream_log and astream events implementations."""


# THIS IS USED IN LANGGRAPH.
class _V2StreamingCallbackHandler:
    """Marker base class for handlers that consume `on_stream_event` (v2).

    A handler inheriting from this class signals that it wants content-
    block lifecycle events from `stream_events(version="v3")` (and its
    async equivalent) rather than the v1 `on_llm_new_token` chunks.
    `BaseChatModel.invoke` uses
    `isinstance(handler, _V2StreamingCallbackHandler)` to decide whether
    to route an invoke through the v2 event generator.

    Implemented as a concrete marker class (not a `Protocol`) so opt-in
    is explicit via inheritance. An empty `runtime_checkable` Protocol
    would match every object and misroute every call. The event
    delivery contract itself lives on
    `BaseCallbackHandler.on_stream_event`.
    """


__all__ = [
    "_StreamingCallbackHandler",
    "_V2StreamingCallbackHandler",
]
