"""Internal tracers used for stream_log and astream events implementations."""
import abc
from typing import AsyncIterator, Iterator, TypeVar
from uuid import UUID

T = TypeVar("T")


class _StreamingCallbackHandler(abc.ABC):
    """For internal use.

    This is a common mixin that the callback handlers
    for both astream events and astream log inherit from.

    The `tap_output_aiter` method is invoked in some contexts
    to produce callbacks for intermediate results.
    """

    @abc.abstractmethod
    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Used for internal astream_log and astream events implementations."""

    @abc.abstractmethod
    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Used for internal astream_log and astream events implementations."""


__all__ = [
    "_StreamingCallbackHandler",
]
