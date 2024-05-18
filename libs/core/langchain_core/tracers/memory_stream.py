"""Module implements a memory stream for communication between two co-routines.

This module provides a way to communicate between two co-routines using a memory
channel. The writer and reader can be in the same event loop or in different event
loops. When they're in different event loops, they will also be in different
threads.

This is useful in situations when there's a mix of synchronous and asynchronous
used in the code.
"""
import asyncio
from asyncio import AbstractEventLoop, Queue
from typing import AsyncIterator, Generic, TypeVar

T = TypeVar("T")


class _SendStream(Generic[T]):
    def __init__(
        self, reader_loop: AbstractEventLoop, queue: Queue, done: object
    ) -> None:
        """Create a writer for the queue and done object.

        Args:
            reader_loop: The event loop to use for the writer. This loop will be used
                         to schedule the writes to the queue.
            queue: The queue to write to. This is an asyncio queue.
            done: Special sentinel object to indicate that the writer is done.
        """
        self._reader_loop = reader_loop
        self._queue = queue
        self._done = done

    async def send(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop."""
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop."""
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed

    async def aclose(self) -> None:
        """Schedule the done object write the queue using the original loop."""
        return self.close()

    def close(self) -> None:
        """Schedule the done object write the queue using the original loop."""
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, self._done)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed


class _ReceiveStream(Generic[T]):
    def __init__(self, queue: Queue, done: object) -> None:
        """Create a reader for the queue and done object.

        This reader should be used in the same loop as the loop that was passed
        to the channel.
        """
        self._queue = queue
        self._done = done
        self._is_closed = False

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is self._done:
                self._is_closed = True
                break
            yield item


class _MemoryStream(Generic[T]):
    """Stream data from a writer to a reader even if they are in different threads.

    Uses asyncio queues to communicate between two co-routines. This implementation
    should work even if the writer and reader co-routines belong to two different
    event loops (e.g. one running from an event loop in the main thread
    and the other running in an event loop in a background thread).

    This implementation is meant to be used with a single writer and a single reader.

    This is an internal implementation to LangChain please do not use it directly.
    """

    def __init__(self, loop: AbstractEventLoop) -> None:
        """Create a channel for the given loop.

        Args:
            loop: The event loop to use for the channel. The reader is assumed
                  to be running in the same loop as the one passed to this constructor.
                  This will NOT be validated at run time.
        """
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=0)
        self._done = object()

    def get_send_stream(self) -> _SendStream[T]:
        """Get a writer for the channel."""
        return _SendStream[T](
            reader_loop=self._loop, queue=self._queue, done=self._done
        )

    def get_receive_stream(self) -> _ReceiveStream[T]:
        """Get a reader for the channel."""
        return _ReceiveStream[T](queue=self._queue, done=self._done)
