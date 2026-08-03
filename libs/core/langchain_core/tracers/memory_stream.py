"""Module implements a memory stream for communication between two co-routines.

This module provides a way to communicate between two co-routines using a memory
channel. The writer and reader can be in the same event loop or in different event
loops. When they're in different event loops, they will also be in different threads.

Useful in situations when there's a mix of synchronous and asynchronous used in the
code.
"""

import asyncio
import contextlib
import weakref
from asyncio import AbstractEventLoop, Queue
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _close_loop_quietly(loop: AbstractEventLoop) -> None:
    """Close an event loop that was created internally, as a finalizer callback.

    Errors are suppressed rather than logged. `weakref.finalize` defaults to
    `atexit=True`, so this can run during interpreter shutdown, where `logging`
    may already be torn down. Python routes finalizer exceptions to
    `sys.unraisablehook` rather than propagating them, so suppressing here only
    avoids dumping a traceback to stderr at a nondeterministic moment, which is
    the same kind of noise this cleanup exists to remove. If `close()` does
    fail, the loop's own `__del__` still emits
    `ResourceWarning: unclosed event loop`, so the leak stays detectable.

    Args:
        loop: The event loop to close. A running loop is left alone.
    """
    if loop.is_running():
        # `close()` would raise; leave the `ResourceWarning` as the signal.
        return
    with contextlib.suppress(Exception):
        loop.close()


def _get_or_create_loop(owner: object) -> AbstractEventLoop:
    """Get the current event loop, creating one if this thread has none.

    Args:
        owner: Object whose lifetime bounds a loop created here.

            A loop created by this function is closed once `owner` is
            garbage collected.

    Returns:
        The event loop to back a `_MemoryStream` with.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # No loop is set on this thread. That is always the case in a sync
        # context on Python 3.14+, where `get_event_loop()` no longer creates
        # one implicitly, and on any version in a non-main thread.
        #
        # This loop is ours: unlike one returned by `get_event_loop()`, no
        # caller owns it, so close it once `owner` dies. Otherwise the loop's
        # `__del__` emits `ResourceWarning: unclosed event loop` at whatever
        # nondeterministic point GC happens to run.
        loop = asyncio.new_event_loop()
        weakref.finalize(owner, _close_loop_quietly, loop)
        return loop


class _SendStream(Generic[T]):
    def __init__(
        self, reader_loop: AbstractEventLoop, queue: Queue[Any], done: object
    ) -> None:
        """Create a writer for the queue and done object.

        Args:
            reader_loop: The event loop to use for the writer.

                This loop will be used to schedule the writes to the queue.
            queue: The queue to write to.

                This is an asyncio queue.
            done: Special sentinel object to indicate that the writer is done.
        """
        self._reader_loop = reader_loop
        self._queue = queue
        self._done = done

    async def send(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop.

        This is a coroutine that can be awaited.

        Args:
            item: The item to write to the queue.
        """
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop.

        This is a non-blocking call.

        Args:
            item: The item to write to the queue.

        Raises:
            RuntimeError: If the event loop is already closed when trying to write to
                the queue.
        """
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed

    async def aclose(self) -> None:
        """Async schedule the done object write the queue using the original loop."""
        return self.close()

    def close(self) -> None:
        """Schedule the done object write the queue using the original loop.

        This is a non-blocking call.

        Raises:
            RuntimeError: If the event loop is already closed when trying to write to
                the queue.
        """
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, self._done)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed


class _ReceiveStream(Generic[T]):
    def __init__(self, queue: Queue[Any], done: object) -> None:
        """Create a reader for the queue and done object.

        This reader should be used in the same loop as the loop that was passed to the
        channel.
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
    should work even if the writer and reader co-routines belong to two different event
    loops (e.g. one running from an event loop in the main thread and the other running
    in an event loop in a background thread).

    This implementation is meant to be used with a single writer and a single reader.

    This is an internal implementation to LangChain. Do not use it directly.
    """

    def __init__(self, loop: AbstractEventLoop) -> None:
        """Create a channel for the given loop.

        Args:
            loop: The event loop to use for the channel.

                The reader is assumed to be running in the same loop as the one passed
                to this constructor. This will NOT be validated at run time.
        """
        self._loop = loop
        self._queue = asyncio.Queue[Any](maxsize=0)
        self._done = object()

    def get_send_stream(self) -> _SendStream[T]:
        """Get a writer for the channel.

        Returns:
            The writer for the channel.
        """
        return _SendStream[T](
            reader_loop=self._loop, queue=self._queue, done=self._done
        )

    def get_receive_stream(self) -> _ReceiveStream[T]:
        """Get a reader for the channel.

        Returns:
            The reader for the channel.
        """
        return _ReceiveStream[T](queue=self._queue, done=self._done)
