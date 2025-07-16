"""Utilities for working with iterators."""

from collections import deque
from collections.abc import Generator, Iterable, Iterator
from contextlib import AbstractContextManager
from itertools import islice
from types import TracebackType
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Literal

T = TypeVar("T")


class NoLock:
    """Dummy lock that provides the proper interface but no protection."""

    def __enter__(self) -> None:
        """Do nothing."""

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Exception not handled."""
        return False


def tee_peer(
    iterator: Iterator[T],
    # the buffer specific to this peer
    buffer: deque[T],
    # the buffers of all peers, including our own
    peers: list[deque[T]],
    lock: AbstractContextManager[Any],
) -> Generator[T, None, None]:
    """An individual iterator of a :py:func:`~.tee`.

    This function is a generator that yields items from the shared iterator
    ``iterator``. It buffers items until the least advanced iterator has
    yielded them as well. The buffer is shared with all other peers.

    Args:
        iterator: The shared iterator.
        buffer: The buffer for this peer.
        peers: The buffers of all peers.
        lock: The lock to synchronise access to the shared buffers.

    Yields:
        The next item from the shared iterator.
    """
    try:
        while True:
            if not buffer:
                with lock:
                    # Another peer produced an item while we were waiting for the lock.
                    # Proceed with the next loop iteration to yield the item.
                    if buffer:
                        continue
                    try:
                        item = next(iterator)
                    except StopIteration:
                        break
                    else:
                        # Append to all buffers, including our own. We'll fetch our
                        # item from the buffer again, instead of yielding it directly.
                        # This ensures the proper item ordering if any of our peers
                        # are fetching items concurrently. They may have buffered their
                        # item already.
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        with lock:
            # this peer is done - remove its buffer
            for idx, peer_buffer in enumerate(peers):  # pragma: no branch
                if peer_buffer is buffer:
                    peers.pop(idx)
                    break
            # if we are the last peer, try and close the iterator
            if not peers and hasattr(iterator, "close"):
                iterator.close()


class Tee(Generic[T]):
    """Create ``n`` separate asynchronous iterators over ``iterable``.

    This splits a single ``iterable`` into multiple iterators, each providing
    the same items in the same order.
    All child iterators may advance separately but share the same items
    from ``iterable`` -- when the most advanced iterator retrieves an item,
    it is buffered until the least advanced iterator has yielded it as well.
    A ``tee`` works lazily and can handle an infinite ``iterable``, provided
    that all iterators advance.

    .. code-block:: python3

        async def derivative(sensor_data):
            previous, current = a.tee(sensor_data, n=2)
            await a.anext(previous)  # advance one iterator
            return a.map(operator.sub, previous, current)

    Unlike :py:func:`itertools.tee`, :py:func:`~.tee` returns a custom type instead
    of a :py:class:`tuple`. Like a tuple, it can be indexed, iterated and unpacked
    to get the child iterators. In addition, its :py:meth:`~.tee.aclose` method
    immediately closes all children, and it can be used in an ``async with`` context
    for the same effect.

    If ``iterable`` is an iterator and read elsewhere, ``tee`` will *not*
    provide these items. Also, ``tee`` must internally buffer each item until the
    last iterator has yielded it; if the most and least advanced iterator differ
    by most data, using a :py:class:`list` is more efficient (but not lazy).

    If the underlying iterable is concurrency safe (``anext`` may be awaited
    concurrently) the resulting iterators are concurrency safe as well. Otherwise,
    the iterators are safe if there is only ever one single "most advanced" iterator.
    To enforce sequential use of ``anext``, provide a ``lock``
    - e.g. an :py:class:`asyncio.Lock` instance in an :py:mod:`asyncio` application -
    and access is automatically synchronised.
    """

    def __init__(
        self,
        iterable: Iterator[T],
        n: int = 2,
        *,
        lock: Optional[AbstractContextManager[Any]] = None,
    ):
        """Create a ``tee``.

        Args:
            iterable: The iterable to split.
            n: The number of iterators to create. Defaults to 2.
            lock: The lock to synchronise access to the shared buffers.
                Defaults to None.
        """
        self._iterator = iter(iterable)
        self._buffers: list[deque[T]] = [deque() for _ in range(n)]
        self._children = tuple(
            tee_peer(
                iterator=self._iterator,
                buffer=buffer,
                peers=self._buffers,
                lock=lock if lock is not None else NoLock(),
            )
            for buffer in self._buffers
        )

    def __len__(self) -> int:
        """Return the number of child iterators."""
        return len(self._children)

    @overload
    def __getitem__(self, item: int) -> Iterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[Iterator[T], ...]: ...

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[Iterator[T], tuple[Iterator[T], ...]]:
        """Return the child iterator(s) at the given index or slice."""
        return self._children[item]

    def __iter__(self) -> Iterator[Iterator[T]]:
        """Return an iterator over the child iterators."""
        yield from self._children

    def __enter__(self) -> "Tee[T]":
        """Return Tee instance."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        """Close all child iterators."""
        self.close()
        return False

    def close(self) -> None:
        """Close all child iterators."""
        for child in self._children:
            child.close()


# Why this is needed https://stackoverflow.com/a/44638570
safetee = Tee


def batch_iterate(size: Optional[int], iterable: Iterable[T]) -> Iterator[list[T]]:
    """Utility batching function.

    Args:
        size: The size of the batch. If None, returns a single batch.
        iterable: The iterable to batch.

    Yields:
        The batches of the iterable.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk
