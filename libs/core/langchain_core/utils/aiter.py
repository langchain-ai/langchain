"""Asynchronous iterator utilities.

Adapted from
https://github.com/maxfischer2781/asyncstdlib/blob/master/asyncstdlib/itertools.py
MIT License.
"""

from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Iterator,
)
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import override

T = TypeVar("T")

_no_default = object()


# https://github.com/python/cpython/blob/main/Lib/test/test_asyncgen.py#L54
# before 3.10, the builtin anext() was not available
def py_anext(
    iterator: AsyncIterator[T], default: Union[T, Any] = _no_default
) -> Awaitable[Union[T, None, Any]]:
    """Pure-Python implementation of anext() for testing purposes.

    Closely matches the builtin anext() C implementation.
    Can be used to compare the built-in implementation of the inner
    coroutines machinery to C-implementation of __anext__() and send()
    or throw() on the returned generator.

    Args:
        iterator: The async iterator to advance.
        default: The value to return if the iterator is exhausted.
            If not provided, a StopAsyncIteration exception is raised.

    Returns:
        The next value from the iterator, or the default value
            if the iterator is exhausted.

    Raises:
        TypeError: If the iterator is not an async iterator.
    """
    try:
        __anext__ = cast(
            "Callable[[AsyncIterator[T]], Awaitable[T]]", type(iterator).__anext__
        )
    except AttributeError as e:
        msg = f"{iterator!r} is not an async iterator"
        raise TypeError(msg) from e

    if default is _no_default:
        return __anext__(iterator)

    async def anext_impl() -> Union[T, Any]:
        try:
            # The C code is way more low-level than this, as it implements
            # all methods of the iterator protocol. In this implementation
            # we're relying on higher-level coroutine concepts, but that's
            # exactly what we want -- crosstest pure-Python high-level
            # implementation and low-level C anext() iterators.
            return await __anext__(iterator)
        except StopAsyncIteration:
            return default

    return anext_impl()


class NoLock:
    """Dummy lock that provides the proper interface but no protection."""

    async def __aenter__(self) -> None:
        """Do nothing."""

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Exception not handled."""
        return False


async def tee_peer(
    iterator: AsyncIterator[T],
    # the buffer specific to this peer
    buffer: deque[T],
    # the buffers of all peers, including our own
    peers: list[deque[T]],
    lock: AbstractAsyncContextManager[Any],
) -> AsyncGenerator[T, None]:
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
                async with lock:
                    # Another peer produced an item while we were waiting for the lock.
                    # Proceed with the next loop iteration to yield the item.
                    if buffer:
                        continue
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
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
        async with lock:
            # this peer is done - remove its buffer
            for idx, peer_buffer in enumerate(peers):  # pragma: no branch
                if peer_buffer is buffer:
                    peers.pop(idx)
                    break
            # if we are the last peer, try and close the iterator
            if not peers and hasattr(iterator, "aclose"):
                await iterator.aclose()


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
        iterable: AsyncIterator[T],
        n: int = 2,
        *,
        lock: Optional[AbstractAsyncContextManager[Any]] = None,
    ):
        """Create a ``tee``.

        Args:
            iterable: The iterable to split.
            n: The number of iterators to create. Defaults to 2.
            lock: The lock to synchronise access to the shared buffers.
                Defaults to None.
        """
        self._iterator = iterable.__aiter__()  # before 3.10 aiter() doesn't exist
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
    def __getitem__(self, item: int) -> AsyncIterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[AsyncIterator[T], ...]: ...

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[AsyncIterator[T], tuple[AsyncIterator[T], ...]]:
        """Return the child iterator(s) for the given index or slice."""
        return self._children[item]

    def __iter__(self) -> Iterator[AsyncIterator[T]]:
        """Iterate over the child iterators."""
        yield from self._children

    async def __aenter__(self) -> "Tee[T]":
        """Return the tee instance."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Close all child iterators."""
        await self.aclose()
        return False

    async def aclose(self) -> None:
        """Async close all child iterators."""
        for child in self._children:
            await child.aclose()


atee = Tee


class aclosing(AbstractAsyncContextManager):  # noqa: N801
    """Async context manager to wrap an AsyncGenerator that has a ``aclose()`` method.

    Code like this:

        async with aclosing(<module>.fetch(<arguments>)) as agen:
            <block>

    is equivalent to this:

        agen = <module>.fetch(<arguments>)
        try:
            <block>
        finally:
            await agen.aclose()

    """

    def __init__(
        self, thing: Union[AsyncGenerator[Any, Any], AsyncIterator[Any]]
    ) -> None:
        """Create the context manager.

        Args:
            thing: The resource to wrap.
        """
        self.thing = thing

    @override
    async def __aenter__(self) -> Union[AsyncGenerator[Any, Any], AsyncIterator[Any]]:
        return self.thing

    @override
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if hasattr(self.thing, "aclose"):
            await self.thing.aclose()


async def abatch_iterate(
    size: int, iterable: AsyncIterable[T]
) -> AsyncIterator[list[T]]:
    """Utility batching function for async iterables.

    Args:
        size: The size of the batch.
        iterable: The async iterable to batch.

    Returns:
        An async iterator over the batches.
    """
    batch: list[T] = []
    async for element in iterable:
        if len(batch) < size:
            batch.append(element)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch
