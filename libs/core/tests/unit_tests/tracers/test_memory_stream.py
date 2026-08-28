import asyncio
import gc
import math
import threading
import time
import weakref
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from langchain_core.tracers.event_stream import _AstreamEventsCallbackHandler
from langchain_core.tracers.log_stream import LogStreamCallbackHandler
from langchain_core.tracers.memory_stream import _close_loop_quietly, _MemoryStream

T = TypeVar("T")


async def test_same_event_loop() -> None:
    """Test that the memory stream works when the same event loop is used.

    This is the easy case.
    """
    reader_loop = asyncio.get_event_loop()
    channel = _MemoryStream[dict[str, int | float]](reader_loop)
    writer = channel.get_send_stream()
    reader = channel.get_receive_stream()

    async def producer() -> None:
        """Produce items with slight delay."""
        tic = time.time()
        for i in range(3):
            await asyncio.sleep(0.10)
            toc = time.time()
            await writer.send(
                {
                    "item": i,
                    "produce_time": toc - tic,
                }
            )
        await writer.aclose()

    async def consumer() -> AsyncIterator[dict[str, int | float]]:
        tic = time.time()
        async for item in reader:
            toc = time.time()
            yield {
                "receive_time": toc - tic,
                **item,
            }

    producer_task = asyncio.create_task(producer())

    items = [item async for item in consumer()]

    for item in items:
        delta_time = item["receive_time"] - item["produce_time"]
        # Allow a generous 10ms of delay
        # The test is meant to verify that the producer and consumer are running in
        # parallel despite the fact that the producer is running from another thread.
        # abs_tol is used to allow for some delay in the producer and consumer
        # due to overhead.
        # To verify that the producer and consumer are running in parallel, we
        # expect the delta_time to be smaller than the sleep delay in the producer
        # * # of items = 30 ms
        assert math.isclose(delta_time, 0, abs_tol=0.010) is True, (
            f"delta_time: {delta_time}"
        )

    await producer_task


async def test_queue_for_streaming_via_sync_call() -> None:
    """Test via async -> sync -> async path."""
    reader_loop = asyncio.get_event_loop()
    channel = _MemoryStream[dict[str, int | float]](reader_loop)
    writer = channel.get_send_stream()
    reader = channel.get_receive_stream()

    async def producer() -> None:
        """Produce items with slight delay."""
        tic = time.time()
        for i in range(3):
            await asyncio.sleep(0.2)
            toc = time.time()
            await writer.send(
                {
                    "item": i,
                    "produce_time": toc - tic,
                }
            )
        await writer.aclose()

    def sync_call() -> None:
        """Blocking sync call."""
        asyncio.run(producer())

    async def consumer() -> AsyncIterator[dict[str, int | float]]:
        tic = time.time()
        async for item in reader:
            toc = time.time()
            yield {
                "receive_time": toc - tic,
                **item,
            }

    task = asyncio.create_task(asyncio.to_thread(sync_call))
    items = [item async for item in consumer()]
    await task

    assert len(items) == 3

    for item in items:
        delta_time = item["receive_time"] - item["produce_time"]
        # The test verifies that the producer and consumer are running in parallel
        # despite the producer running from another thread via asyncio.to_thread.
        # Cross-thread communication has overhead that varies with system load,
        # so we use a tolerance of 150ms. This still proves parallelism because
        # serial execution would show deltas of 200ms+ (the sleep interval).
        assert math.isclose(delta_time, 0, abs_tol=0.15) is True, (
            f"delta_time: {delta_time}"
        )


def test_send_to_closed_stream() -> None:
    """Test that sending to a closed stream doesn't raise an error.

    We may want to handle this in a better way in the future.
    """
    event_loop = asyncio.new_event_loop()
    channel = _MemoryStream[str](event_loop)
    writer = channel.get_send_stream()
    # send with an open even loop
    writer.send_nowait("hello")
    event_loop.close()
    writer.send_nowait("hello")
    # now close the loop
    event_loop.close()
    writer.close()
    writer.send_nowait("hello")


async def test_closed_stream() -> None:
    reader_loop = asyncio.get_event_loop()
    channel = _MemoryStream[str](reader_loop)
    writer = channel.get_send_stream()
    reader = channel.get_receive_stream()
    await writer.aclose()

    assert [chunk async for chunk in reader] == []


def _run_in_thread(fn: Callable[[], T]) -> T:
    """Run `fn` in a worker thread and return its result.

    Two reasons for the thread: a worker thread never has an implicit current
    event loop on any supported Python version, which is what forces the
    `asyncio.new_event_loop()` fallback in `_get_or_create_loop`; and event loop
    policy state is thread-local, so any loop set here cannot leak into the rest
    of the test session and change which branch later tests take.
    """
    result: list[T] = []
    thread = threading.Thread(target=lambda: result.append(fn()))
    thread.start()
    thread.join()
    assert len(result) == 1, "worker thread raised before returning (see stderr)"
    return result[0]


def _make_handlers() -> list[Any]:
    """Construct one handler of each streaming type from a sync context."""
    return [_AstreamEventsCallbackHandler(), LogStreamCallbackHandler()]


def test_internally_created_event_loop_is_closed() -> None:
    """Regression test for `ResourceWarning: unclosed event loop` at GC time.

    When a streaming callback handler is constructed from a synchronous context
    with no current event loop, it falls back to `asyncio.new_event_loop()`.
    That loop must be closed once the handler is garbage collected; otherwise
    the loop's `__del__` emits a `ResourceWarning` at whatever nondeterministic
    point GC runs. Landing inside a `warnings.catch_warnings(record=True)`
    block turns the leak into an order-dependent flaky failure.

    Asserts the loop is positively closed rather than merely that no warning
    appeared, so the test cannot pass vacuously if the handler stops being
    collectible (e.g. via a global registry or a reference cycle).
    """
    handlers = _run_in_thread(_make_handlers)
    loops = [handler.send_stream._reader_loop for handler in handlers]
    assert len(loops) == 2
    assert all(not loop.is_closed() for loop in loops)

    refs = [weakref.ref(handler) for handler in handlers]
    handlers.clear()
    gc.collect()

    assert all(ref() is None for ref in refs), (
        "handlers were not collected, so the assertion below would be vacuous"
    )
    assert all(loop.is_closed() for loop in loops), (
        "finalizer did not close the internally created event loop"
    )


def test_caller_owned_event_loop_is_not_closed() -> None:
    """A loop we did not create must be left alone when the handler is GCed.

    The complement of `test_internally_created_event_loop_is_closed`, and the
    more dangerous half to get wrong: closing a loop the caller owns would
    break user code at an arbitrary GC point, silently, since
    `_close_loop_quietly` suppresses errors.
    """

    def construct() -> tuple[asyncio.AbstractEventLoop, list[Any]]:
        caller_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(caller_loop)
        return caller_loop, _make_handlers()

    caller_loop, handlers = _run_in_thread(construct)
    try:
        assert all(
            handler.send_stream._reader_loop is caller_loop for handler in handlers
        ), "handlers did not adopt the caller's loop, so this test proves nothing"

        handlers.clear()
        gc.collect()

        assert not caller_loop.is_closed(), (
            "a caller-owned event loop must not be closed when the handler that "
            "borrowed it is garbage collected"
        )
    finally:
        caller_loop.close()


def test_close_loop_quietly_leaves_running_loop_open() -> None:
    """A running loop cannot be closed, so the finalizer must not try."""

    async def close_self_while_running() -> bool:
        loop = asyncio.get_running_loop()
        _close_loop_quietly(loop)
        return loop.is_closed()

    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(close_self_while_running()) is False
    finally:
        loop.close()

    # Idempotent on an already-closed loop.
    _close_loop_quietly(loop)
    assert loop.is_closed()
