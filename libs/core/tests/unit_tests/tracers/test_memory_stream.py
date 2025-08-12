import asyncio
import math
import sys
import time
from collections.abc import AsyncIterator

from langchain_core.tracers.memory_stream import _MemoryStream


async def test_same_event_loop() -> None:
    """Test that the memory stream works when the same event loop is used.

    This is the easy case.
    """
    reader_loop = asyncio.get_event_loop()
    channel = _MemoryStream[dict](reader_loop)
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

    async def consumer() -> AsyncIterator[dict]:
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
    channel = _MemoryStream[dict](reader_loop)
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

    async def consumer() -> AsyncIterator[dict]:
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
        # Allow a generous 10ms of delay
        # The test is meant to verify that the producer and consumer are running in
        # parallel despite the fact that the producer is running from another thread.
        # abs_tol is used to allow for some delay in the producer and consumer
        # due to overhead.
        # To verify that the producer and consumer are running in parallel, we
        # expect the delta_time to be smaller than the sleep delay in the producer
        # * # of items = 30 ms
        tolerance = 0.03 if sys.version_info[:2] in [(3, 9), (3, 11)] else 0.02
        assert math.isclose(delta_time, 0, abs_tol=tolerance), (
            f"delta_time: {delta_time}"
        )


def test_send_to_closed_stream() -> None:
    """Test that sending to a closed stream doesn't raise an error.

    We may want to handle this in a better way in the future.
    """
    event_loop = asyncio.get_event_loop()
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
