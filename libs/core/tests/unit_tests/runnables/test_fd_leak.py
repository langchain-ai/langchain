import asyncio
import pytest
from typing import AsyncGenerator
from langchain_core.runnables import RunnableLambda

@pytest.mark.asyncio
async def test_runnable_sequence_astream_cancellation_closes_generator():
    closed = False
    ready = asyncio.Event()

    async def mock_generator(input: str) -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            yield "chunk 1"
            ready.set()
            # wait forever to simulate hanging network request that gets cancelled
            await asyncio.sleep(100)
        finally:
            closed = True

    runnable1 = RunnableLambda(mock_generator)
    runnable2 = RunnableLambda(lambda x: x + "!")
    chain = runnable1 | runnable2

    async def consume():
        async for chunk in chain.astream("test"):
            pass

    # Run the consume coroutine
    task = asyncio.create_task(consume())
    
    # Wait until generator yields first chunk
    await ready.wait()
    
    # Now CANCEL the consumer task to trigger CancelledError
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Wait a tiny bit for background cleanup tasks to execute
    await asyncio.sleep(0.1)

    # Assert that the underlying generator was properly closed
    assert closed is True
