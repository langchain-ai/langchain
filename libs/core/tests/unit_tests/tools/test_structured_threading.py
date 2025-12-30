
import time
import asyncio
import pytest
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig

def slow_sync_tool(x: int) -> int:
    time.sleep(0.5)
    return x

@pytest.mark.asyncio
async def test_concurrency_respects_config():
    tool = StructuredTool.from_function(slow_sync_tool)
    
    start = time.time()
    # Use max_concurrency=1. This should force sequential execution if respected.
    config = RunnableConfig(max_concurrency=1)
    
    await asyncio.gather(
        tool.ainvoke({"x": 1}, config=config),
        tool.ainvoke({"x": 2}, config=config)
    )
    duration = time.time() - start
    
    print(f"Duration with max_concurrency=1: {duration:.2f}s")
    
    # If working correctly (sequential), should be >= 1.0s
    # If broken (parallel), should be ~0.5s
    assert duration >= 1.0
