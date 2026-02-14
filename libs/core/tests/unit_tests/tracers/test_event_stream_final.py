from collections.abc import AsyncIterator

import pytest

from langchain_core.runnables import RunnableLambda


@pytest.mark.asyncio
async def test_astream_events_include_final_only() -> None:
    """Test that include_final_only filters out intermediate stream events."""

    async def intermediate_step(x: str) -> AsyncIterator[str]:
        yield f"Inter-{x}"

    async def final_step(x: str) -> AsyncIterator[str]:
        yield f"Final-{x}"

    chain = RunnableLambda(intermediate_step) | RunnableLambda(final_step)

    # Use list comprehension to satisfy PERF401
    all_events = [
        event["data"]["chunk"]
        async for event in chain.astream_events("test", version="v2")
        if event["event"] == "on_chain_stream"
    ]

    assert "Inter-test" in all_events
    assert "Final-Inter-test" in all_events

    final_only_events = [
        event["data"]["chunk"]
        async for event in chain.astream_events(
            "test", version="v2", include_final_only=True
        )
        if event["event"] == "on_chain_stream"
    ]

    assert "Inter-test" not in final_only_events
    assert "Final-Inter-test" in final_only_events

    for chunk in final_only_events:
        assert "Inter-" not in chunk or "Final-" in chunk
