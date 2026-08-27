"""Tests for the v3 dispatch path on Runnable.astream_events / stream_events."""

from __future__ import annotations

import pytest

from langchain_core.runnables import RunnableLambda


def _double(x: int) -> int:
    return x * 2


def test_v3_on_plain_runnable_raises_not_implemented_sync() -> None:
    runnable = RunnableLambda(_double)
    with pytest.raises(NotImplementedError, match="v3"):
        runnable.stream_events(2, version="v3")


async def test_v3_on_plain_runnable_raises_not_implemented_async() -> None:
    runnable = RunnableLambda(_double)
    with pytest.raises(NotImplementedError, match="v3"):
        await runnable.astream_events(2, version="v3")
