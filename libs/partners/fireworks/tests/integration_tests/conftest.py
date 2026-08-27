"""Shared fixtures for `ChatFireworks` integration tests.

The 1.x `fireworks-ai` SDK defaults to an aiohttp-backed httpx transport for
`AsyncFireworks`. Each test constructs its own `ChatFireworks`, which opens a
TCP connector lazily on first call. Without explicit cleanup, the connector is
finalized by GC *after* `pytest-asyncio` has stopped the event loop, producing
an `Unclosed connector` warning at teardown.

This conftest tracks every `ChatFireworks` instance created during a test and
calls `aclose()` on it before the loop closes.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import AsyncIterator
from typing import Any

import pytest

from langchain_fireworks import ChatFireworks

# `ChatFireworks` (a Pydantic `BaseModel`) is not hashable, so a `WeakSet`
# does not work; track via weak references keyed by `id()`.
_live_models: dict[int, weakref.ref[ChatFireworks]] = {}
_original_init = ChatFireworks.__init__


def _tracking_init(self: ChatFireworks, *args: Any, **kwargs: Any) -> None:
    _original_init(self, *args, **kwargs)
    _live_models[id(self)] = weakref.ref(self)


@pytest.fixture(autouse=True)
async def _close_chat_fireworks_clients() -> AsyncIterator[None]:
    """Close every `ChatFireworks` created during the test.

    Yields control to the test, then walks the live-instance map and awaits
    each model's `aclose()` while the event loop is still alive.
    """
    ChatFireworks.__init__ = _tracking_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        ChatFireworks.__init__ = _original_init  # type: ignore[method-assign]
        for ref in list(_live_models.values()):
            model = ref()
            if model is not None:
                await model.aclose()
        _live_models.clear()
        gc.collect()
