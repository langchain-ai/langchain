"""Verify ChatFireworks does not orphan aiohttp.ClientSession objects.

The Fireworks SDK (>=0.19) eagerly creates an ``aiohttp.ClientSession``
inside ``FireworksClient.__init__`` when an async event loop is running.
If ``ChatFireworks`` discards the parent ``Fireworks`` /
``AsyncFireworks`` objects after extracting ``.chat.completions``, the
underlying sessions are never closed, producing ``Unclosed client
session`` warnings.

This test constructs the model inside an async context (matching how
LangGraph / Harbor invoke it) and asserts no sessions leak.
"""

import asyncio
import gc
import warnings

from pydantic import SecretStr

from langchain_fireworks import ChatFireworks


async def test_no_unclosed_aiohttp_sessions() -> None:
    """ChatFireworks must not leak aiohttp.ClientSession objects."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = ChatFireworks(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            api_key=SecretStr("test-key"),
        )

        del model
        gc.collect()
        # Yield control so weak-ref / __del__ callbacks can fire.
        await asyncio.sleep(0)
        gc.collect()

    unclosed = [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning)
        and "unclosed" in str(w.message).lower()
    ]
    assert unclosed == [], f"Leaked {len(unclosed)} unclosed session(s):\n" + "\n".join(
        str(w.message) for w in unclosed
    )
