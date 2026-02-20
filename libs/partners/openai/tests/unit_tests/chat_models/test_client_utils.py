"""Test client utility functions."""

from __future__ import annotations


def test_async_client_del_no_running_loop() -> None:
    """Test that __del__ does not raise when no event loop is running.

    Regression test for https://github.com/langchain-ai/langchain/issues/35327
    """
    import asyncio
    import gc

    from langchain_openai.chat_models._client_utils import _AsyncHttpxClientWrapper

    async def _create_client() -> _AsyncHttpxClientWrapper:
        return _AsyncHttpxClientWrapper(base_url="http://test", timeout=10)

    client = asyncio.run(_create_client())
    # After asyncio.run() the event loop is closed.
    # Dropping the reference and forcing GC should not raise RuntimeError.
    del client
    gc.collect()  # triggers __del__
