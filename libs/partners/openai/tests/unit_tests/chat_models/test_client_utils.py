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
    del client
    gc.collect()


def test_async_client_del_closes_transport() -> None:
    """Test that __del__ actually closes the client when no loop is running.

    Regression test for https://github.com/langchain-ai/langchain/issues/35327
    """
    import asyncio

    from langchain_openai.chat_models._client_utils import _AsyncHttpxClientWrapper

    async def _create_client() -> _AsyncHttpxClientWrapper:
        return _AsyncHttpxClientWrapper(base_url="http://test", timeout=10)

    client = asyncio.run(_create_client())
    assert not client.is_closed
    client.__del__()
    assert client.is_closed
