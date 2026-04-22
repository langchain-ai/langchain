"""Tests for async client rejection in Chroma."""

import inspect
from unittest.mock import MagicMock

import pytest

from langchain_chroma.vectorstores import Chroma


def test_rejects_unawaited_async_client() -> None:
    """Passing an unawaited AsyncHttpClient (a coroutine) should raise ValueError."""

    async def fake_async_client() -> MagicMock:
        return MagicMock()

    coro = fake_async_client()
    assert inspect.iscoroutine(coro)

    with pytest.raises(ValueError, match="async"):
        Chroma(client=coro, collection_name="test")  # type: ignore[arg-type]

    # Close the coroutine to avoid RuntimeWarning
    coro.close()


def test_rejects_awaited_async_client() -> None:
    """Passing an awaited AsyncClientAPI instance should raise ValueError."""
    import chromadb

    mock_async_client = MagicMock(spec=chromadb.AsyncClientAPI)

    with pytest.raises(ValueError, match="async"):
        Chroma(client=mock_async_client, collection_name="test")  # type: ignore[arg-type]
