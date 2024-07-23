"""Check that the max time decorator works."""
import asyncio
import time

import pytest

from langchain_standard_tests.utils.timeout import _timeout


@_timeout(seconds=0.5)
def test_sync_fast() -> None:
    """Test function that completes within the allowed time."""
    time.sleep(0.01)


@_timeout(seconds=0.5)
async def test_async_fast() -> None:
    """Test async function that completes within the allowed time."""
    await asyncio.sleep(0.01)


@pytest.mark.xfail(strict=True)
@_timeout(seconds=0)
def test_sync_slow() -> None:
    """Test async function that exceeds the allowed time."""
    time.sleep(0.01)


@pytest.mark.xfail(strict=True)
@_timeout(seconds=0)
async def test_async_slow() -> None:
    """Test async function that exceeds the allowed time."""
    await asyncio.sleep(0.01)


class TestMethodDecoration:
    @_timeout(seconds=0.5)
    def test_sync_fast_method(self) -> None:
        """Test function that completes within the allowed time."""
        time.sleep(0.01)

    @_timeout(seconds=0.5)
    async def test_async_fast_method(self) -> None:
        """Test async function that completes within the allowed time."""
        await asyncio.sleep(0.01)

    @pytest.mark.xfail(strict=True)
    @_timeout(seconds=0)
    def test_sync_slow_method(self) -> None:
        """Test async function that exceeds the allowed time."""
        time.sleep(0.01)

    @pytest.mark.xfail(strict=True)
    @_timeout(seconds=0)
    async def test_async_slow_method(self) -> None:
        """Test async function that exceeds the allowed time."""
        await asyncio.sleep(0.01)
