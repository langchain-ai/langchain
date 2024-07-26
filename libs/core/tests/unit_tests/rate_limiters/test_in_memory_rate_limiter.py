"""Test rate limiter."""

import time

import pytest
from freezegun import freeze_time

from langchain_core.rate_limiters import InMemoryRateLimiter


@pytest.fixture
def rate_limiter() -> InMemoryRateLimiter:
    """Return an instance of InMemoryRateLimiter."""
    return InMemoryRateLimiter(
        requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=2
    )


def test_initial_state(rate_limiter: InMemoryRateLimiter) -> None:
    """Test the initial state of the rate limiter."""
    assert rate_limiter.available_tokens == 0.0


def test_sync_wait(rate_limiter: InMemoryRateLimiter) -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter.last = time.time()
        assert not rate_limiter.acquire(blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not rate_limiter.acquire(blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not rate_limiter.acquire(blocking=False)
        frozen_time.tick(1.8)
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 1.0
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 0
        frozen_time.tick(2.1)
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 1
        frozen_time.tick(0.9)
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 1

        # Check max bucket size
        frozen_time.tick(100)
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 1


async def test_async_wait(rate_limiter: InMemoryRateLimiter) -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter.last = time.time()
        assert not await rate_limiter.aacquire(blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not await rate_limiter.aacquire(blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not await rate_limiter.aacquire(blocking=False)
        frozen_time.tick(1.8)
        assert await rate_limiter.aacquire(blocking=False)
        assert rate_limiter.available_tokens == 1.0
        assert await rate_limiter.aacquire(blocking=False)
        assert rate_limiter.available_tokens == 0
        frozen_time.tick(2.1)
        assert await rate_limiter.aacquire(blocking=False)
        assert rate_limiter.available_tokens == 1
        frozen_time.tick(0.9)
        assert await rate_limiter.aacquire(blocking=False)
        assert rate_limiter.available_tokens == 1


def test_sync_wait_max_bucket_size() -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=500
        )
        rate_limiter.last = time.time()
        frozen_time.tick(100)  # Increment by 100 seconds
        assert rate_limiter.acquire(blocking=False)
        # After 100 seconds we manage to refill the bucket with 200 tokens
        # After consuming 1 token, we should have 199 tokens left
        assert rate_limiter.available_tokens == 199.0
        frozen_time.tick(10000)
        assert rate_limiter.acquire(blocking=False)
        assert rate_limiter.available_tokens == 499.0
        # Assert that sync wait can proceed without blocking
        # since we have enough tokens
        rate_limiter.acquire(blocking=True)


async def test_async_wait_max_bucket_size() -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=500
        )
        rate_limiter.last = time.time()
        frozen_time.tick(100)  # Increment by 100 seconds
        assert await rate_limiter.aacquire(blocking=False)
        # After 100 seconds we manage to refill the bucket with 200 tokens
        # After consuming 1 token, we should have 199 tokens left
        assert rate_limiter.available_tokens == 199.0
        frozen_time.tick(10000)
        assert await rate_limiter.aacquire(blocking=False)
        assert rate_limiter.available_tokens == 499.0
        # Assert that sync wait can proceed without blocking
        # since we have enough tokens
        await rate_limiter.aacquire(blocking=True)
