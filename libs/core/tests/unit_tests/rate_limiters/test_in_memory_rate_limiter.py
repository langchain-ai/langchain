"""Test rate limiter."""

import time

import pytest
from freezegun import freeze_time

from langchain_core.language_models import GenericFakeChatModel
from langchain_core.rate_limiters import InMemoryChatModelRateLimiter


@pytest.fixture
def rate_limiter() -> InMemoryChatModelRateLimiter:
    """Return an instance of InMemoryRateLimiter."""
    return InMemoryChatModelRateLimiter(
        requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=2
    )


def test_initial_state(rate_limiter: InMemoryChatModelRateLimiter) -> None:
    """Test the initial state of the rate limiter."""
    assert rate_limiter.available_tokens == 0.0


def test_sync_wait(rate_limiter: InMemoryChatModelRateLimiter) -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter.last = time.time()
        assert not rate_limiter.acquire([], blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not rate_limiter.acquire([], blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not rate_limiter.acquire([], blocking=False)
        frozen_time.tick(1.8)
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 1.0
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 0
        frozen_time.tick(2.1)
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 1
        frozen_time.tick(0.9)
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 1

        # Check max bucket size
        frozen_time.tick(100)
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 1


async def test_async_wait(rate_limiter: InMemoryChatModelRateLimiter) -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter.last = time.time()
        assert not await rate_limiter.aacquire([], blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not await rate_limiter.aacquire([], blocking=False)
        frozen_time.tick(0.1)  # Increment by 0.1 seconds
        assert rate_limiter.available_tokens == 0
        assert not await rate_limiter.aacquire([], blocking=False)
        frozen_time.tick(1.8)
        assert await rate_limiter.aacquire([], blocking=False)
        assert rate_limiter.available_tokens == 1.0
        assert await rate_limiter.aacquire([], blocking=False)
        assert rate_limiter.available_tokens == 0
        frozen_time.tick(2.1)
        assert await rate_limiter.aacquire([], blocking=False)
        assert rate_limiter.available_tokens == 1
        frozen_time.tick(0.9)
        assert await rate_limiter.aacquire([], blocking=False)
        assert rate_limiter.available_tokens == 1


def test_sync_wait_max_bucket_size() -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryChatModelRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=500
        )
        rate_limiter.last = time.time()
        frozen_time.tick(100)  # Increment by 100 seconds
        assert rate_limiter.acquire([], blocking=False)
        # After 100 seconds we manage to refill the bucket with 200 tokens
        # After consuming 1 token, we should have 199 tokens left
        assert rate_limiter.available_tokens == 199.0
        frozen_time.tick(10000)
        assert rate_limiter.acquire([], blocking=False)
        assert rate_limiter.available_tokens == 499.0
        # Assert that sync wait can proceed without blocking
        # since we have enough tokens
        rate_limiter.acquire([], blocking=True)


async def test_async_wait_max_bucket_size() -> None:
    with freeze_time("2023-01-01 00:00:00") as frozen_time:
        rate_limiter = InMemoryChatModelRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=500
        )
        rate_limiter.last = time.time()
        frozen_time.tick(100)  # Increment by 100 seconds
        assert await rate_limiter.aacquire([], blocking=False)
        # After 100 seconds we manage to refill the bucket with 200 tokens
        # After consuming 1 token, we should have 199 tokens left
        assert rate_limiter.available_tokens == 199.0
        frozen_time.tick(10000)
        assert await rate_limiter.aacquire([], blocking=False)
        assert rate_limiter.available_tokens == 499.0
        # Assert that sync wait can proceed without blocking
        # since we have enough tokens
        await rate_limiter.aacquire([], blocking=True)


def test_rate_limit_invoke() -> None:
    """Add rate limiter."""

    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02

    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # The second time we call the model, we should have 1 extra token
    # to proceed immediately.
    assert toc - tic < 0.005

    # The third time we call the model, we need to wait again for a token
    tic = time.time()
    model.invoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02


async def test_rate_limit_ainvoke() -> None:
    """Add rate limiter."""

    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02

    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # The second time we call the model, we should have 1 extra token
    # to proceed immediately.
    assert toc - tic < 0.005

    # The third time we call the model, we need to wait again for a token
    tic = time.time()
    await model.ainvoke("foo")
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert 0.01 < toc - tic < 0.02


def test_rate_limit_batch() -> None:
    """Test that batch and stream calls work with rate limiters."""
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Need 2 tokens to proceed
    time_to_fill = 2 / 200.0
    tic = time.time()
    model.batch(["foo", "foo"])
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert time_to_fill < toc - tic < time_to_fill + 0.01


async def test_rate_limit_abatch() -> None:
    """Test that batch and stream calls work with rate limiters."""
    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Need 2 tokens to proceed
    time_to_fill = 2 / 200.0
    tic = time.time()
    await model.abatch(["foo", "foo"])
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    # with 0 tokens.
    assert time_to_fill < toc - tic < time_to_fill + 0.01


def test_rate_limit_stream() -> None:
    """Test rate limit by stream."""
    model = GenericFakeChatModel(
        messages=iter(["hello world", "hello world", "hello world"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Check astream
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds

    # Second time around we should have 1 token left
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert toc - tic < 0.005  # Slightly smaller than check every n seconds

    # Third time around we should have 0 tokens left
    tic = time.time()
    response = list(model.stream("foo"))
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds


async def test_rate_limit_astream() -> None:
    """Test rate limiting astream."""
    model = GenericFakeChatModel(
        messages=iter(["hello world", "hello world", "hello world"]),
        rate_limiter=InMemoryChatModelRateLimiter(
            requests_per_second=200, check_every_n_seconds=0.01, max_bucket_size=10
        ),
    )
    # Check astream
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds

    # Second time around we should have 1 token left
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert toc - tic < 0.005  # Slightly smaller than check every n seconds

    # Third time around we should have 0 tokens left
    tic = time.time()
    response = [chunk async for chunk in model.astream("foo")]
    assert [msg.content for msg in response] == ["hello", " ", "world"]
    toc = time.time()
    # Should be larger than check every n seconds since the token bucket starts
    assert 0.01 < toc - tic < 0.02  # Slightly smaller than check every n seconds
