"""Provides a rate limiter based on a token bucket algorithm.

This rate limiter is designed to only limit based on the number of requests
per second.

It does not take into account the size of the request or any other factors.
"""

from __future__ import annotations

import abc
import asyncio
import threading
import time
from typing import (
    Optional,
)

from langchain_core._api import beta
from langchain_core.runnables.base import (
    Input,
    Output,
    Runnable,
    RunnableLambda,
)


class BaseRateLimiter(abc.ABC):
    """Base class for rate limiters.

    Usage of the base limiter is through the sync_wait and async_wait methods depending
    on whether running in a sync or async context.

    .. versionadded:: 0.2.24
    """

    @abc.abstractmethod
    def sync_wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""

    @abc.abstractmethod
    async def async_wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""


class InMemoryRateLimiter(BaseRateLimiter):
    """An in memory rate limiter.

    This is an in memory rate limiter, so it cannot rate limit across
    different processes.

    The rate limiter only allows time-based rate limiting and does not
    take into account any information about the input or the output, so it
    cannot be used to rate limit based on the size of the request.

    It is thread safe and can be used in either a sync or async context.

    .. versionadded:: 0.2.24
    """

    def __init__(
        self,
        *,
        requests_per_second: float = 1,
        check_every_n_seconds: float = 0.1,
        max_bucket_size: float = 1,
    ) -> None:
        """A rate limiter based on a token bucket.

        These *tokens* have NOTHING to do with LLM tokens. They are just
        a way to keep track of how many requests can be made at a given time.

        This rate limiter is designed to work in a threaded environment.

        It works by filling up a bucket with tokens at a given rate. Each
        request consumes a given number of tokens. If there are not enough
        tokens in the bucket, the request is blocked until there are enough
        tokens.

        Args:
            requests_per_second: The number of tokens to add per second to the bucket.
                Must be at least 1. The tokens represent "credit" that can be used
                to make requests.
            check_every_n_seconds: check whether the tokens are available
                every this many seconds. Can be a float to represent
                fractions of a second.
            max_bucket_size: The maximum number of tokens that can be in the bucket.
                This is used to prevent bursts of requests.
        """
        # Number of requests that we can make per second.
        self.requests_per_second = requests_per_second
        # Number of tokens in the bucket.
        self.available_tokens = 0.0
        self.max_bucket_size = max_bucket_size
        # A lock to ensure that tokens can only be consumed by one thread
        # at a given time.
        self._consume_lock = threading.Lock()
        # The last time we tried to consume tokens.
        self.last: Optional[time.time] = None
        self.check_every_n_seconds = check_every_n_seconds

    def _consume(self) -> bool:
        """Consume the given amount of tokens if possible.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """
        with self._consume_lock:
            now = time.time()

            # initialize on first call to avoid a burst
            if self.last is None:
                self.last = now

            elapsed = now - self.last

            if elapsed * self.requests_per_second >= 1:
                self.available_tokens += elapsed * self.requests_per_second
                self.last = now

            # Make sure that we don't exceed the bucket size.
            # This is used to prevent bursts of requests.
            self.available_tokens = min(self.available_tokens, self.max_bucket_size)

            # As long as we have at least one token, we can proceed.
            if self.available_tokens >= 1:
                self.available_tokens -= 1
                return True

            return False

    def sync_wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""
        while not self._consume():
            time.sleep(self.check_every_n_seconds)

    async def _aconsume(self) -> bool:
        """Consume the given amount of tokens if possible. Async variant.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """
        return self._consume()

    async def async_wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""
        while not await self._aconsume():
            await asyncio.sleep(self.check_every_n_seconds)


@beta(message="API was added in 0.2.24")
def add_rate_limiter(
    runnable: Runnable[Input, Output],
    rate_limiter: BaseRateLimiter,
) -> Runnable[Input, Output]:
    """Prepends a rate limiter in front of the given runnable.

    The rate limiter will be used to throttle the requests to the runnable.

    .. code-block:: python

        from langchain_core.runnables.rate_limiter import (
            InMemoryRateLimiter,
            add_rate_limiter
        )

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=2, check_every_n_seconds=0.1, max_bucket_size=2
        )

        runnable = RunnableLambda(lambda x: x)
        runnable_with_rate_limiter = add_rate_limiter(runnable, rate_limiter)

        # Now runnable_with_rate_limiter will only allow 2 requests per second.
        runnable_with_rate_limiter.invoke(1)

    Args:
        runnable: The runnable to throttle.
        rate_limiter: The throttle to use.

    Returns:
        A runnable lambda that acts as a throttled passthrough.

    .. versionadded:: 0.2.24
    """

    def _wait(input: dict) -> dict:
        """Wait for the rate limiter to allow the request to proceed."""
        rate_limiter.sync_wait()
        return input

    async def _await(input: dict) -> dict:
        """Wait for the rate limiter to allow the request to proceed."""
        await rate_limiter.async_wait()
        return input

    return (
        RunnableLambda(_wait, afunc=_await).with_config({"name": "RunnableWait"})
        | runnable
    )
