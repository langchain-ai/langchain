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
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
)

from langchain_core.runnables.base import (
    Input,
    Output,
    Runnable,
    RunnableBindingBase,
    RunnableLambda,
)

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class BaseRateLimiter(abc.ABC):
    """Base class for rate limiters.

    Usage of the base limiter is through the wait and await_ methods depending
    on whether running in a sync or async context.
    """

    @abc.abstractmethod
    def _consume(self) -> bool:
        """Consume the given amount of tokens if possible.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """

    @abc.abstractmethod
    async def _aconsume(self) -> bool:
        """Consume the given amount of tokens if possible.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """

    @abc.abstractmethod
    def wait(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""

    @abc.abstractmethod
    async def await_(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""


class RateLimiter(BaseRateLimiter):
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

    def wait(self) -> None:
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

    async def await_(self) -> None:
        """Blocking call to wait until the given number of tokens are available."""
        while not await self._aconsume():
            await asyncio.sleep(self.check_every_n_seconds)


def with_rate_limit(
    runnable: Runnable[Input, Output],
    rate_limiter: BaseRateLimiter,
) -> Runnable[Input, Output]:
    """Add a rate limiter to the runnable.

    Args:
        runnable: The runnable to throttle.
        rate_limiter: The throttle to use.

    Returns:
        A runnable lambda that acts as a throttled passthrough.
    """

    def _wait(input: dict) -> dict:
        """Wait for the rate limiter to allow the request to proceed."""
        rate_limiter.wait()
        return input

    async def _await(input: dict) -> dict:
        """Wait for the rate limiter to allow the request to proceed."""
        await rate_limiter.await_()
        return input

    return (
        RunnableLambda(_wait, afunc=_await).with_config({"name": "RunnableWait"})
        | runnable
    )


class RunnableRateLimiter(RunnableBindingBase[Input, Output]):
    """A rate limiter that can be used with runnables."""

    rate_limiter: BaseRateLimiter

    def __init__(
        self, *, bound: Runnable, rate_limiter: Optional[BaseRateLimiter], **kwargs: Any
    ) -> None:
        """Create a new rate limiter."""
        bound = with_rate_limit(bound, rate_limiter)
        super().__init__(rate_limiter=rate_limiter, bound=bound, **kwargs)
