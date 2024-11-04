"""Interface for a rate limiter and an in-memory rate limiter."""

from __future__ import annotations

import abc
import asyncio
import threading
import time
from typing import (
    Optional,
)

from langchain_core._api import beta


@beta(message="Introduced in 0.2.24. API subject to change.")
class BaseRateLimiter(abc.ABC):
    """Base class for rate limiters.

    Usage of the base limiter is through the acquire and aacquire methods depending
    on whether running in a sync or async context.

    Implementations are free to add a timeout parameter to their initialize method
    to allow users to specify a timeout for acquiring the necessary tokens when
    using a blocking call.

    Current limitations:

    - Rate limiting information is not surfaced in tracing or callbacks. This means
      that the total time it takes to invoke a chat model will encompass both
      the time spent waiting for tokens and the time spent making the request.


    .. versionadded:: 0.2.24
    """

    @abc.abstractmethod
    def acquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire the necessary tokens for the rate limiter.

        This method blocks until the required tokens are available if `blocking`
        is set to True.

        If `blocking` is set to False, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If True, the method will block until the tokens are available.
                If False, the method will return immediately with the result of
                the attempt. Defaults to True.

        Returns:
           True if the tokens were successfully acquired, False otherwise.
        """

    @abc.abstractmethod
    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire the necessary tokens for the rate limiter.

        This method blocks until the required tokens are available if `blocking`
        is set to True.

        If `blocking` is set to False, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If True, the method will block until the tokens are available.
                If False, the method will return immediately with the result of
                the attempt. Defaults to True.

        Returns:
           True if the tokens were successfully acquired, False otherwise.
        """


@beta(message="Introduced in 0.2.24. API subject to change.")
class InMemoryRateLimiter(BaseRateLimiter):
    """An in memory rate limiter based on a token bucket algorithm.

    This is an in memory rate limiter, so it cannot rate limit across
    different processes.

    The rate limiter only allows time-based rate limiting and does not
    take into account any information about the input or the output, so it
    cannot be used to rate limit based on the size of the request.

    It is thread safe and can be used in either a sync or async context.

    The in memory rate limiter is based on a token bucket. The bucket is filled
    with tokens at a given rate. Each request consumes a token. If there are
    not enough tokens in the bucket, the request is blocked until there are
    enough tokens.

    These *tokens* have NOTHING to do with LLM tokens. They are just
    a way to keep track of how many requests can be made at a given time.

    Current limitations:

    - The rate limiter is not designed to work across different processes. It is
      an in-memory rate limiter, but it is thread safe.
    - The rate limiter only supports time-based rate limiting. It does not take
      into account the size of the request or any other factors.

    Example:

        .. code-block:: python

            import time

            from langchain_core.rate_limiters import InMemoryRateLimiter

            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.1,  # <-- Can only make a request once every 10 seconds!!
                check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
                max_bucket_size=10,  # Controls the maximum burst size.
            )

            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(
                model_name="claude-3-opus-20240229",
                rate_limiter=rate_limiter
            )

            for _ in range(5):
                tic = time.time()
                model.invoke("hello")
                toc = time.time()
                print(toc - tic)


    .. versionadded:: 0.2.24
    """  # noqa: E501

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
        self.last: Optional[float] = None
        self.check_every_n_seconds = check_every_n_seconds

    def _consume(self) -> bool:
        """Try to consume a token.

        Returns:
            True means that the tokens were consumed, and the caller can proceed to
            make the request. A False means that the tokens were not consumed, and
            the caller should try again later.
        """
        with self._consume_lock:
            now = time.monotonic()

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

    def acquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter.

        This method blocks until the required tokens are available if `blocking`
        is set to True.

        If `blocking` is set to False, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If True, the method will block until the tokens are available.
                If False, the method will return immediately with the result of
                the attempt. Defaults to True.

        Returns:
           True if the tokens were successfully acquired, False otherwise.
        """
        if not blocking:
            return self._consume()

        while not self._consume():
            time.sleep(self.check_every_n_seconds)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter. Async version.

        This method blocks until the required tokens are available if `blocking`
        is set to True.

        If `blocking` is set to False, the method will immediately return the result
        of the attempt to acquire the tokens.

        Args:
            blocking: If True, the method will block until the tokens are available.
                If False, the method will return immediately with the result of
                the attempt. Defaults to True.

        Returns:
           True if the tokens were successfully acquired, False otherwise.
        """
        if not blocking:
            return self._consume()

        while not self._consume():
            # This code ignores the ASYNC110 warning which is a false positive in this
            # case.
            # There is no external actor that can mark that the Event is done
            # since the tokens are managed by the rate limiter itself.
            # It needs to wake up to re-fill the tokens.
            # https://docs.astral.sh/ruff/rules/async-busy-wait/
            await asyncio.sleep(self.check_every_n_seconds)  # ruff: noqa: ASYNC110
        return True


__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
]
