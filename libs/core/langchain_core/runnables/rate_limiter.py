"""Interface and implementation for time based rate limiters.

This module defines an interface for rate limiting requests based on time.

The interface cannot account for the size of the request or any other factors.

The module also provides an in-memory implementation of the rate limiter.
"""

from __future__ import annotations

import abc
import asyncio
import threading
import time
from typing import (
    Any,
    Optional,
    cast,
)

from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import (
    Input,
    Output,
    Runnable,
)


@beta(message="Introduced in 0.2.24. API subject to change.")
class BaseRateLimiter(Runnable[Input, Output], abc.ABC):
    """Base class for rate limiters.

    Usage of the base limiter is through the acquire and aacquire methods depending
    on whether running in a sync or async context.

    Implementations are free to add a timeout parameter to their initialize method
    to allow users to specify a timeout for acquiring the necessary tokens when
    using a blocking call.

    Current limitations:

    - The rate limiter is not designed to work across different processes. It is
      an in-memory rate limiter, but it is thread safe.
    - The rate limiter only supports time-based rate limiting. It does not take
      into account the size of the request or any other factors.
    - The current implementation does not handle streaming inputs well and will
      consume all inputs even if the rate limit has not been reached. Better support
      for streaming inputs will be added in the future.
    - When the rate limiter is combined with another runnable via a RunnableSequence,
      usage of .batch() or .abatch() will only respect the average rate limit.
      There will be bursty behavior as .batch() and .abatch() wait for each step
      to complete before starting the next step. One way to mitigate this is to
      use batch_as_completed() or abatch_as_completed().

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

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Invoke the rate limiter.

        This is a blocking call that waits until the given number of tokens are
        available.

        Args:
            input: The input to the rate limiter.
            config: The configuration for the rate limiter.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the rate limiter.
        """

        def _invoke(input: Input) -> Output:
            """Invoke the rate limiter. Internal function."""
            self.acquire(blocking=True)
            return cast(Output, input)

        return self._call_with_config(_invoke, input, config, **kwargs)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Invoke the rate limiter. Async version.

        This is a blocking call that waits until the given number of tokens are
        available.

        Args:
            input: The input to the rate limiter.
            config: The configuration for the rate limiter.
            **kwargs: Additional keyword arguments.
        """

        async def _ainvoke(input: Input) -> Output:
            """Invoke the rate limiter. Internal function."""
            await self.aacquire(blocking=True)
            return cast(Output, input)

        return await self._acall_with_config(_ainvoke, input, config, **kwargs)


@beta(message="Introduced in 0.2.24. API subject to change.")
class InMemoryRateLimiter(BaseRateLimiter):
    """An in memory rate limiter.

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
    - The current implementation does not handle streaming inputs well and will
      consume all inputs even if the rate limit has not been reached. Better support
      for streaming inputs will be added in the future.
    - When the rate limiter is combined with another runnable via a RunnableSequence,
      usage of .batch() or .abatch() will only respect the average rate limit.
      There will be bursty behavior as .batch() and .abatch() wait for each step
      to complete before starting the next step. One way to mitigate this is to
      use batch_as_completed() or abatch_as_completed().

    Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda, InMemoryRateLimiter

            rate_limiter = InMemoryRateLimiter(
                requests_per_second=100, check_every_n_seconds=0.1, max_bucket_size=10
            )

            def foo(x: int) -> int:
                return x

            foo_ = RunnableLambda(foo)
            chain = rate_limiter | foo_
            assert chain.invoke(1) == 1

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
        self.last: Optional[float] = None
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
            await asyncio.sleep(self.check_every_n_seconds)
        return True
