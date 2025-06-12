"""Interface for a rate limiter and an in-memory rate limiter."""

from __future__ import annotations

import abc
import asyncio
import logging
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

# Optional Redis imports - will be None if not available
coredis = None
try:
    import coredis
except ImportError:
    pass


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

        while not self._consume():  # noqa: ASYNC110
            # This code ignores the ASYNC110 warning which is a false positive in this
            # case.
            # There is no external actor that can mark that the Event is done
            # since the tokens are managed by the rate limiter itself.
            # It needs to wake up to re-fill the tokens.
            # https://docs.astral.sh/ruff/rules/async-busy-wait/
            await asyncio.sleep(self.check_every_n_seconds)
        return True


class RedisRateLimiter(BaseRateLimiter):
    """A Redis-based distributed rate limiter using sliding window algorithm.
    
    This rate limiter provides distributed rate limiting across multiple processes,
    pods, or containers using Redis as the shared state store. It uses a sliding
    window counter algorithm implemented with Redis sorted sets and Lua scripts
    for atomic operations.
    
    Key features:
    - Distributed rate limiting across multiple processes/pods
    - Sliding window algorithm for accurate rate limiting
    - Atomic operations using Lua scripts
    - Connection pooling and Redis cluster support
    - Comprehensive error handling with fallback options
    - High performance with minimal Redis round trips
    
    Example:
        .. code-block:: python
        
            from langchain_core.rate_limiters import RedisRateLimiter
            
            # Basic usage
            rate_limiter = RedisRateLimiter(
                redis_url="redis://localhost:6379",
                requests_per_second=10,
                window_size=60,  # 60 second sliding window
            )
            
            # With Redis cluster
            rate_limiter = RedisRateLimiter(
                redis_url="redis://node1:7000,redis://node2:7000,redis://node3:7000",
                requests_per_second=100,
                window_size=60,
                redis_cluster=True,
            )
            
            # Use with chat model
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(
                model_name="claude-3-opus-20240229",
                rate_limiter=rate_limiter
            )
    
    .. versionadded:: 0.2.25
    """
    
    # Lua script for atomic rate limit checking and updating
    _CHECK_AND_UPDATE_SCRIPT = """
    local key = KEYS[1]
    local window_size = tonumber(ARGV[1])
    local max_requests = tonumber(ARGV[2])
    local current_time = tonumber(ARGV[3])
    local cleanup_probability = tonumber(ARGV[4])
    
    -- Remove expired entries (sliding window cleanup)
    local cutoff_time = current_time - window_size
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_time)
    
    -- Count current requests in window
    local current_count = redis.call('ZCARD', key)
    
    -- Check if we can proceed
    if current_count < max_requests then
        -- Add current request with unique score to handle concurrent requests
        local score = current_time + math.random() * 0.001
        redis.call('ZADD', key, score, score)
        
        -- Set expiration for the key (window_size + buffer)
        redis.call('EXPIRE', key, window_size + 60)
        
        -- Probabilistic cleanup of old entries
        if math.random() < cleanup_probability then
            redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_time - window_size)
        end
        
        return {1, current_count + 1, max_requests}
    else
        return {0, current_count, max_requests}
    end
    """
    
    # Lua script for non-blocking rate limit check
    _CHECK_ONLY_SCRIPT = """
    local key = KEYS[1]
    local window_size = tonumber(ARGV[1])
    local max_requests = tonumber(ARGV[2])
    local current_time = tonumber(ARGV[3])
    
    -- Remove expired entries
    local cutoff_time = current_time - window_size
    redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_time)
    
    -- Count current requests in window
    local current_count = redis.call('ZCARD', key)
    
    -- Return availability without consuming
    if current_count < max_requests then
        return {1, current_count, max_requests}
    else
        return {0, current_count, max_requests}
    end
    """
    
    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        requests_per_second: float = 1,
        window_size: int = 60,
        key_prefix: str = "langchain:ratelimit",
        identifier: Optional[str] = None,
        redis_cluster: bool = False,
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        fallback_to_memory: bool = True,
        cleanup_probability: float = 0.1,
        check_every_n_seconds: float = 0.1,
        **redis_kwargs: Any,
    ) -> None:
        """Initialize Redis-based distributed rate limiter.
        
        Args:
            redis_url: Redis connection URL. For clusters, provide comma-separated URLs.
            requests_per_second: Maximum requests allowed per second.
            window_size: Sliding window size in seconds.
            key_prefix: Prefix for Redis keys to avoid collisions.
            identifier: Unique identifier for this rate limiter instance.
                If None, will be auto-generated.
            redis_cluster: Whether to use Redis cluster mode.
            connection_pool_size: Size of the Redis connection pool.
            socket_timeout: Socket timeout for Redis operations.
            socket_connect_timeout: Socket connection timeout.
            retry_on_timeout: Whether to retry on timeout errors.
            health_check_interval: Interval for Redis health checks.
            fallback_to_memory: Whether to fallback to InMemoryRateLimiter on Redis errors.
            cleanup_probability: Probability of performing cleanup on each request (0.0-1.0).
            check_every_n_seconds: Sleep interval when blocking and waiting for tokens.
            **redis_kwargs: Additional Redis client configuration.
        """
        if coredis is None:
            raise ImportError(
                "RedisRateLimiter requires the 'coredis' package. "
                "Install it with: pip install coredis"
            )
        
        self.redis_url = redis_url
        self.requests_per_second = requests_per_second
        self.window_size = window_size
        self.max_requests = int(requests_per_second * window_size)
        self.key_prefix = key_prefix
        self.identifier = identifier or f"rl_{int(time.time())}_{random.randint(1000, 9999)}"
        self.redis_cluster = redis_cluster
        self.connection_pool_size = connection_pool_size
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.fallback_to_memory = fallback_to_memory
        self.cleanup_probability = cleanup_probability
        self.check_every_n_seconds = check_every_n_seconds
        self.redis_kwargs = redis_kwargs
        
        # Redis client and connection management
        self._redis_client: Optional[Union[coredis.Redis, coredis.RedisCluster]] = None
        self._connection_lock = threading.Lock()
        self._last_health_check = 0.0
        self._redis_healthy = True
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60.0
        
        # Fallback rate limiter
        self._fallback_limiter: Optional[InMemoryRateLimiter] = None
        if self.fallback_to_memory:
            self._fallback_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=check_every_n_seconds,
                max_bucket_size=max(1, requests_per_second * 2),  # Allow some burst
            )
        
        # Lua script SHA hashes (will be populated on first use)
        self._check_and_update_sha: Optional[str] = None
        self._check_only_sha: Optional[str] = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
    
    def _get_redis_key(self) -> str:
        """Generate Redis key for this rate limiter instance."""
        return f"{self.key_prefix}:{self.identifier}"
    
    def _create_redis_client(self) -> Union[coredis.Redis, coredis.RedisCluster]:
        """Create and configure Redis client."""
        common_config = {
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
            "health_check_interval": self.health_check_interval,
            **self.redis_kwargs,
        }
        
        if self.redis_cluster:
            # Parse cluster URLs
            urls = [url.strip() for url in self.redis_url.split(",")]
            startup_nodes = []
            for url in urls:
                if "://" in url:
                    # Parse redis://host:port format
                    parts = url.split("://")[1].split(":")
                    host = parts[0]
                    port = int(parts[1]) if len(parts) > 1 else 6379
                else:
                    # Parse host:port format
                    parts = url.split(":")
                    host = parts[0]
                    port = int(parts[1]) if len(parts) > 1 else 6379
                startup_nodes.append({"host": host, "port": port})
            
            return coredis.RedisCluster(
                startup_nodes=startup_nodes,
                **common_config,
            )
        else:
            return coredis.Redis.from_url(
                self.redis_url,
                max_connections=self.connection_pool_size,
                **common_config,
            )
    
    def _get_redis_client(self) -> Union[coredis.Redis, coredis.RedisCluster]:
        """Get Redis client with connection management and health checking."""
        current_time = time.time()
        
        # Check circuit breaker
        if (self._circuit_breaker_failures >= self._circuit_breaker_threshold and
            current_time - self._circuit_breaker_last_failure < self._circuit_breaker_timeout):
            raise ConnectionError("Circuit breaker is open - Redis unavailable")
        
        with self._connection_lock:
            # Create client if needed
            if self._redis_client is None:
                self._redis_client = self._create_redis_client()
            
            # Periodic health check
            if current_time - self._last_health_check > self.health_check_interval:
                try:
                    # Simple ping to check health
                    asyncio.create_task(self._redis_client.ping())
                    self._redis_healthy = True
                    self._circuit_breaker_failures = 0
                    self._last_health_check = current_time
                except Exception as e:
                    self._logger.warning(f"Redis health check failed: {e}")
                    self._redis_healthy = False
                    self._circuit_breaker_failures += 1
                    self._circuit_breaker_last_failure = current_time
                    
                    # Recreate client on health check failure
                    try:
                        if hasattr(self._redis_client, 'close'):
                            asyncio.create_task(self._redis_client.close())
                    except Exception:
                        pass
                    self._redis_client = None
                    raise ConnectionError(f"Redis health check failed: {e}")
        
        return self._redis_client
    
    async def _ensure_scripts_loaded(self, redis_client: Union[coredis.Redis, coredis.RedisCluster]) -> None:
        """Ensure Lua scripts are loaded into Redis."""
        try:
            if self._check_and_update_sha is None:
                self._check_and_update_sha = await redis_client.script_load(self._CHECK_AND_UPDATE_SCRIPT)
            
            if self._check_only_sha is None:
                self._check_only_sha = await redis_client.script_load(self._CHECK_ONLY_SCRIPT)
        except Exception as e:
            self._logger.warning(f"Failed to load Lua scripts: {e}")
            # Reset SHAs to force reload on next attempt
            self._check_and_update_sha = None
            self._check_only_sha = None
            raise
    
    async def _execute_rate_limit_check(
        self, 
        redis_client: Union[coredis.Redis, coredis.RedisCluster], 
        consume_token: bool = True
    ) -> tuple[bool, int, int]:
        """Execute rate limit check using Lua script.
        
        Returns:
            Tuple of (allowed, current_count, max_requests)
        """
        await self._ensure_scripts_loaded(redis_client)
        
        key = self._get_redis_key()
        current_time = time.time()
        
        try:
            if consume_token:
                result = await redis_client.evalsha(
                    self._check_and_update_sha,
                    1,  # number of keys
                    key,
                    str(self.window_size),
                    str(self.max_requests),
                    str(current_time),
                    str(self.cleanup_probability),
                )
            else:
                result = await redis_client.evalsha(
                    self._check_only_sha,
                    1,  # number of keys
                    key,
                    str(self.window_size),
                    str(self.max_requests),
                    str(current_time),
                )
            
            allowed, current_count, max_requests = result
            return bool(allowed), int(current_count), int(max_requests)
            
        except Exception as e:
            # If script execution fails, try to reload and retry once
            self._logger.warning(f"Lua script execution failed, reloading: {e}")
            self._check_and_update_sha = None
            self._check_only_sha = None
            
            await self._ensure_scripts_loaded(redis_client)
            
            if consume_token:
                result = await redis_client.evalsha(
                    self._check_and_update_sha,
                    1,
                    key,
                    str(self.window_size),
                    str(self.max_requests),
                    str(current_time),
                    str(self.cleanup_probability),
                )
            else:
                result = await redis_client.evalsha(
                    self._check_only_sha,
                    1,
                    key,
                    str(self.window_size),
                    str(self.max_requests),
                    str(current_time),
                )
            
            allowed, current_count, max_requests = result
            return bool(allowed), int(current_count), int(max_requests)
    
    def _should_use_fallback(self, error: Exception) -> bool:
        """Determine if we should use fallback rate limiter based on error type."""
        if not self.fallback_to_memory or self._fallback_limiter is None:
            return False
        
        # Use fallback for connection errors, timeouts, and Redis unavailability
        return isinstance(error, (
            ConnectionError,
            TimeoutError,
            OSError,  # Network errors
        )) or "connection" in str(error).lower()
    
    def acquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter.
        
        Args:
            blocking: If True, block until token is available.
                     If False, return immediately with result.
        
        Returns:
            True if token was acquired, False otherwise.
        """
        # Run async method in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.aacquire(blocking=blocking))
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.aacquire(blocking=blocking))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.aacquire(blocking=blocking))
    
    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Attempt to acquire a token from the rate limiter. Async version.
        
        Args:
            blocking: If True, block until token is available.
                     If False, return immediately with result.
        
        Returns:
            True if token was acquired, False otherwise.
        """
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                redis_client = self._get_redis_client()
                
                if not blocking:
                    # Non-blocking: just check availability
                    allowed, current_count, max_requests = await self._execute_rate_limit_check(
                        redis_client, consume_token=False
                    )
                    if allowed:
                        # If available, consume the token
                        allowed, _, _ = await self._execute_rate_limit_check(
                            redis_client, consume_token=True
                        )
                    return allowed
                else:
                    # Blocking: keep trying until we get a token
                    while True:
                        allowed, current_count, max_requests = await self._execute_rate_limit_check(
                            redis_client, consume_token=True
                        )
                        if allowed:
                            return True
                        
                        # Wait before retrying
                        await asyncio.sleep(self.check_every_n_seconds)
            
            except Exception as e:
                self._logger.warning(f"Redis rate limiter error (attempt {attempt + 1}): {e}")
                
                # Update circuit breaker
                self._circuit_breaker_failures += 1
                self._circuit_breaker_last_failure = time.time()
                
                if self._should_use_fallback(e):
                    self._logger.info("Falling back to in-memory rate limiter")
                    return await self._fallback_limiter.aacquire(blocking=blocking)
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Final attempt failed
                    if self.fallback_to_memory and self._fallback_limiter:
                        self._logger.error(f"All Redis attempts failed, using fallback: {e}")
                        return await self._fallback_limiter.aacquire(blocking=blocking)
                    else:
                        raise ConnectionError(f"Redis rate limiter unavailable: {e}")
        
        # Should not reach here, but just in case
        if self.fallback_to_memory and self._fallback_limiter:
            return await self._fallback_limiter.aacquire(blocking=blocking)
        return False
    
    async def get_current_usage(self) -> Dict[str, Union[int, float]]:
        """Get current rate limiter usage statistics.
        
        Returns:
            Dictionary with usage statistics including current count,
            max requests, and utilization percentage.
        """
        try:
            redis_client = self._get_redis_client()
            allowed, current_count, max_requests = await self._execute_rate_limit_check(
                redis_client, consume_token=False
            )
            
            utilization = (current_count / max_requests) * 100 if max_requests > 0 else 0
            
            return {
                "current_count": current_count,
                "max_requests": max_requests,
                "utilization_percent": utilization,
                "requests_per_second": self.requests_per_second,
                "window_size": self.window_size,
                "redis_healthy": self._redis_healthy,
            }
        except Exception as e:
            self._logger.warning(f"Failed to get usage statistics: {e}")
            return {
                "current_count": -1,
                "max_requests": self.max_requests,
                "utilization_percent": -1,
                "requests_per_second": self.requests_per_second,
                "window_size": self.window_size,
                "redis_healthy": False,
                "error": str(e),
            }
    
    async def reset(self) -> bool:
        """Reset the rate limiter by clearing all stored data.
        
        Returns:
            True if reset was successful, False otherwise.
        """
        try:
            redis_client = self._get_redis_client()
            key = self._get_redis_key()
            await redis_client.delete(key)
            self._logger.info(f"Rate limiter reset for key: {key}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to reset rate limiter: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connections and cleanup resources."""
        if self._redis_client:
            try:
                if hasattr(self._redis_client, 'close'):
                    await self._redis_client.close()
            except Exception as e:
                self._logger.warning(f"Error closing Redis client: {e}")
            finally:
                self._redis_client = None
    
    def __del__(self) -> None:
        """Cleanup on object destruction."""
        if self._redis_client:
            try:
                asyncio.create_task(self.close())
            except Exception:
                pass


__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
    "RedisRateLimiter",
]

