"""Base abstraction and in-memory implementation of rate limiters.

These rate limiters can be used to limit the rate of requests to an API.

The rate limiters can be used together with `BaseChatModel`.
"""

from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
]
