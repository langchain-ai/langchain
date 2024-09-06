import time
from typing import Optional as Optional

from langchain_core.caches import InMemoryCache
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter


async def test_rate_limit_ainvoke() -> None:
    """Add rate limiter."""
    tic = time.time()

    model = GenericFakeChatModel(
        messages=iter(["hello", "world", "!"]),
    )

    for _ in range(1_000):
        await model.ainvoke("foo")
    toc = time.time()

    # Verify that the time taken to run the loop is less than 0.5 seconds

    assert (toc - tic) < 0.5
