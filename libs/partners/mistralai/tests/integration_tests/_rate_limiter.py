"""Shared rate limiter for Mistral integration tests.

Scaled by `PYTEST_XDIST_WORKER_COUNT` so aggregate QPS across all xdist
workers stays bounded near the target rate.
"""

from __future__ import annotations

import os

from langchain_core.rate_limiters import InMemoryRateLimiter

_TARGET_REQUESTS_PER_SECOND = 0.5
_WORKER_COUNT = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))

rate_limiter = InMemoryRateLimiter(
    requests_per_second=_TARGET_REQUESTS_PER_SECOND / _WORKER_COUNT,
)
