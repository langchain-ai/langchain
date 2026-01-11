# SLO-aware routing middleware

This guide shows a minimal, IP-safe pattern for routing requests based on
service-level objectives (SLOs). It focuses on deadline enforcement, tail
latency risk, and a simple Bayesian miss-probability estimate. The example
uses only the Python standard library.

## Why SLO-aware routing

SLO routing gates traffic when a downstream model or service is likely to miss
latency or availability targets. If you route purely on averages, you miss the
impact of tail latency (p95 and beyond). A healthy mean can still hide a large
tail risk, and the tail is what breaks user-visible SLOs.

## Deadline enforcement

Assume each request has a deadline in milliseconds. The middleware should:

1. Measure the end-to-end latency.
2. Record whether the deadline was met.
3. Route future traffic to alternatives when miss risk grows.

## Bayesian miss-probability estimate

Model deadline misses as a Bernoulli event. With a Beta prior and Binomial
observations, the posterior mean is:

posterior_mean = (alpha + misses) / (alpha + beta + total)

This estimate is stable for small samples and adapts as data arrives. The
example below uses a uniform prior (alpha=1, beta=1).

## Deterministic example

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.utils.math import beta_binomial_posterior_mean


@dataclass
class RouteStats:
    deadline_ms: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    misses: int = 0
    hits: int = 0

    def record_latency(self, latency_ms: int) -> None:
        if latency_ms > self.deadline_ms:
            self.misses += 1
        else:
            self.hits += 1

    def miss_probability(self) -> float:
        return beta_binomial_posterior_mean(
            self.misses,
            self.hits,
            alpha=self.prior_alpha,
            beta=self.prior_beta,
        )


def route_choice(
    stats: RouteStats,
    *,
    miss_threshold: float,
    primary: str,
    fallback: str,
) -> str:
    return fallback if stats.miss_probability() > miss_threshold else primary


def replay_latencies(stats: RouteStats, latencies_ms: Iterable[int]) -> None:
    for latency_ms in latencies_ms:
        stats.record_latency(latency_ms)


def run_demo(
    latencies_ms: Iterable[int],
    *,
    deadline_ms: int,
    miss_threshold: float,
    primary: str = "primary",
    fallback: str = "fallback",
) -> str:
    stats = RouteStats(deadline_ms=deadline_ms)
    replay_latencies(stats, latencies_ms)
    return route_choice(
        stats,
        miss_threshold=miss_threshold,
        primary=primary,
        fallback=fallback,
    )


if __name__ == "__main__":
    latencies = [120, 95, 110, 140, 105]
    decision = run_demo(
        latencies,
        deadline_ms=100,
        miss_threshold=0.4,
        primary="model-a",
        fallback="model-b",
    )
    print(decision)
```

### What to notice

- deadline_ms sets the hard cutoff for a miss.
- Tail latency matters: one or two slow calls can raise the miss probability
  above the threshold even if the average stays low.
- The posterior mean is deterministic given the observed misses and hits.
