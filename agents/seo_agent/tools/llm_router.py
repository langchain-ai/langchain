"""LLM router — central model selection, cost tracking, and output caching.

Every node in the graph must call this module rather than instantiating a model
directly. This is the primary cost-control mechanism.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import anthropic

from agents.seo_agent.config import MAX_WEEKLY_SPEND_USD, TOKEN_BUDGETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task → model mapping
# ---------------------------------------------------------------------------

TASK_MODEL_MAP: dict[str, str] = {
    # Haiku — classification, extraction, filtering (cheapest)
    "classify_prospect": "claude-haiku-4-5-20251001",
    "score_prospect": "claude-haiku-4-5-20251001",
    "extract_contact_email": "claude-haiku-4-5-20251001",
    "detect_page_type": "claude-haiku-4-5-20251001",
    "summarise_page": "claude-haiku-4-5-20251001",
    "check_reply_intent": "claude-haiku-4-5-20251001",
    "filter_keywords": "claude-haiku-4-5-20251001",
    # Sonnet — drafting, analysis, briefs (mid-cost, default)
    "write_content_brief": "claude-sonnet-4-6",
    "write_blog_post": "claude-sonnet-4-6",
    "write_location_page": "claude-sonnet-4-6",
    "write_tier2_email": "claude-sonnet-4-6",
    "analyse_content_gap": "claude-sonnet-4-6",
    "generate_pr_angles": "claude-sonnet-4-6",
    "write_followup_email": "claude-sonnet-4-6",
    # Opus — highest-stakes output only (most expensive, use sparingly)
    "write_tier1_email": "claude-opus-4-6",
    "write_digital_pr_pitch": "claude-opus-4-6",
}

MODEL_COSTS_PER_1M: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}

# Default freshness thresholds for output caching (days)
CACHE_FRESHNESS_DAYS: dict[str, int] = {
    "write_content_brief": 7,
    "write_blog_post": 14,
    "write_location_page": 14,
    "generate_pr_angles": 30,
    "analyse_content_gap": 7,
    "summarise_page": 30,
}


def get_model(task: str, budget_remaining: float = 999.0) -> str:
    """Return the appropriate model for a task, downgrading if budget is low.

    Args:
        task: The task identifier (must match a key in TASK_MODEL_MAP).
        budget_remaining: Fraction of weekly budget remaining (0.0–1.0).

    Returns:
        A Claude model identifier string.
    """
    model = TASK_MODEL_MAP.get(task, "claude-sonnet-4-6")
    # If over 80% of weekly budget spent, downgrade Sonnet tasks to Haiku
    if budget_remaining < 0.20 and model == "claude-sonnet-4-6":
        return "claude-haiku-4-5-20251001"
    # If over 80% of budget, skip Opus entirely — return Sonnet instead
    if budget_remaining < 0.20 and model == "claude-opus-4-6":
        return "claude-sonnet-4-6"
    return model


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Calculate USD cost of an API call including cached token discount.

    Args:
        model: The Claude model identifier.
        input_tokens: Total input tokens (including cached).
        output_tokens: Total output tokens generated.
        cached_tokens: Number of input tokens served from cache.

    Returns:
        Cost in USD, rounded to 6 decimal places.
    """
    rates = MODEL_COSTS_PER_1M.get(model, MODEL_COSTS_PER_1M["claude-sonnet-4-6"])
    uncached_input = input_tokens - cached_tokens
    cost = uncached_input / 1_000_000 * rates["input"]
    cost += cached_tokens / 1_000_000 * rates["input"] * 0.10  # 10% for cached
    cost += output_tokens / 1_000_000 * rates["output"]
    return round(cost, 6)


def get_budget_remaining(weekly_spend: float) -> float:
    """Return the fraction of weekly budget remaining.

    Args:
        weekly_spend: Total USD spent this week so far.

    Returns:
        Fraction remaining (0.0–1.0). Values below 0.20 trigger downgrades.
    """
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", str(MAX_WEEKLY_SPEND_USD)))
    if cap <= 0:
        return 1.0
    return max(0.0, 1.0 - (weekly_spend / cap))


# ---------------------------------------------------------------------------
# Anthropic SDK helper — every LLM call goes through here
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Return a singleton Anthropic client.

    Supports OpenRouter as a proxy: set ``OPENROUTER_API_KEY`` and optionally
    ``OPENROUTER_BASE_URL`` in the environment. Falls back to the standard
    ``ANTHROPIC_API_KEY`` when no OpenRouter key is present.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            base_url = os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            _client = anthropic.Anthropic(
                api_key=openrouter_key,
                base_url=base_url,
            )
        else:
            _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def call_llm(
    task: str,
    messages: list[dict[str, str]],
    *,
    system: str | list[dict] = "",
    weekly_spend: float = 0.0,
    site: str = "",
    log_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Call the Claude API via the Anthropic SDK with cost tracking.

    Uses prompt caching when `system` is passed as a list of content blocks
    (include ``"cache_control": {"type": "ephemeral"}`` on static blocks).

    Args:
        task: Task identifier for model routing and token budgets.
        messages: Conversation messages in Anthropic format.
        system: System prompt — string or list of content blocks for caching.
        weekly_spend: Current week's LLM spend in USD.
        site: Target site name (for cost logging).
        log_fn: Optional callable to log cost rows to Supabase.

    Returns:
        Dict with keys: ``text``, ``model``, ``input_tokens``,
        ``output_tokens``, ``cached_tokens``, ``cost_usd``.
    """
    budget_remaining = get_budget_remaining(weekly_spend)
    model = get_model(task, budget_remaining)
    max_tokens = TOKEN_BUDGETS.get(task, 1024)

    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    # Extract token usage
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)

    # Log cost if a logging function is provided
    if log_fn is not None:
        try:
            log_fn(
                task_type=task,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
                site=site,
            )
        except Exception:
            logger.warning("Failed to log LLM cost", exc_info=True)

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    return {
        "text": text,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": cost,
    }


# ---------------------------------------------------------------------------
# Output caching — check Supabase before calling LLM
# ---------------------------------------------------------------------------


def _cache_key(task: str, key: str) -> str:
    """Generate a deterministic cache key from task and input key."""
    raw = f"{task}:{key}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_or_generate(
    task: str,
    key: str,
    generator_fn: Callable[[], dict[str, Any]],
    supabase_client: Any | None = None,
    freshness_days: int | None = None,
) -> dict[str, Any]:
    """Return a cached result or generate a new one.

    Checks the ``llm_output_cache`` Supabase table for a recent result before
    calling the generator function. Results are stored after generation.

    Args:
        task: Task identifier.
        key: A string uniquely identifying the input (e.g. keyword slug).
        generator_fn: Callable that produces the result dict when cache misses.
        supabase_client: Optional Supabase client for cache reads/writes.
        freshness_days: Max age in days for cached results. Defaults per task.

    Returns:
        The result dict (either cached or freshly generated).
    """
    if freshness_days is None:
        freshness_days = CACHE_FRESHNESS_DAYS.get(task, 7)

    cache_id = _cache_key(task, key)

    # Try cache lookup
    if supabase_client is not None:
        try:
            cutoff = datetime.now(tz=timezone.utc).timestamp() - (
                freshness_days * 86400
            )
            resp = (
                supabase_client.table("llm_output_cache")
                .select("result, created_at")
                .eq("cache_key", cache_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if resp.data:
                row = resp.data[0]
                created = datetime.fromisoformat(row["created_at"]).timestamp()
                if created >= cutoff:
                    logger.info("Cache hit for task=%s key=%s", task, key)
                    return json.loads(row["result"])
        except Exception:
            logger.warning("Cache lookup failed", exc_info=True)

    # Cache miss — generate
    result = generator_fn()

    # Store in cache
    if supabase_client is not None:
        try:
            supabase_client.table("llm_output_cache").insert(
                {
                    "cache_key": cache_id,
                    "task": task,
                    "input_key": key,
                    "result": json.dumps(result),
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            ).execute()
        except Exception:
            logger.warning("Cache store failed", exc_info=True)

    return result
