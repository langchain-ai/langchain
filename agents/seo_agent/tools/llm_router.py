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

from agents.seo_agent.config import MAX_WEEKLY_SPEND_USD, TOKEN_BUDGETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task → model mapping
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal model tier names (provider-agnostic)
# ---------------------------------------------------------------------------
_HAIKU = "haiku"
_SONNET = "sonnet"
_OPUS = "opus"

TASK_MODEL_MAP: dict[str, str] = {
    # Haiku — classification, extraction, filtering (cheapest)
    "classify_prospect": _HAIKU,
    "score_prospect": _HAIKU,
    "extract_contact_email": _HAIKU,
    "detect_page_type": _HAIKU,
    "summarise_page": _HAIKU,
    "check_reply_intent": _HAIKU,
    "filter_keywords": _HAIKU,
    "review_blog_post": _HAIKU,
    # Sonnet — drafting, analysis, briefs (mid-cost, default)
    "write_content_brief": _SONNET,
    "write_blog_post": _SONNET,
    "write_location_page": _SONNET,
    "write_tier2_email": _SONNET,
    "analyse_content_gap": _SONNET,
    "generate_pr_angles": _SONNET,
    "write_followup_email": _SONNET,
    # Opus — highest-stakes output only (most expensive, use sparingly)
    "write_tier1_email": _OPUS,
    "write_digital_pr_pitch": _OPUS,
}

# Anthropic native model IDs (used when ANTHROPIC_API_KEY is set)
_ANTHROPIC_MODELS: dict[str, str] = {
    _HAIKU: "claude-haiku-4-5-20251001",
    _SONNET: "claude-sonnet-4-6",
    _OPUS: "claude-opus-4-6",
}

# OpenRouter model IDs (used when OPENROUTER_API_KEY is set)
_OPENROUTER_MODELS: dict[str, str] = {
    _HAIKU: "anthropic/claude-haiku-4.5",
    _SONNET: "anthropic/claude-sonnet-4.6",
    _OPUS: "anthropic/claude-opus-4.6",
}

MODEL_COSTS_PER_1M: dict[str, dict[str, float]] = {
    _HAIKU: {"input": 1.00, "output": 5.00},
    _SONNET: {"input": 3.00, "output": 15.00},
    _OPUS: {"input": 5.00, "output": 25.00},
}


def _use_openrouter() -> bool:
    """Return True if OpenRouter is configured as the LLM provider."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _resolve_model_id(tier: str) -> str:
    """Map an internal tier name to the provider-specific model ID."""
    if _use_openrouter():
        return _OPENROUTER_MODELS.get(tier, _OPENROUTER_MODELS[_SONNET])
    return _ANTHROPIC_MODELS.get(tier, _ANTHROPIC_MODELS[_SONNET])

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
    """Return the appropriate model tier for a task, downgrading if budget is low.

    Args:
        task: The task identifier (must match a key in TASK_MODEL_MAP).
        budget_remaining: Fraction of weekly budget remaining (0.0–1.0).

    Returns:
        An internal model tier string (haiku/sonnet/opus).
    """
    tier = TASK_MODEL_MAP.get(task, _SONNET)
    # If over 80% of weekly budget spent, downgrade Sonnet tasks to Haiku
    if budget_remaining < 0.20 and tier == _SONNET:
        return _HAIKU
    # If over 80% of budget, skip Opus entirely — return Sonnet instead
    if budget_remaining < 0.20 and tier == _OPUS:
        return _SONNET
    return tier


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
    rates = MODEL_COSTS_PER_1M.get(model, MODEL_COSTS_PER_1M[_SONNET])
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
# LLM client — supports both Anthropic SDK and OpenRouter (OpenAI-compat)
# ---------------------------------------------------------------------------

_anthropic_client: Any = None
_openai_client: Any = None


def _get_anthropic_client() -> Any:
    """Return a singleton Anthropic client (direct Anthropic API)."""
    global _anthropic_client  # noqa: PLW0603
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _anthropic_client


def _get_openrouter_client() -> Any:
    """Return a singleton OpenAI-compatible client pointed at OpenRouter."""
    global _openai_client  # noqa: PLW0603
    if _openai_client is None:
        from openai import OpenAI
        api_key = "".join(os.environ.get("OPENROUTER_API_KEY", "").split())
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        logger.info("OpenRouter client init: key_len=%d prefix=%s", len(api_key), api_key[:5])
        _openai_client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            default_headers={
                "HTTP-Referer": "https://kitchensdirectory.co.uk",
                "X-Title": "SEO Agent",
            },
        )
    return _openai_client


def _call_openrouter(
    model_id: str,
    messages: list[dict[str, str]],
    system: str | list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    """Call OpenRouter via the OpenAI-compatible chat completions API."""
    client = _get_openrouter_client()

    # Prepend system prompt as a system message
    oai_messages: list[dict[str, str]] = []
    if system:
        sys_text = system if isinstance(system, str) else " ".join(
            block.get("text", "") for block in system if isinstance(block, dict)
        )
        if sys_text.strip():
            oai_messages.append({"role": "system", "content": sys_text})
    oai_messages.extend(messages)

    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=oai_messages,
    )

    choice = response.choices[0]
    usage = response.usage

    return {
        "text": choice.message.content or "",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "cached_tokens": 0,
    }


def _call_anthropic(
    model_id: str,
    messages: list[dict[str, str]],
    system: str | list[dict],
    max_tokens: int,
) -> dict[str, Any]:
    """Call the Anthropic messages API directly."""
    client = _get_anthropic_client()
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    usage = response.usage
    text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return {
        "text": text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def call_llm(
    task: str,
    messages: list[dict[str, str]],
    *,
    system: str | list[dict] = "",
    weekly_spend: float = 0.0,
    site: str = "",
    log_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Call Claude via Anthropic API or OpenRouter with cost tracking.

    Automatically selects the correct provider based on environment variables:
    - ``OPENROUTER_API_KEY`` set → uses OpenRouter (OpenAI-compatible API)
    - Otherwise → uses Anthropic SDK (reads ``ANTHROPIC_API_KEY``)

    Args:
        task: Task identifier for model routing and token budgets.
        messages: Conversation messages (``{"role": ..., "content": ...}``).
        system: System prompt — string or list of content blocks for caching.
        weekly_spend: Current week's LLM spend in USD.
        site: Target site name (for cost logging).
        log_fn: Optional callable to log cost rows to Supabase.

    Returns:
        Dict with keys: ``text``, ``model``, ``input_tokens``,
        ``output_tokens``, ``cached_tokens``, ``cost_usd``.
    """
    budget_remaining = get_budget_remaining(weekly_spend)
    tier = get_model(task, budget_remaining)
    model_id = _resolve_model_id(tier)
    max_tokens = TOKEN_BUDGETS.get(task, 1024)

    if _use_openrouter():
        result = _call_openrouter(model_id, messages, system, max_tokens)
    else:
        result = _call_anthropic(model_id, messages, system, max_tokens)

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    cached_tokens = result["cached_tokens"]

    cost = calculate_cost(tier, input_tokens, output_tokens, cached_tokens)

    # Log cost — use explicit log_fn if provided, otherwise auto-log
    if log_fn is not None:
        try:
            log_fn(
                task_type=task,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
                site=site,
            )
        except Exception:
            logger.warning("Failed to log LLM cost", exc_info=True)
    else:
        try:
            from agents.seo_agent.tools.supabase_tools import log_llm_cost
            log_llm_cost(
                task_type=task,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
                site=site,
            )
        except Exception:
            logger.debug("Auto cost logging failed", exc_info=True)

    return {
        "text": result["text"],
        "model": model_id,
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
