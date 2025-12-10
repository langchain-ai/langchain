"""Cost tracking utilities for Perplexity API calls.

This module provides comprehensive cost tracking capabilities including:
- Real-time cost aggregation via callbacks
- Budget management with warnings and limits
- Detailed cost breakdowns
- Pre-call cost estimation
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from langchain_perplexity.data._pricing import get_model_pricing

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed breakdown of costs for a single API call."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    citation_cost: float = 0.0
    reasoning_cost: float = 0.0
    search_cost: float = 0.0

    @property
    def total(self) -> float:
        """Total cost of all components."""
        return (
            self.input_cost
            + self.output_cost
            + self.citation_cost
            + self.reasoning_cost
            + self.search_cost
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "citation_cost": self.citation_cost,
            "reasoning_cost": self.reasoning_cost,
            "search_cost": self.search_cost,
            "total_cost": self.total,
        }


@dataclass
class UsageRecord:
    """Record of a single API call's usage and cost."""

    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    citation_tokens: int = 0
    search_queries: int = 0
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        """Total cost of this call."""
        return self.cost_breakdown.total


@dataclass
class CostSummary:
    """Aggregated cost summary across multiple API calls."""

    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_citation_tokens: int = 0
    total_search_queries: int = 0
    call_count: int = 0
    cost_by_model: dict[str, float] = field(default_factory=dict)
    records: list[UsageRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all calls."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def average_cost_per_call(self) -> float:
        """Average cost per API call."""
        return self.total_cost / self.call_count if self.call_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_citation_tokens": self.total_citation_tokens,
            "total_search_queries": self.total_search_queries,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "average_cost_per_call": self.average_cost_per_call,
            "cost_by_model": self.cost_by_model,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== Perplexity Cost Summary ===",
            f"Total Cost: ${self.total_cost:.6f}",
            f"API Calls: {self.call_count}",
            f"Avg Cost/Call: ${self.average_cost_per_call:.6f}",
            "",
            "Token Usage:",
            f"  Input: {self.total_input_tokens:,}",
            f"  Output: {self.total_output_tokens:,}",
        ]
        if self.total_reasoning_tokens:
            lines.append(f"  Reasoning: {self.total_reasoning_tokens:,}")
        if self.total_citation_tokens:
            lines.append(f"  Citation: {self.total_citation_tokens:,}")
        if self.total_search_queries:
            lines.append(f"  Search Queries: {self.total_search_queries}")

        if self.cost_by_model:
            lines.append("")
            lines.append("Cost by Model:")
            for model, cost in sorted(self.cost_by_model.items(), key=lambda x: -x[1]):
                lines.append(f"  {model}: ${cost:.6f}")

        return "\n".join(lines)


class BudgetExceededError(Exception):
    """Raised when the cost budget has been exceeded."""

    def __init__(self, budget: float, current_cost: float, attempted_cost: float):
        self.budget = budget
        self.current_cost = current_cost
        self.attempted_cost = attempted_cost
        super().__init__(
            f"Budget exceeded: limit=${budget:.4f}, "
            f"current=${current_cost:.4f}, attempted=${attempted_cost:.4f}"
        )


class PerplexityCostTracker(BaseCallbackHandler):
    """Callback handler for tracking Perplexity API costs.

    This handler integrates with LangChain's callback system to automatically
    track costs across all Perplexity API calls in a session.

    Example:
        ```python
        from langchain_perplexity import ChatPerplexity, PerplexityCostTracker

        # Create tracker with optional budget
        tracker = PerplexityCostTracker(budget=1.00, warn_at=0.80)

        # Use with model
        model = ChatPerplexity(model="sonar", callbacks=[tracker])
        response = model.invoke("Hello!")

        # Check costs
        print(tracker.summary)
        print(f"Total cost: ${tracker.total_cost:.4f}")
        ```

    Args:
        budget: Maximum allowed cost in USD. If exceeded, raises BudgetExceededError.
        warn_at: Cost threshold (as fraction of budget) to trigger warnings.
        on_budget_warning: Callback function when warn_at threshold is reached.
        on_cost_update: Callback function called after each API call with the record.
        store_records: Whether to store individual usage records (default True).
    """

    def __init__(
        self,
        budget: float | None = None,
        warn_at: float = 0.8,
        on_budget_warning: Any | None = None,
        on_cost_update: Any | None = None,
        store_records: bool = True,
    ):
        super().__init__()
        self.budget = budget
        self.warn_at = warn_at
        self.on_budget_warning = on_budget_warning
        self.on_cost_update = on_cost_update
        self.store_records = store_records

        self._summary = CostSummary()
        self._lock = threading.Lock()
        self._warning_triggered = False

    @property
    def total_cost(self) -> float:
        """Total cost accumulated so far."""
        return self._summary.total_cost

    @property
    def summary(self) -> CostSummary:
        """Get the current cost summary."""
        return self._summary

    @property
    def remaining_budget(self) -> float | None:
        """Remaining budget, or None if no budget set."""
        if self.budget is None:
            return None
        return max(0, self.budget - self.total_cost)

    def reset(self) -> CostSummary:
        """Reset the tracker and return the final summary.

        Returns:
            The cost summary before reset.
        """
        with self._lock:
            final_summary = self._summary
            self._summary = CostSummary()
            self._warning_triggered = False
            return final_summary

    def _check_budget(self, estimated_cost: float = 0.0) -> None:
        """Check if budget would be exceeded."""
        if self.budget is None:
            return

        projected = self.total_cost + estimated_cost

        # Check warning threshold
        if not self._warning_triggered and projected >= self.budget * self.warn_at:
            self._warning_triggered = True
            msg = (
                f"Perplexity cost warning: ${projected:.4f} of ${self.budget:.4f} "
                f"budget used ({projected / self.budget * 100:.1f}%)"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            logger.warning(msg)
            if self.on_budget_warning:
                self.on_budget_warning(projected, self.budget)

        # Check hard limit
        if projected > self.budget:
            raise BudgetExceededError(self.budget, self.total_cost, estimated_cost)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Process LLM response and update cost tracking."""
        for generation_list in response.generations:
            for generation in generation_list:
                message = generation.message if hasattr(generation, "message") else None
                if message is None:
                    continue

                usage = getattr(message, "usage_metadata", None)
                response_meta = getattr(message, "response_metadata", {}) or {}

                if usage is None:
                    continue

                model_name = response_meta.get("model_name", "unknown")
                cost_breakdown = response_meta.get("cost_breakdown")

                # Build usage record
                output_details = usage.get("output_token_details", {}) or {}
                record = UsageRecord(
                    model=model_name,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    reasoning_tokens=output_details.get("reasoning", 0) or 0,
                    citation_tokens=output_details.get("citation_tokens", 0) or 0,
                    search_queries=response_meta.get("num_search_queries", 0),
                    run_id=kwargs.get("run_id"),
                )

                # Set cost breakdown
                if cost_breakdown:
                    record.cost_breakdown = CostBreakdown(**cost_breakdown)
                elif "cost" in response_meta:
                    # Fallback to simple total cost
                    record.cost_breakdown = CostBreakdown(
                        output_cost=response_meta["cost"]
                    )

                self._record_usage(record)

    def _record_usage(self, record: UsageRecord) -> None:
        """Record usage and update summary."""
        with self._lock:
            # Check budget before recording
            self._check_budget(record.total_cost)

            # Update summary
            self._summary.total_cost += record.total_cost
            self._summary.total_input_tokens += record.input_tokens
            self._summary.total_output_tokens += record.output_tokens
            self._summary.total_reasoning_tokens += record.reasoning_tokens
            self._summary.total_citation_tokens += record.citation_tokens
            self._summary.total_search_queries += record.search_queries
            self._summary.call_count += 1

            # Update per-model costs
            if record.model not in self._summary.cost_by_model:
                self._summary.cost_by_model[record.model] = 0.0
            self._summary.cost_by_model[record.model] += record.total_cost

            # Store record if enabled
            if self.store_records:
                self._summary.records.append(record)

            # Trigger callback
            if self.on_cost_update:
                self.on_cost_update(record, self._summary)


def calculate_cost_breakdown(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    citation_tokens: int = 0,
    num_search_queries: int = 0,
) -> CostBreakdown:
    """Calculate detailed cost breakdown for an API call.

    Args:
        model_name: The Perplexity model name.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        reasoning_tokens: Number of reasoning tokens (for reasoning models).
        citation_tokens: Number of citation tokens (for deep research).
        num_search_queries: Number of search queries (for deep research).

    Returns:
        CostBreakdown with itemized costs.
    """
    pricing = get_model_pricing(model_name)
    if pricing is None:
        return CostBreakdown()

    breakdown = CostBreakdown()

    # Base costs
    breakdown.input_cost = (input_tokens / 1_000_000) * pricing[
        "input_cost_per_million"
    ]
    breakdown.output_cost = (output_tokens / 1_000_000) * pricing[
        "output_cost_per_million"
    ]

    # Additional costs for deep research
    if pricing["citation_cost_per_million"] and citation_tokens > 0:
        breakdown.citation_cost = (citation_tokens / 1_000_000) * pricing[
            "citation_cost_per_million"
        ]

    if pricing["reasoning_cost_per_million"] and reasoning_tokens > 0:
        breakdown.reasoning_cost = (reasoning_tokens / 1_000_000) * pricing[
            "reasoning_cost_per_million"
        ]

    if pricing["search_cost_per_thousand"] and num_search_queries > 0:
        breakdown.search_cost = (num_search_queries / 1_000) * pricing[
            "search_cost_per_thousand"
        ]

    return breakdown


def estimate_cost(
    model_name: str,
    messages: Sequence[BaseMessage] | str,
    estimated_output_tokens: int = 500,
) -> float | None:
    """Estimate the cost of an API call before making it.

    This provides a rough estimate based on input length and expected output.
    Actual costs may vary based on actual token counts.

    Args:
        model_name: The Perplexity model to use.
        messages: Input messages or a string prompt.
        estimated_output_tokens: Expected number of output tokens (default 500).

    Returns:
        Estimated cost in USD, or None if model pricing unavailable.

    Example:
        ```python
        from langchain_perplexity import estimate_cost

        # Estimate before calling
        estimated = estimate_cost("sonar-pro", "What is quantum computing?")
        print(f"Estimated cost: ${estimated:.4f}")
        ```
    """
    pricing = get_model_pricing(model_name)
    if pricing is None:
        return None

    # Estimate input tokens (rough: ~4 chars per token)
    if isinstance(messages, str):
        input_chars = len(messages)
    else:
        input_chars = sum(
            len(str(msg.content)) for msg in messages if hasattr(msg, "content")
        )

    estimated_input_tokens = max(1, input_chars // 4)

    breakdown = calculate_cost_breakdown(
        model_name,
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
    )

    return breakdown.total


def format_cost(cost: float, precision: int = 4) -> str:
    """Format a cost value as a currency string.

    Args:
        cost: Cost in USD.
        precision: Decimal places to show.

    Returns:
        Formatted string like "$0.0015".
    """
    return f"${cost:.{precision}f}"


# Convenience type aliases
CostCallback = Any  # Callable[[UsageRecord, CostSummary], None]
BudgetCallback = Any  # Callable[[float, float], None]
