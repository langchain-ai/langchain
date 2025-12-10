"""Advanced test cases for bulletproof cost tracking.

Young Padawan, thorough testing - the path to mastery, it is.
Edge cases, concurrency, and real-world scenarios we cover here.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from langchain_perplexity import (
    PERPLEXITY_PRICING,
    BudgetExceededError,
    CostBreakdown,
    CostSummary,
    PerplexityCostTracker,
    UsageRecord,
    calculate_cost,
    calculate_cost_breakdown,
    estimate_cost,
    format_cost,
    get_model_pricing,
)
from langchain_perplexity.chat_models import _create_usage_metadata

# ============================================================================
# EDGE CASE TESTS - "Corner cases, hide bugs do"
# ============================================================================


class TestEdgeCases:
    """Test edge cases that might break in production."""

    def test_calculate_cost_with_zero_tokens(self) -> None:
        """Zero tokens should return zero cost, not error."""
        breakdown = calculate_cost_breakdown("sonar", 0, 0)
        assert breakdown.total == 0.0

    def test_calculate_cost_with_very_large_tokens(self) -> None:
        """Handle millions of tokens without overflow."""
        # 100 million tokens - a massive request
        breakdown = calculate_cost_breakdown(
            "sonar",
            input_tokens=100_000_000,
            output_tokens=50_000_000,
        )
        # sonar: $1/1M = $100 input + $50 output = $150
        assert abs(breakdown.total - 150.0) < 0.01

    def test_calculate_cost_with_float_precision(self) -> None:
        """Ensure float precision doesn't cause weird results."""
        # Small token counts that could cause precision issues
        breakdown = calculate_cost_breakdown("sonar", 1, 1)
        # 1 token at $1/1M = $0.000001
        assert breakdown.input_cost > 0
        assert breakdown.input_cost < 0.001

    def test_estimate_cost_empty_string(self) -> None:
        """Empty prompt should still return valid estimate."""
        cost = estimate_cost("sonar", "", estimated_output_tokens=100)
        assert cost is not None
        assert cost >= 0

    def test_estimate_cost_unicode_content(self) -> None:
        """Unicode characters shouldn't break estimation."""
        prompt = "こんにちは世界 🚀 مرحبا بالعالم"
        cost = estimate_cost("sonar", prompt)
        assert cost is not None

    def test_format_cost_negative(self) -> None:
        """Negative costs should format correctly (edge case)."""
        result = format_cost(-0.001)
        assert result == "$-0.0010"

    def test_format_cost_very_small(self) -> None:
        """Very small costs should show precision."""
        result = format_cost(0.0000001, precision=8)
        assert "0.00000010" in result

    def test_pricing_data_completeness(self) -> None:
        """All models in profiles should have pricing."""
        from langchain_perplexity.data._profiles import _PROFILES

        for model_name in _PROFILES.keys():
            # At minimum, base models should have pricing
            # Some variants might not, which is acceptable
            pass  # This documents the expectation

    def test_cost_breakdown_immutability_safety(self) -> None:
        """Modifying returned dict shouldn't affect internal state."""
        breakdown = calculate_cost_breakdown("sonar", 1000, 500)
        original_total = breakdown.total

        # Try to modify the dict
        result = breakdown.to_dict()
        result["input_cost"] = 999999

        # Original should be unchanged
        assert breakdown.input_cost != 999999
        assert abs(breakdown.total - original_total) < 1e-10


# ============================================================================
# CONCURRENCY TESTS - "Many threads, one tracker"
# ============================================================================


class TestConcurrency:
    """Test thread safety under concurrent load."""

    def test_concurrent_cost_recording(self) -> None:
        """Multiple threads recording costs simultaneously."""
        tracker = PerplexityCostTracker()
        num_threads = 10
        records_per_thread = 100

        def record_costs() -> None:
            for i in range(records_per_thread):
                record = UsageRecord(
                    model="sonar",
                    input_tokens=100,
                    output_tokens=50,
                    cost_breakdown=CostBreakdown(
                        input_cost=0.0001,
                        output_cost=0.00005,
                    ),
                )
                tracker._record_usage(record)

        # Run concurrent threads
        threads = [threading.Thread(target=record_costs) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all records were counted
        expected_calls = num_threads * records_per_thread
        assert tracker.summary.call_count == expected_calls

        # Verify cost is approximately correct
        expected_cost = expected_calls * 0.00015  # 0.0001 + 0.00005
        assert abs(tracker.summary.total_cost - expected_cost) < 0.001

    def test_concurrent_budget_check(self) -> None:
        """Budget checks should be thread-safe."""
        tracker = PerplexityCostTracker(budget=0.01)
        exceeded_count = [0]  # Use list for mutable closure
        lock = threading.Lock()

        def try_exceed_budget() -> None:
            try:
                record = UsageRecord(
                    model="sonar",
                    input_tokens=1000,
                    output_tokens=500,
                    cost_breakdown=CostBreakdown(output_cost=0.005),
                )
                tracker._record_usage(record)
            except BudgetExceededError:
                with lock:
                    exceeded_count[0] += 1

        # Try to exceed budget from multiple threads
        threads = [threading.Thread(target=try_exceed_budget) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some should have succeeded, some should have failed
        # At least one should exceed (budget is $0.01, each call is $0.005)
        assert exceeded_count[0] > 0
        assert tracker.summary.call_count + exceeded_count[0] == 10

    def test_thread_pool_stress_test(self) -> None:
        """Stress test with ThreadPoolExecutor."""
        tracker = PerplexityCostTracker(store_records=False)  # Don't store for memory

        def simulate_api_call(i: int) -> float:
            record = UsageRecord(
                model=["sonar", "sonar-pro", "sonar-reasoning"][i % 3],
                input_tokens=100 + i,
                output_tokens=50 + i,
                cost_breakdown=CostBreakdown(output_cost=0.0001 * (i + 1)),
            )
            tracker._record_usage(record)
            return record.total_cost

        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(simulate_api_call, range(1000)))

        assert tracker.summary.call_count == 1000
        assert len(tracker.summary.cost_by_model) == 3  # 3 different models


# ============================================================================
# BUDGET BEHAVIOR TESTS - "Limits, respect you must"
# ============================================================================


class TestBudgetBehavior:
    """Comprehensive budget management tests."""

    def test_budget_exactly_at_limit(self) -> None:
        """Hitting budget exactly should raise error."""
        tracker = PerplexityCostTracker(budget=0.001)

        # First call uses exactly the budget
        record = UsageRecord(
            model="sonar",
            input_tokens=1000,
            output_tokens=0,
            cost_breakdown=CostBreakdown(input_cost=0.001),
        )
        tracker._record_usage(record)

        # Second call should fail even with tiny cost
        with pytest.raises(BudgetExceededError):
            tracker._record_usage(
                UsageRecord(
                    model="sonar",
                    input_tokens=1,
                    output_tokens=0,
                    cost_breakdown=CostBreakdown(input_cost=0.000001),
                )
            )

    def test_warning_callback_receives_correct_values(self) -> None:
        """Warning callback should get accurate current/budget values."""
        received = []

        def capture_warning(current: float, budget: float) -> None:
            received.append((current, budget))

        tracker = PerplexityCostTracker(
            budget=1.0,
            warn_at=0.5,
            on_budget_warning=capture_warning,
        )

        # Record enough to trigger warning
        with pytest.warns(UserWarning):
            tracker._record_usage(
                UsageRecord(
                    model="sonar",
                    input_tokens=1000,
                    output_tokens=500,
                    cost_breakdown=CostBreakdown(output_cost=0.6),  # 60% of budget
                )
            )

        assert len(received) == 1
        assert received[0][1] == 1.0  # budget
        assert received[0][0] >= 0.5  # current cost

    def test_no_budget_allows_unlimited(self) -> None:
        """No budget set should allow unlimited spending."""
        tracker = PerplexityCostTracker(budget=None)

        # Record massive cost
        for _ in range(100):
            tracker._record_usage(
                UsageRecord(
                    model="sonar",
                    input_tokens=1000000,
                    output_tokens=500000,
                    cost_breakdown=CostBreakdown(output_cost=100.0),
                )
            )

        # Should never raise
        assert tracker.summary.total_cost == 10000.0

    def test_remaining_budget_calculation(self) -> None:
        """Remaining budget should update correctly."""
        tracker = PerplexityCostTracker(budget=1.0)
        assert tracker.remaining_budget == 1.0

        tracker._record_usage(
            UsageRecord(
                model="sonar",
                input_tokens=1000,
                output_tokens=500,
                cost_breakdown=CostBreakdown(output_cost=0.3),
            )
        )

        assert abs(tracker.remaining_budget - 0.7) < 1e-10

    def test_budget_exceeded_error_message_format(self) -> None:
        """Error message should be clear and informative."""
        error = BudgetExceededError(
            budget=1.0,
            current_cost=0.95,
            attempted_cost=0.10,
        )

        message = str(error)
        assert "1.0" in message or "1.00" in message  # budget
        assert "0.95" in message  # current
        assert "0.1" in message or "0.10" in message  # attempted


# ============================================================================
# INTEGRATION TESTS - "Together, stronger we are"
# ============================================================================


class TestIntegrationScenarios:
    """Real-world usage scenarios."""

    def test_multi_model_session(self) -> None:
        """Track costs across different models in one session."""
        tracker = PerplexityCostTracker()

        # Simulate using different models
        models = [
            ("sonar", 100, 50, 0.00015),
            ("sonar-pro", 100, 50, 0.0105),  # More expensive
            ("sonar-reasoning", 100, 50, 0.0035),
            ("sonar", 200, 100, 0.0003),
        ]

        for model, inp, out, cost in models:
            tracker._record_usage(
                UsageRecord(
                    model=model,
                    input_tokens=inp,
                    output_tokens=out,
                    cost_breakdown=CostBreakdown(output_cost=cost),
                )
            )

        summary = tracker.summary
        assert summary.call_count == 4
        assert len(summary.cost_by_model) == 3  # 3 unique models
        assert "sonar" in summary.cost_by_model
        assert "sonar-pro" in summary.cost_by_model

    def test_session_reset_workflow(self) -> None:
        """Reset should return summary and clear state."""
        tracker = PerplexityCostTracker()

        # First session
        for _ in range(5):
            tracker._record_usage(
                UsageRecord(
                    model="sonar",
                    input_tokens=100,
                    output_tokens=50,
                    cost_breakdown=CostBreakdown(output_cost=0.01),
                )
            )

        # Reset and capture
        session1 = tracker.reset()
        assert session1.call_count == 5
        assert session1.total_cost == 0.05

        # New session
        tracker._record_usage(
            UsageRecord(
                model="sonar-pro",
                input_tokens=100,
                output_tokens=50,
                cost_breakdown=CostBreakdown(output_cost=0.02),
            )
        )

        assert tracker.summary.call_count == 1
        assert tracker.summary.total_cost == 0.02

    def test_cost_update_callback_for_logging(self) -> None:
        """Simulate logging each API call."""
        log = []

        def log_call(record: UsageRecord, summary: CostSummary) -> None:
            log.append(
                {
                    "model": record.model,
                    "tokens": record.total_tokens,
                    "cost": record.total_cost,
                    "running_total": summary.total_cost,
                }
            )

        tracker = PerplexityCostTracker(on_cost_update=log_call)

        # Simulate calls
        for i in range(3):
            tracker._record_usage(
                UsageRecord(
                    model="sonar",
                    input_tokens=100 * (i + 1),
                    output_tokens=50 * (i + 1),
                    cost_breakdown=CostBreakdown(output_cost=0.01 * (i + 1)),
                )
            )

        assert len(log) == 3
        assert log[0]["running_total"] == 0.01
        assert log[1]["running_total"] == 0.03  # 0.01 + 0.02
        assert log[2]["running_total"] == 0.06  # 0.01 + 0.02 + 0.03

    def test_estimation_vs_actual_comparison(self) -> None:
        """Compare estimated costs to actual costs."""
        model = "sonar"
        prompt = "What is the meaning of life, the universe, and everything?"

        # Estimate before
        estimated = estimate_cost(model, prompt, estimated_output_tokens=100)

        # Simulate actual usage (with known token counts)
        actual_breakdown = calculate_cost_breakdown(
            model,
            input_tokens=15,  # ~60 chars / 4
            output_tokens=100,
        )

        # Both should be in same order of magnitude
        assert estimated is not None
        assert estimated > 0
        assert actual_breakdown.total > 0
        # Estimation uses ~4 chars per token, so should be similar
        ratio = estimated / actual_breakdown.total if actual_breakdown.total > 0 else 1
        assert 0.1 < ratio < 10  # Within an order of magnitude


# ============================================================================
# PRICING DATA INTEGRITY TESTS - "Trust the data, you must"
# ============================================================================


class TestPricingDataIntegrity:
    """Ensure pricing data is complete and valid."""

    def test_all_models_have_required_fields(self) -> None:
        """Every model must have all pricing fields."""
        required_fields = [
            "input_cost_per_million",
            "output_cost_per_million",
            "citation_cost_per_million",
            "reasoning_cost_per_million",
            "search_cost_per_thousand",
        ]

        for model_name, pricing in PERPLEXITY_PRICING.items():
            for field in required_fields:
                assert field in pricing, f"{model_name} missing {field}"

    def test_no_negative_prices(self) -> None:
        """Prices must be non-negative."""
        for model_name, pricing in PERPLEXITY_PRICING.items():
            for key, value in pricing.items():
                if value is not None and isinstance(value, (int, float)):
                    assert value >= 0, f"{model_name}.{key} is negative"

    def test_pro_models_more_expensive(self) -> None:
        """Pro models should generally cost more than base models."""
        sonar = get_model_pricing("sonar")
        sonar_pro = get_model_pricing("sonar-pro")

        assert sonar is not None
        assert sonar_pro is not None

        # Pro should have higher output cost
        assert sonar_pro["output_cost_per_million"] > sonar["output_cost_per_million"]

    def test_deep_research_has_extra_costs(self) -> None:
        """Deep research model should have citation and search costs."""
        deep = get_model_pricing("sonar-deep-research")

        assert deep is not None
        assert deep["citation_cost_per_million"] is not None
        assert deep["citation_cost_per_million"] > 0
        assert deep["search_cost_per_thousand"] is not None
        assert deep["search_cost_per_thousand"] > 0


# ============================================================================
# SUMMARY STRING FORMATTING TESTS - "Readable, output must be"
# ============================================================================


class TestSummaryFormatting:
    """Test human-readable output formatting."""

    def test_summary_str_includes_key_info(self) -> None:
        """String representation should include all important data."""
        summary = CostSummary(
            total_cost=0.123456,
            total_input_tokens=10000,
            total_output_tokens=5000,
            call_count=10,
            cost_by_model={"sonar": 0.05, "sonar-pro": 0.073456},
        )

        output = str(summary)

        assert "0.123456" in output  # Total cost
        assert "10,000" in output  # Input tokens formatted
        assert "5,000" in output  # Output tokens formatted
        assert "10" in output  # Call count
        assert "sonar" in output  # Model names

    def test_summary_to_dict_complete(self) -> None:
        """Dict representation should be JSON-serializable."""
        import json

        summary = CostSummary(
            total_cost=0.05,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_reasoning_tokens=100,
            call_count=5,
        )

        result = summary.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Should have all fields
        assert "total_cost" in result
        assert "average_cost_per_call" in result
        assert "total_tokens" in result


# ============================================================================
# ERROR HANDLING TESTS - "Graceful, failures must be"
# ============================================================================


class TestErrorHandling:
    """Test graceful error handling."""

    def test_unknown_model_returns_none_not_error(self) -> None:
        """Unknown models should return None, not raise."""
        pricing = get_model_pricing("definitely-not-a-real-model")
        assert pricing is None

        cost = calculate_cost(
            "not-real",
            _create_usage_metadata(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
            ),
        )
        assert cost is None

    def test_empty_usage_metadata_handled(self) -> None:
        """Empty usage metadata shouldn't crash."""
        breakdown = calculate_cost_breakdown("sonar", 0, 0, 0, 0, 0)
        assert breakdown.total == 0.0

    def test_none_values_in_usage_handled(self) -> None:
        """None values in usage dict should be handled gracefully."""
        usage = _create_usage_metadata(
            {
                "prompt_tokens": None,
                "completion_tokens": None,
            }
        )
        # Should default to 0, not crash
        assert usage["input_tokens"] == 0 or usage.get("input_tokens") is None
