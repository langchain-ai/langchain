"""Tests for cost tracking functionality."""

import pytest

from langchain_perplexity import (
    BudgetExceededError,
    CostBreakdown,
    CostSummary,
    PerplexityCostTracker,
    UsageRecord,
    calculate_cost_breakdown,
    estimate_cost,
    format_cost,
)


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_total_calculation(self) -> None:
        """Test that total correctly sums all cost components."""
        breakdown = CostBreakdown(
            input_cost=0.001,
            output_cost=0.002,
            citation_cost=0.003,
            reasoning_cost=0.004,
            search_cost=0.005,
        )
        assert abs(breakdown.total - 0.015) < 1e-10

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        breakdown = CostBreakdown(input_cost=0.001, output_cost=0.002)
        result = breakdown.to_dict()

        assert result["input_cost"] == 0.001
        assert result["output_cost"] == 0.002
        assert result["citation_cost"] == 0.0
        assert result["reasoning_cost"] == 0.0
        assert result["search_cost"] == 0.0
        assert abs(result["total_cost"] - 0.003) < 1e-10

    def test_default_values(self) -> None:
        """Test default zero values."""
        breakdown = CostBreakdown()
        assert breakdown.total == 0.0


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_total_tokens(self) -> None:
        """Test total token calculation."""
        record = UsageRecord(
            model="sonar",
            input_tokens=100,
            output_tokens=50,
        )
        assert record.total_tokens == 150

    def test_total_cost(self) -> None:
        """Test total cost from breakdown."""
        record = UsageRecord(
            model="sonar",
            input_tokens=100,
            output_tokens=50,
            cost_breakdown=CostBreakdown(input_cost=0.01, output_cost=0.02),
        )
        assert abs(record.total_cost - 0.03) < 1e-10


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_average_cost_per_call(self) -> None:
        """Test average cost calculation."""
        summary = CostSummary(total_cost=0.10, call_count=5)
        assert abs(summary.average_cost_per_call - 0.02) < 1e-10

    def test_average_cost_zero_calls(self) -> None:
        """Test average cost with no calls."""
        summary = CostSummary()
        assert summary.average_cost_per_call == 0.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        summary = CostSummary(
            total_cost=0.05,
            total_input_tokens=1000,
            total_output_tokens=500,
            call_count=2,
        )
        result = summary.to_dict()

        assert result["total_cost"] == 0.05
        assert result["total_tokens"] == 1500
        assert result["call_count"] == 2
        assert abs(result["average_cost_per_call"] - 0.025) < 1e-10

    def test_str_representation(self) -> None:
        """Test human-readable string output."""
        summary = CostSummary(
            total_cost=0.0123,
            total_input_tokens=1000,
            total_output_tokens=500,
            call_count=3,
        )
        output = str(summary)

        assert "Perplexity Cost Summary" in output
        assert "$0.012300" in output
        assert "3" in output
        assert "1,000" in output


class TestCalculateCostBreakdown:
    """Tests for calculate_cost_breakdown function."""

    def test_sonar_basic(self) -> None:
        """Test cost breakdown for sonar model."""
        breakdown = calculate_cost_breakdown(
            model_name="sonar",
            input_tokens=1000,
            output_tokens=500,
        )

        # sonar: $1/1M input, $1/1M output
        assert abs(breakdown.input_cost - 0.001) < 1e-10
        assert abs(breakdown.output_cost - 0.0005) < 1e-10
        assert breakdown.citation_cost == 0.0
        assert breakdown.reasoning_cost == 0.0
        assert breakdown.search_cost == 0.0

    def test_sonar_pro(self) -> None:
        """Test cost breakdown for sonar-pro model."""
        breakdown = calculate_cost_breakdown(
            model_name="sonar-pro",
            input_tokens=1000,
            output_tokens=500,
        )

        # sonar-pro: $3/1M input, $15/1M output
        assert abs(breakdown.input_cost - 0.003) < 1e-10
        assert abs(breakdown.output_cost - 0.0075) < 1e-10

    def test_deep_research_all_costs(self) -> None:
        """Test cost breakdown for deep research with all cost types."""
        breakdown = calculate_cost_breakdown(
            model_name="sonar-deep-research",
            input_tokens=1000,
            output_tokens=5000,
            reasoning_tokens=10000,
            citation_tokens=20000,
            num_search_queries=10,
        )

        # deep-research: $2/1M input, $8/1M output, $3/1M reasoning,
        # $2/1M citation, $5/1K search
        assert abs(breakdown.input_cost - 0.002) < 1e-10
        assert abs(breakdown.output_cost - 0.04) < 1e-10
        assert abs(breakdown.reasoning_cost - 0.03) < 1e-10
        assert abs(breakdown.citation_cost - 0.04) < 1e-10
        assert abs(breakdown.search_cost - 0.05) < 1e-10

    def test_unknown_model(self) -> None:
        """Test that unknown models return empty breakdown."""
        breakdown = calculate_cost_breakdown(
            model_name="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert breakdown.total == 0.0


class TestEstimateCost:
    """Tests for estimate_cost function."""

    def test_estimate_from_string(self) -> None:
        """Test cost estimation from string prompt."""
        # ~100 chars = ~25 tokens input
        prompt = "What is the capital of France? Please provide details."
        cost = estimate_cost("sonar", prompt, estimated_output_tokens=100)

        assert cost is not None
        assert cost > 0

    def test_estimate_unknown_model(self) -> None:
        """Test that unknown models return None."""
        cost = estimate_cost("unknown-model", "Hello")
        assert cost is None

    def test_estimate_with_different_models(self) -> None:
        """Test that different models have different costs."""
        prompt = "Hello world"

        sonar_cost = estimate_cost("sonar", prompt)
        sonar_pro_cost = estimate_cost("sonar-pro", prompt)

        assert sonar_cost is not None
        assert sonar_pro_cost is not None
        assert sonar_pro_cost > sonar_cost  # sonar-pro is more expensive


class TestFormatCost:
    """Tests for format_cost utility."""

    def test_default_precision(self) -> None:
        """Test default 4 decimal places."""
        assert format_cost(0.0015) == "$0.0015"

    def test_custom_precision(self) -> None:
        """Test custom precision."""
        assert format_cost(0.123456, precision=2) == "$0.12"
        assert format_cost(0.123456, precision=6) == "$0.123456"

    def test_zero_cost(self) -> None:
        """Test zero cost formatting."""
        assert format_cost(0.0) == "$0.0000"


class TestPerplexityCostTracker:
    """Tests for PerplexityCostTracker callback handler."""

    def test_initial_state(self) -> None:
        """Test tracker initializes with zero values."""
        tracker = PerplexityCostTracker()

        assert tracker.total_cost == 0.0
        assert tracker.summary.call_count == 0
        assert tracker.remaining_budget is None

    def test_with_budget(self) -> None:
        """Test tracker with budget set."""
        tracker = PerplexityCostTracker(budget=1.0)

        assert tracker.budget == 1.0
        assert tracker.remaining_budget == 1.0

    def test_reset(self) -> None:
        """Test resetting tracker returns summary and clears state."""
        tracker = PerplexityCostTracker()
        tracker._summary.total_cost = 0.05
        tracker._summary.call_count = 5

        final_summary = tracker.reset()

        assert final_summary.total_cost == 0.05
        assert final_summary.call_count == 5
        assert tracker.total_cost == 0.0
        assert tracker.summary.call_count == 0

    def test_budget_exceeded_error(self) -> None:
        """Test BudgetExceededError contains relevant info."""
        error = BudgetExceededError(budget=1.0, current_cost=0.8, attempted_cost=0.3)

        assert error.budget == 1.0
        assert error.current_cost == 0.8
        assert error.attempted_cost == 0.3
        assert "exceeded" in str(error).lower()

    def test_record_usage_updates_summary(self) -> None:
        """Test that recording usage updates the summary correctly."""
        tracker = PerplexityCostTracker()

        record = UsageRecord(
            model="sonar",
            input_tokens=1000,
            output_tokens=500,
            cost_breakdown=CostBreakdown(input_cost=0.001, output_cost=0.0005),
        )
        tracker._record_usage(record)

        assert tracker.summary.call_count == 1
        assert tracker.summary.total_input_tokens == 1000
        assert tracker.summary.total_output_tokens == 500
        assert abs(tracker.summary.total_cost - 0.0015) < 1e-10
        assert "sonar" in tracker.summary.cost_by_model

    def test_budget_check_raises_on_exceed(self) -> None:
        """Test that exceeding budget raises error."""
        tracker = PerplexityCostTracker(budget=0.01)
        tracker._summary.total_cost = 0.008

        with pytest.raises(BudgetExceededError):
            tracker._check_budget(estimated_cost=0.005)

    def test_budget_warning_triggered(self) -> None:
        """Test that budget warning is triggered at threshold."""
        warning_called = []

        def on_warning(current: float, budget: float) -> None:
            warning_called.append((current, budget))

        tracker = PerplexityCostTracker(
            budget=1.0,
            warn_at=0.5,
            on_budget_warning=on_warning,
        )

        # First check below threshold - no warning
        tracker._check_budget(0.4)
        assert len(warning_called) == 0

        # Check at threshold - warning triggered
        with pytest.warns(UserWarning):
            tracker._check_budget(0.6)
        assert len(warning_called) == 1

        # Second check - no duplicate warning
        tracker._check_budget(0.7)
        assert len(warning_called) == 1

    def test_cost_update_callback(self) -> None:
        """Test that cost update callback is called."""
        updates = []

        def on_update(record: UsageRecord, summary: CostSummary) -> None:
            updates.append((record.model, summary.total_cost))

        tracker = PerplexityCostTracker(on_cost_update=on_update)

        record = UsageRecord(
            model="sonar",
            input_tokens=100,
            output_tokens=50,
            cost_breakdown=CostBreakdown(input_cost=0.001),
        )
        tracker._record_usage(record)

        assert len(updates) == 1
        assert updates[0][0] == "sonar"

    def test_store_records_disabled(self) -> None:
        """Test that records are not stored when disabled."""
        tracker = PerplexityCostTracker(store_records=False)

        record = UsageRecord(
            model="sonar",
            input_tokens=100,
            output_tokens=50,
        )
        tracker._record_usage(record)

        assert len(tracker.summary.records) == 0
        assert tracker.summary.call_count == 1

    def test_multiple_models_tracked_separately(self) -> None:
        """Test that costs are tracked per model."""
        tracker = PerplexityCostTracker()

        # Record sonar usage
        tracker._record_usage(
            UsageRecord(
                model="sonar",
                input_tokens=100,
                output_tokens=50,
                cost_breakdown=CostBreakdown(output_cost=0.01),
            )
        )

        # Record sonar-pro usage
        tracker._record_usage(
            UsageRecord(
                model="sonar-pro",
                input_tokens=100,
                output_tokens=50,
                cost_breakdown=CostBreakdown(output_cost=0.02),
            )
        )

        assert len(tracker.summary.cost_by_model) == 2
        assert abs(tracker.summary.cost_by_model["sonar"] - 0.01) < 1e-10
        assert abs(tracker.summary.cost_by_model["sonar-pro"] - 0.02) < 1e-10


class TestIntegration:
    """Integration tests for the full cost tracking flow."""

    def test_full_tracking_workflow(self) -> None:
        """Test complete workflow from estimation to tracking."""
        # 1. Estimate cost before call
        estimated = estimate_cost("sonar", "What is AI?", estimated_output_tokens=100)
        assert estimated is not None

        # 2. Create tracker with budget based on estimate
        tracker = PerplexityCostTracker(budget=estimated * 10)

        # 3. Simulate API calls
        for _ in range(3):
            record = UsageRecord(
                model="sonar",
                input_tokens=50,
                output_tokens=100,
                cost_breakdown=calculate_cost_breakdown("sonar", 50, 100),
            )
            tracker._record_usage(record)

        # 4. Check summary
        assert tracker.summary.call_count == 3
        assert tracker.total_cost > 0
        remaining = tracker.remaining_budget
        assert remaining is not None and remaining > 0

        # 5. Reset and verify
        final = tracker.reset()
        assert final.call_count == 3
        assert tracker.summary.call_count == 0
