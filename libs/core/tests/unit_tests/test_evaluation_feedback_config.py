"""Regression test for langchain-ai/langchain#31802.

Verify that EvaluationResult.feedback_config preserves dict fields
and does not silently drop unknown or partial dict keys.

The EvaluationResult class is defined in the langsmith package.
This test ensures langchain_core's usage of it preserves dict data.

Background: In older langsmith versions, Pydantic's TypedDict validation
on Union[FeedbackConfig, dict] would silently strip unknown keys from
dicts passed to feedback_config. The fix is in langsmith-sdk PR #2718.
"""

from __future__ import annotations

from typing import Any

from langsmith.evaluation.evaluator import EvaluationResult


class TestEvaluationResultFeedbackConfig:
    """Tests for EvaluationResult.feedback_config dict preservation.

    See: https://github.com/langchain-ai/langchain/issues/31802
    See: https://github.com/langchain-ai/langsmith-sdk/pull/2718
    """

    def test_feedback_config_preserves_unknown_keys(self) -> None:
        """Unknown dict keys should be preserved, not silently dropped."""
        config: dict[str, Any] = {
            "custom_key": "custom_value",
            "another_key": 123,
        }
        result = EvaluationResult(
            key="test",
            score=1.0,
            feedback_config=config,
        )
        assert result.feedback_config is not None
        assert result.feedback_config["custom_key"] == "custom_value"
        assert result.feedback_config["another_key"] == 123

    def test_feedback_config_preserves_mixed_keys(self) -> None:
        """Known FeedbackConfig keys mixed with unknown keys should all be preserved."""
        config: dict[str, Any] = {
            "type": "continuous",
            "min": 0,
            "max": 1,
            "custom_key": "custom_value",
        }
        result = EvaluationResult(
            key="test",
            score=1.0,
            feedback_config=config,
        )
        assert result.feedback_config is not None
        assert result.feedback_config["type"] == "continuous"
        assert result.feedback_config["min"] == 0
        assert result.feedback_config["max"] == 1
        assert result.feedback_config["custom_key"] == "custom_value"

    def test_feedback_config_empty_dict(self) -> None:
        """An empty dict should be preserved as-is."""
        result = EvaluationResult(
            key="test",
            score=1.0,
            feedback_config={},
        )
        assert result.feedback_config == {}

    def test_feedback_config_none_default(self) -> None:
        """When not provided, feedback_config should be None."""
        result = EvaluationResult(key="test", score=1.0)
        assert result.feedback_config is None

    def test_feedback_config_nested_dict(self) -> None:
        """Nested dict structures should be preserved."""
        config: dict[str, Any] = {
            "nested": {"a": 1, "b": [2, 3]},
            "flat": "value",
        }
        result = EvaluationResult(
            key="test",
            score=1.0,
            feedback_config=config,
        )
        assert result.feedback_config is not None
        assert result.feedback_config["nested"] == {"a": 1, "b": [2, 3]}
        assert result.feedback_config["flat"] == "value"
