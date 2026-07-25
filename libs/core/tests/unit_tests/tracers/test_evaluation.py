"""Tests for EvaluatorCallbackHandler feedback_config handling."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def test_evaluation_result_preserves_feedback_config_dict() -> None:
    """Regression for silent drop of feedback_config dict fields.

    https://github.com/langchain-ai/langchain/issues/31802
    """
    feedback_config = {"threshold": 1.0}
    result = EvaluationResult(
        key="sentiment",
        value="positive",
        feedback_config=feedback_config,
    )
    assert result.feedback_config == feedback_config

    mixed = {"type": "continuous", "threshold": 1.0, "min": 0.0}
    result_mixed = EvaluationResult(
        key="sentiment",
        value="positive",
        feedback_config=mixed,
    )
    assert result_mixed.feedback_config == mixed


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """Test that feedback_config is forwarded to create_feedback.

    Regression test for: https://github.com/langchain-ai/langchain/issues/31802
    """
    mock_client = MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
    )

    mock_run = MagicMock(spec=Run)
    mock_run.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_run.reference_example_id = None
    mock_run.outputs = {"result": "test"}

    feedback_config = {"type": "continuous", "min": 0, "max": 1}
    eval_result = EvaluationResult(
        key="test-key",
        value="test-value",
        score=0.5,
        feedback_config=feedback_config,
    )

    handler._log_evaluation_feedback(eval_result, mock_run)

    mock_client.create_feedback.assert_called_once()
    call_args = mock_client.create_feedback.call_args[0]
    call_kwargs = mock_client.create_feedback.call_args[1]
    assert call_kwargs["feedback_config"] == feedback_config
    assert call_args[1] == "test-key"
    assert call_kwargs["value"] == "test-value"
    assert call_kwargs["score"] == 0.5


def test_log_evaluation_feedback_handles_none_feedback_config() -> None:
    """Test that None custom feedback_config is still forwarded."""
    mock_client = MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
    )

    mock_run = MagicMock(spec=Run)
    mock_run.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_run.reference_example_id = None
    mock_run.outputs = {"result": "test"}

    eval_result = EvaluationResult(
        key="test-key",
        value="test-value",
        score=0.5,
    )

    handler._log_evaluation_feedback(eval_result, mock_run)

    mock_client.create_feedback.assert_called_once()
    call_kwargs = mock_client.create_feedback.call_args[1]
    assert call_kwargs["feedback_config"] is None
    assert call_kwargs["value"] == "test-value"
    assert call_kwargs["score"] == 0.5


def test_log_evaluation_feedback_forwards_categorical_feedback_config() -> None:
    """Categorical feedback_config should be forwarded intact."""
    mock_client = MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
    )

    mock_run = MagicMock(spec=Run)
    mock_run.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_run.reference_example_id = None

    feedback_config = {
        "type": "categorical",
        "categories": [{"value": "positive"}, {"value": "negative"}],
    }
    eval_result = EvaluationResult(
        key="sentiment",
        value="positive",
        feedback_config=feedback_config,
    )

    handler._log_evaluation_feedback(eval_result, mock_run)

    call_kwargs = mock_client.create_feedback.call_args[1]
    assert call_kwargs["feedback_config"] == feedback_config
