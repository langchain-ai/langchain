"""Tests for EvaluatorCallbackHandler feedback_config forwarding."""

from unittest.mock import MagicMock
from uuid import UUID

from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """Test that feedback_config is forwarded to create_feedback.

    Regression test for: https://github.com/langchain-ai/langchain/issues/31802
    """
    # Create a mock client
    mock_client = MagicMock()
    mock_client.create_feedback = MagicMock()

    # Create handler with mock client
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
    )

    # Create a mock run
    mock_run = MagicMock(spec=Run)
    mock_run.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_run.reference_example_id = None
    mock_run.outputs = {"result": "test"}

    # Create evaluation result with feedback_config
    feedback_config = {"type": "continuous", "min": 0, "max": 1}
    eval_result = EvaluationResult(
        key="test-key",
        value="test-value",
        score=0.5,
        feedback_config=feedback_config,
    )

    # Call the method under test
    handler._log_evaluation_feedback(eval_result, mock_run)

    # Verify create_feedback was called with feedback_config
    mock_client.create_feedback.assert_called_once()
    call_args = mock_client.create_feedback.call_args[0]
    call_kwargs = mock_client.create_feedback.call_args[1]
    assert call_kwargs["feedback_config"] == feedback_config
    assert call_args[1] == "test-key"  # key is positional
    assert call_kwargs["value"] == "test-value"
    assert call_kwargs["score"] == 0.5


def test_log_evaluation_feedback_handles_none_feedback_config() -> None:
    """Test that None feedback_config is handled correctly."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.create_feedback = MagicMock()

    # Create handler with mock client
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
    )

    # Create a mock run
    mock_run = MagicMock(spec=Run)
    mock_run.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_run.reference_example_id = None
    mock_run.outputs = {"result": "test"}

    # Create evaluation result without feedback_config
    eval_result = EvaluationResult(
        key="test-key",
        value="test-value",
        score=0.5,
    )

    # Call the method under test
    handler._log_evaluation_feedback(eval_result, mock_run)

    # Verify create_feedback was called with feedback_config=None
    mock_client.create_feedback.assert_called_once()
    call_args = mock_client.create_feedback.call_args[0]
    call_kwargs = mock_client.create_feedback.call_args[1]
    assert call_kwargs["feedback_config"] is None
    assert call_args[1] == "test-key"  # key is positional
    assert call_kwargs["value"] == "test-value"
    assert call_kwargs["score"] == 0.5
