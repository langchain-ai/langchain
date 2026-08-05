"""Tests for EvaluatorCallbackHandler."""

from __future__ import annotations

from unittest.mock import MagicMock

from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run() -> Run:
    """Create a minimal Run for evaluator tests."""
    return Run(
        name="test-run",
        run_type="chain",
        outputs={"output": "test"},
    )


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """`feedback_config` from the EvaluationResult is passed to create_feedback."""
    client = MagicMock()
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)
    run = _make_run()
    result = EvaluationResult(
        key="sentiment",
        value="positive",
        feedback_config={"criteria": "sentiment", "threshold": 0.5},
    )

    handler._log_evaluation_feedback(result, run)

    client.create_feedback.assert_called_once()
    kwargs = client.create_feedback.call_args.kwargs
    assert kwargs["feedback_config"] == {
        "criteria": "sentiment",
        "threshold": 0.5,
    }


def test_log_evaluation_feedback_none_feedback_config() -> None:
    """A missing `feedback_config` forwards `None` (no-op, no TypeError)."""
    client = MagicMock()
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)
    run = _make_run()
    result = EvaluationResult(key="sentiment", value="positive")

    handler._log_evaluation_feedback(result, run)

    client.create_feedback.assert_called_once()
    assert client.create_feedback.call_args.kwargs["feedback_config"] is None
