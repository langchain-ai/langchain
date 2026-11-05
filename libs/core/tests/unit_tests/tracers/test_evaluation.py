"""Tests for EvaluatorCallbackHandler feedback logging."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run() -> Run:
    return Run(
        id=uuid4(),
        name="test",
        start_time=datetime.now(timezone.utc),
        run_type="chain",
        trace_id=uuid4(),
        dotted_order="20250101T000000000000Z" + str(uuid4()),
    )


def _log_with_mock_client(result: EvaluationResult) -> MagicMock:
    client = MagicMock()
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)
    handler._log_evaluation_feedback(result, _make_run())
    return client


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """A valid feedback_config on the result must reach create_feedback."""
    config = {"type": "continuous", "min": 0, "max": 1}
    result = EvaluationResult(key="sentiment", score=0.9, feedback_config=config)
    client = _log_with_mock_client(result)

    client.create_feedback.assert_called_once()
    assert client.create_feedback.call_args.kwargs["feedback_config"] == config


def test_log_evaluation_feedback_forwards_none_feedback_config() -> None:
    """A result without feedback_config must not change the call."""
    result = EvaluationResult(key="sentiment", score=0.9)
    client = _log_with_mock_client(result)

    client.create_feedback.assert_called_once()
    assert client.create_feedback.call_args.kwargs["feedback_config"] is None


def test_log_evaluation_feedback_forwards_arbitrary_dict_unchanged() -> None:
    """A dict config is forwarded unchanged; LangSmith owns validation."""
    config = {"threshold": 1.0}
    result = EvaluationResult(key="sentiment", value="positive", feedback_config=config)
    client = _log_with_mock_client(result)

    client.create_feedback.assert_called_once()
    assert client.create_feedback.call_args.kwargs["feedback_config"] == config
