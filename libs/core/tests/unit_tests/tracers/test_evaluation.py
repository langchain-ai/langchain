"""Tests for the EvaluatorCallbackHandler tracer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run() -> Run:
    """Build a minimal completed Run for testing."""
    now = datetime.now(timezone.utc)
    return Run(
        id=uuid4(),
        name="test_run",
        run_type="chain",
        start_time=now,
        end_time=now,
        inputs={},
        outputs={"result": "ok"},
    )


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """`feedback_config` on EvaluationResult must be forwarded to create_feedback.

    Regression test for langchain-ai/langchain#31802. Previously the callback
    handler dropped `feedback_config` when calling `client.create_feedback`,
    silently discarding any user-supplied feedback configuration.
    """
    client = MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    feedback_config: dict[str, Any] = {
        "type": "continuous",
        "min": 0,
        "max": 1,
    }
    eval_result = EvaluationResult(
        key="sentiment",
        score=0.9,
        feedback_config=feedback_config,
    )
    run = _make_run()

    handler._log_evaluation_feedback(eval_result, run)

    client.create_feedback.assert_called_once()
    _, kwargs = client.create_feedback.call_args
    assert kwargs["feedback_config"] == feedback_config


def test_log_evaluation_feedback_forwards_none_feedback_config() -> None:
    """When feedback_config is None it should still be forwarded as None."""
    client = MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    eval_result = EvaluationResult(key="quality", score=0.5)
    run = _make_run()

    handler._log_evaluation_feedback(eval_result, run)

    client.create_feedback.assert_called_once()
    _, kwargs = client.create_feedback.call_args
    assert kwargs["feedback_config"] is None
