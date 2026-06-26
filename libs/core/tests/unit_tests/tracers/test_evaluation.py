"""Unit tests for EvaluatorCallbackHandler feedback forwarding."""

from __future__ import annotations

import unittest.mock
import uuid
from typing import Any
from uuid import UUID

from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run(run_id: UUID | None = None) -> Run:
    """Create a minimal Run object for testing."""
    return Run(
        id=run_id or uuid.uuid4(),
        name="test_run",
        run_type="chain",
        inputs={},
        outputs={"result": "ok"},
        extra={},
        serialized={},
        events=[],
        tags=[],
    )


def _make_handler(client: Any) -> EvaluatorCallbackHandler:
    """Create an EvaluatorCallbackHandler with the given mock client."""
    return EvaluatorCallbackHandler(evaluators=[], client=client)


class TestLogEvaluationFeedbackForwardsFeedbackConfig:
    """Verify that feedback_config is forwarded to client.create_feedback."""

    def test_feedback_config_forwarded_when_set(self) -> None:
        """feedback_config on EvaluationResult is passed to create_feedback."""
        client = unittest.mock.MagicMock(spec=Client)
        handler = _make_handler(client)
        run = _make_run()

        feedback_config = {"type": "continuous", "min": 0, "max": 1}
        result = EvaluationResult(
            key="score",
            score=0.9,
            feedback_config=feedback_config,
        )

        handler._log_evaluation_feedback(result, run)

        client.create_feedback.assert_called_once()
        _, kwargs = client.create_feedback.call_args
        assert kwargs["feedback_config"] == feedback_config

    def test_feedback_config_none_when_not_set(self) -> None:
        """create_feedback receives feedback_config=None when not supplied."""
        client = unittest.mock.MagicMock(spec=Client)
        handler = _make_handler(client)
        run = _make_run()

        result = EvaluationResult(key="score", score=0.5)

        handler._log_evaluation_feedback(result, run)

        _, kwargs = client.create_feedback.call_args
        assert kwargs["feedback_config"] is None

    def test_feedback_config_categorical(self) -> None:
        """Categorical feedback_config is forwarded intact."""
        client = unittest.mock.MagicMock(spec=Client)
        handler = _make_handler(client)
        run = _make_run()

        feedback_config = {
            "type": "categorical",
            "categories": [{"value": "good"}, {"value": "bad"}],
        }
        result = EvaluationResult(
            key="quality",
            value="good",
            feedback_config=feedback_config,
        )

        handler._log_evaluation_feedback(result, run)

        _, kwargs = client.create_feedback.call_args
        assert kwargs["feedback_config"] == feedback_config

    def test_other_fields_still_forwarded(self) -> None:
        """Adding feedback_config does not break forwarding of other fields."""
        client = unittest.mock.MagicMock(spec=Client)
        handler = _make_handler(client)
        run = _make_run()

        result = EvaluationResult(
            key="relevance",
            score=0.7,
            value="relevant",
            comment="looks good",
            feedback_config={"type": "continuous", "min": 0, "max": 1},
        )

        handler._log_evaluation_feedback(result, run)

        _, kwargs = client.create_feedback.call_args
        assert kwargs["score"] == 0.7
        assert kwargs["value"] == "relevant"
        assert kwargs["comment"] == "looks good"
        assert kwargs["feedback_config"] == {"type": "continuous", "min": 0, "max": 1}
